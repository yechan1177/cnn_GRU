from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config import SpatialConfig
from ..interfaces import SpatialEncoder
from ..schemas import FramePacket

logger = logging.getLogger(__name__)


class MockSpatialEncoder(SpatialEncoder):
    """경량 모델 교체를 위한 baseline spatial encoder."""

    def __init__(self, cfg: SpatialConfig) -> None:
        self._feature_dim = max(8, int(cfg.feature_dim))
        self._feature_keys = [f"feature_{idx}" for idx in range(self._feature_dim)]

    def encode(self, frame: FramePacket) -> list[float]:
        if not frame.pixels:
            return [0.0] * self._feature_dim

        pixel_count = len(frame.pixels)
        mean_val = sum(frame.pixels) / pixel_count
        min_val = min(frame.pixels)
        max_val = max(frame.pixels)
        spread = max_val - min_val
        variance = sum((value - mean_val) ** 2 for value in frame.pixels) / pixel_count

        features: list[float] = []
        for idx in range(self._feature_dim):
            src = frame.pixels[idx % pixel_count]
            harmonic = ((frame.frame_id + idx) % 11) / 10.0
            value = (
                (0.45 * src)
                + (0.25 * mean_val)
                + (0.15 * spread)
                + (0.10 * variance)
                + (0.05 * harmonic)
            )
            features.append(round(value, 6))

        return features

    def get_feature_keys(self) -> list[str]:
        return list(self._feature_keys)


class YOLOSpatialEncoder(SpatialEncoder):
    """YOLO 검출 결과를 고정 길이 특징 벡터로 변환하는 spatial encoder."""

    BASE_FEATURE_KEYS: list[str] = [
        "det_norm",
        "mean_conf",
        "max_conf",
        "mean_area",
        "area_var",
        "vehicle_ratio",
        "person_ratio",
        "bike_ratio",
        "vehicle_conf",
        "person_conf",
        "bike_conf",
        "roi_risk",
        "center_closeness",
        "roi_mean_area",
        "roi_count_norm",
        "roi_vertical_bias",
    ]

    def __init__(self, cfg: SpatialConfig) -> None:
        if not cfg.weights_path:
            raise ValueError("YOLOSpatialEncoder 사용 시 spatial.weights_path가 필요합니다.")

        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "YOLOSpatialEncoder 사용 시 ultralytics 설치가 필요합니다."
            ) from exc

        self._feature_dim = max(8, int(cfg.feature_dim))
        self._device = str(cfg.device)
        self._conf_threshold = max(0.01, min(0.95, float(cfg.conf_threshold)))
        self._max_det = max(1, int(cfg.max_det))
        self._input_size = max(64, int(cfg.input_size))
        self._weights_path = Path(cfg.weights_path)

        if not self._weights_path.exists():
            raise FileNotFoundError(f"YOLO 가중치 파일을 찾을 수 없습니다: {self._weights_path}")

        self._model = YOLO(str(self._weights_path))
        names = getattr(self._model.model, "names", None) or getattr(self._model, "names", None)
        if isinstance(names, dict):
            self._class_names = [str(name) for _, name in sorted(names.items())]
        elif isinstance(names, list):
            self._class_names = [str(name) for name in names]
        else:
            self._class_names = []
        self._class_count = len(self._class_names)
        self._feature_keys = self._fit_feature_keys(self.BASE_FEATURE_KEYS)
        self._mock_fallback = MockSpatialEncoder(cfg)
        self._warning_issued = False
        self._last_detection: dict[str, Any] = {
            "count": 0,
            "boxes": [],
            "orig_shape": None,
            "mean_conf": 0.0,
            "max_conf": 0.0,
        }

        logger.info(
            "YOLOSpatialEncoder 초기화: weights=%s, device=%s, class_count=%s",
            self._weights_path,
            self._device,
            self._class_count,
        )

    def encode(self, frame: FramePacket) -> list[float]:
        source: object | None = None
        if frame.raw_path:
            image_path = Path(frame.raw_path)
            if image_path.exists():
                source = str(image_path)
        if source is None and frame.image is not None:
            source = frame.image
        if source is None:
            self._reset_last_detection()
            return self._mock_fallback.encode(frame)

        try:
            results = self._model.predict(
                source=source,
                conf=self._conf_threshold,
                imgsz=self._input_size,
                device=self._device,
                max_det=self._max_det,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - 런타임 환경 의존
            if not self._warning_issued:
                logger.warning("YOLO 추론 실패로 mock fallback 사용: %s", exc)
                self._warning_issued = True
            self._reset_last_detection()
            return self._mock_fallback.encode(frame)

        if not results:
            self._reset_last_detection()
            return [0.0] * self._feature_dim

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            h, w = result.orig_shape
            self._last_detection = {
                "count": 0,
                "boxes": [],
                "orig_shape": [int(h), int(w)],
                "mean_conf": 0.0,
                "max_conf": 0.0,
            }
            return [0.0] * self._feature_dim

        counts = [0.0] * max(1, self._class_count)
        conf_sums = [0.0] * max(1, self._class_count)
        normalized_areas: list[float] = []
        roi_area_ratios: list[float] = []
        roi_center_scores: list[float] = []
        roi_vertical_scores: list[float] = []

        h, w = result.orig_shape
        image_area = max(1.0, float(h * w))
        roi_x1 = 0.25 * float(w)
        roi_x2 = 0.75 * float(w)
        roi_y1 = 0.20 * float(h)
        roi_y2 = 0.90 * float(h)

        cls_values = boxes.cls.detach().cpu().tolist()
        conf_values = boxes.conf.detach().cpu().tolist()
        xyxy_values = boxes.xyxy.detach().cpu().tolist()

        box_items: list[dict[str, Any]] = []
        for idx, cls_value in enumerate(cls_values):
            cls_id = int(cls_value)
            if cls_id < 0 or cls_id >= len(counts):
                continue

            confidence = float(conf_values[idx]) if idx < len(conf_values) else 0.0
            counts[cls_id] += 1.0
            conf_sums[cls_id] += confidence

            if idx < len(xyxy_values):
                x1, y1, x2, y2 = xyxy_values[idx]
                box_w = max(0.0, float(x2) - float(x1))
                box_h = max(0.0, float(y2) - float(y1))
                area_ratio = (box_w * box_h) / image_area
                normalized_areas.append(area_ratio)
                center_x = (float(x1) + float(x2)) / 2.0
                center_y = (float(y1) + float(y2)) / 2.0

                in_roi = roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2
                if in_roi:
                    roi_area_ratios.append(area_ratio)
                    center_dx = abs(center_x - (float(w) * 0.5)) / max(1.0, float(w) * 0.5)
                    center_score = max(0.0, 1.0 - center_dx)
                    roi_center_scores.append(center_score)
                    vertical_score = min(1.0, max(0.0, center_y / max(1.0, float(h))))
                    roi_vertical_scores.append(vertical_score)

                class_name = (
                    self._class_names[cls_id]
                    if 0 <= cls_id < len(self._class_names)
                    else f"class_{cls_id}"
                )
                box_items.append(
                    {
                        "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                        "cls_id": cls_id,
                        "cls_name": class_name,
                        "conf": round(confidence, 4),
                        "area_ratio": round(float(area_ratio), 6),
                    }
                )

        det_count = float(sum(counts))
        mean_conf = float(sum(conf_values) / max(1, len(conf_values)))
        max_conf = float(max(conf_values)) if conf_values else 0.0
        mean_area = float(sum(normalized_areas) / max(1, len(normalized_areas)))
        area_var = 0.0
        if normalized_areas:
            area_var = sum((item - mean_area) ** 2 for item in normalized_areas) / len(normalized_areas)

        count_ratios = [count / max(1.0, det_count) for count in counts]
        class_mean_conf = [
            conf_sums[idx] / counts[idx] if counts[idx] > 0 else 0.0 for idx in range(len(counts))
        ]

        roi_count = len(roi_area_ratios)
        roi_mean_area = float(sum(roi_area_ratios) / max(1, roi_count))
        roi_center_closeness = float(sum(roi_center_scores) / max(1, roi_count))
        roi_vertical_bias = float(sum(roi_vertical_scores) / max(1, roi_count))
        roi_risk = min(1.0, (0.55 * min(1.0, roi_mean_area * 6.0)) + (0.45 * roi_center_closeness))
        roi_count_norm = min(1.0, float(roi_count) / 3.0)

        base_vector = [
            min(1.0, det_count / max(1.0, float(self._max_det))),
            mean_conf,
            max_conf,
            mean_area,
            min(1.0, area_var * 10.0),
        ] + count_ratios + class_mean_conf + [
            roi_risk,
            roi_center_closeness,
            roi_mean_area,
            roi_count_norm,
            roi_vertical_bias,
        ]

        self._last_detection = {
            "count": int(len(box_items)),
            "boxes": box_items,
            "orig_shape": [int(h), int(w)],
            "mean_conf": round(mean_conf, 6),
            "max_conf": round(max_conf, 6),
        }

        return self._fit_feature_dim(base_vector)

    def get_last_detection(self) -> dict[str, Any]:
        """가장 최근 프레임의 검출 메타데이터를 반환한다."""
        return dict(self._last_detection)

    def get_feature_keys(self) -> list[str]:
        """현재 spatial vector의 key 목록을 반환한다."""
        return list(self._feature_keys)

    def _reset_last_detection(self) -> None:
        self._last_detection = {
            "count": 0,
            "boxes": [],
            "orig_shape": None,
            "mean_conf": 0.0,
            "max_conf": 0.0,
        }

    def _fit_feature_dim(self, values: list[float]) -> list[float]:
        if not values:
            return [0.0] * self._feature_dim

        output = [0.0] * self._feature_dim
        fold_count = [0] * self._feature_dim
        for idx, value in enumerate(values):
            target_idx = idx % self._feature_dim
            output[target_idx] += float(value)
            fold_count[target_idx] += 1

        for idx in range(self._feature_dim):
            if fold_count[idx] > 1:
                output[idx] /= float(fold_count[idx])
            output[idx] = round(output[idx], 6)
        return output

    def _fit_feature_keys(self, base_keys: list[str]) -> list[str]:
        if len(base_keys) >= self._feature_dim:
            return base_keys[: self._feature_dim]
        return base_keys + [f"pad_feature_{idx}" for idx in range(self._feature_dim - len(base_keys))]


def build_spatial_encoder(cfg: SpatialConfig) -> SpatialEncoder:
    """설정에 맞는 spatial encoder 구현체를 생성한다."""

    model_name = cfg.model_name.strip().lower()
    if "yolo" in model_name:
        try:
            return YOLOSpatialEncoder(cfg)
        except Exception as exc:
            if cfg.fallback_to_mock:
                logger.warning("YOLOSpatialEncoder 초기화 실패, mock encoder로 대체: %s", exc)
                return MockSpatialEncoder(cfg)
            raise
    return MockSpatialEncoder(cfg)
