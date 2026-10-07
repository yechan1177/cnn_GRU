from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..config import SpatialConfig
from ..features import Detection, FrameDetections, build_feature_extractor
from ..features.semantic_v1 import V1_KEYS, _fit_dim
from ..utils.device import resolve_device
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
    """YOLO 검출 결과를 고정 길이 의미 특징 벡터로 변환하는 spatial encoder.

    특징 계산은 `vcp.features`(v1/v2)에 위임한다. v1은 2026-03 제출본과 같은 값을 낸다.
    """

    BASE_FEATURE_KEYS: list[str] = list(V1_KEYS)

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
        self._device = resolve_device(str(cfg.device))
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

        self._feature_version = str(cfg.feature_version or "v1").strip().lower()
        self._extractor = build_feature_extractor(
            self._feature_version,
            conf_threshold=0.0,  # YOLO predict 단계에서 이미 임계값을 적용한다.
            max_det=self._max_det,
            class_count=max(1, self._class_count),
        )
        self._feature_keys = self._fit_feature_keys(list(self._extractor.keys))

        self._mock_fallback = MockSpatialEncoder(cfg)
        self._warning_issued = False
        self._reset_last_detection()

        logger.info(
            "YOLOSpatialEncoder 초기화: weights=%s, device=%s, class_count=%s, feature_version=%s",
            self._weights_path,
            self._device,
            self._class_count,
            self._feature_version,
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
        h, w = result.orig_shape
        detections: list[Detection] = []
        box_items: list[dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            cls_values = boxes.cls.detach().cpu().tolist()
            conf_values = boxes.conf.detach().cpu().tolist()
            xyxy_values = boxes.xyxy.detach().cpu().tolist()
            image_area = max(1.0, float(h * w))
            for (x1, y1, x2, y2), cls_value, conf in zip(xyxy_values, cls_values, conf_values, strict=True):
                cls_id = int(cls_value)
                det = Detection(float(x1), float(y1), float(x2), float(y2), cls_id, float(conf))
                detections.append(det)
                box_items.append(
                    {
                        "xyxy": [det.x1, det.y1, det.x2, det.y2],
                        "cls_id": cls_id,
                        "cls_name": self._class_names[cls_id]
                        if 0 <= cls_id < len(self._class_names)
                        else f"class_{cls_id}",
                        "conf": round(det.conf, 4),
                        "area_ratio": round(det.width * det.height / image_area, 6),
                    }
                )

        frame_det = FrameDetections(
            frame_id=int(frame.frame_id),
            t=float(frame.sensor_timestamp),
            width=int(w),
            height=int(h),
            boxes=detections,
        )
        vector = self._extractor.update(frame_det)
        confs = [d.conf for d in detections]
        self._last_detection = {
            "count": len(box_items),
            "boxes": box_items,
            "orig_shape": [int(h), int(w)],
            "mean_conf": round(sum(confs) / len(confs), 6) if confs else 0.0,
            "max_conf": round(max(confs), 6) if confs else 0.0,
        }
        return self._fit_feature_dim(vector)

    def get_last_detection(self) -> dict[str, Any]:
        """가장 최근 프레임의 검출 메타데이터를 반환한다."""
        return dict(self._last_detection)

    def get_feature_keys(self) -> list[str]:
        """현재 spatial vector의 key 목록을 반환한다."""
        return list(self._feature_keys)

    @property
    def feature_version(self) -> str:
        return self._feature_version

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
        return _fit_dim(values, self._feature_dim)

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
