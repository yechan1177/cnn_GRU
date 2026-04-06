from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import torch
except Exception:  # pragma: no cover - torch 미설치 환경 방어
    torch = None


# ---------------------------------------------------------------------------
# 간단 재현용 설정
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
INPUT_SOURCE = ROOT_DIR / "data" / "raw" / "videos" / "people_braking.mp4"
YOLO_WEIGHTS = ROOT_DIR / "models" / "checkpoints" / "yolo3cls_best.pt"
FRAME_INDEX = 180
CONF_THRESHOLD = 0.65
INPUT_SIZE = 640
MAX_DET = 30
DEFAULT_DEVICE = "cuda:0"

OUTPUT_PREVIEW = ROOT_DIR / "practice_spatial_encoder_preview.png"


FEATURE_KEYS = [
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


def resolve_device() -> str:
    """현재 환경에서 사용할 YOLO 장치를 정한다."""
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return DEFAULT_DEVICE
    return "cpu"


def load_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    """영상에서 특정 프레임 하나를 읽는다."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    capture.release()

    if not ok or frame is None:
        raise RuntimeError(f"프레임을 읽지 못했습니다. frame_index={frame_index}")
    return frame


def run_yolo(model: YOLO, frame: np.ndarray, device: str) -> Any:
    """현재 프레임에 대해 YOLO 추론을 1회 수행한다."""
    results = model.predict(
        source=frame,
        conf=CONF_THRESHOLD,
        imgsz=INPUT_SIZE,
        device=device,
        max_det=MAX_DET,
        verbose=False,
    )
    if not results:
        raise RuntimeError("YOLO 추론 결과가 비어 있습니다.")
    return results[0]


def compute_feature_vector(result: Any, class_names: list[str]) -> tuple[list[float], dict[str, Any]]:
    """YOLO 검출 결과를 16차원 순간 특징 벡터로 변환한다.

    이 함수는 `src/vcp/components/spatial.py`의 핵심 로직을
    처음 보는 사람도 따라가기 쉽게 줄인 교육용 버전이다.
    """
    boxes = getattr(result, "boxes", None)
    h, w = result.orig_shape
    image_area = max(1.0, float(h * w))

    # 차량 전방 시야를 단순화한 중앙 ROI
    roi_x1 = 0.25 * float(w)
    roi_x2 = 0.75 * float(w)
    roi_y1 = 0.20 * float(h)
    roi_y2 = 0.90 * float(h)

    if boxes is None or len(boxes) == 0:
        return [0.0] * len(FEATURE_KEYS), {
            "boxes": [],
            "roi": [roi_x1, roi_y1, roi_x2, roi_y2],
            "orig_shape": [int(h), int(w)],
        }

    cls_values = boxes.cls.detach().cpu().tolist()
    conf_values = boxes.conf.detach().cpu().tolist()
    xyxy_values = boxes.xyxy.detach().cpu().tolist()

    class_count = max(3, len(class_names))
    counts = [0.0] * class_count
    conf_sums = [0.0] * class_count
    areas: list[float] = []
    roi_areas: list[float] = []
    roi_center_scores: list[float] = []
    roi_vertical_scores: list[float] = []
    box_items: list[dict[str, Any]] = []

    for idx, cls_value in enumerate(cls_values):
        cls_id = int(cls_value)
        if cls_id < 0 or cls_id >= class_count:
            continue

        conf = float(conf_values[idx]) if idx < len(conf_values) else 0.0
        x1, y1, x2, y2 = [float(v) for v in xyxy_values[idx]]
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        area_ratio = (box_w * box_h) / image_area
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        counts[cls_id] += 1.0
        conf_sums[cls_id] += conf
        areas.append(area_ratio)

        in_roi = roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2
        if in_roi:
            roi_areas.append(area_ratio)
            center_dx = abs(cx - (w * 0.5)) / max(1.0, w * 0.5)
            center_score = max(0.0, 1.0 - center_dx)
            roi_center_scores.append(center_score)
            roi_vertical_scores.append(min(1.0, max(0.0, cy / max(1.0, float(h)))))

        class_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        box_items.append(
            {
                "xyxy": [x1, y1, x2, y2],
                "cls_id": cls_id,
                "cls_name": class_name,
                "conf": round(conf, 4),
                "area_ratio": round(area_ratio, 6),
                "in_roi": in_roi,
            }
        )

    det_count = float(sum(counts))
    mean_conf = float(sum(conf_values) / max(1, len(conf_values)))
    max_conf = float(max(conf_values)) if conf_values else 0.0
    mean_area = float(sum(areas) / max(1, len(areas)))
    area_var = float(np.var(areas)) if areas else 0.0

    count_ratios = [count / max(1.0, det_count) for count in counts]
    class_mean_conf = [
        conf_sums[idx] / counts[idx] if counts[idx] > 0 else 0.0
        for idx in range(class_count)
    ]

    roi_count = len(roi_areas)
    roi_mean_area = float(sum(roi_areas) / max(1, roi_count))
    center_closeness = float(sum(roi_center_scores) / max(1, roi_count))
    roi_vertical_bias = float(sum(roi_vertical_scores) / max(1, roi_count))
    roi_count_norm = min(1.0, roi_count / 3.0)
    roi_risk = min(1.0, (0.55 * min(1.0, roi_mean_area * 6.0)) + (0.45 * center_closeness))

    feature_vector = [
        min(1.0, det_count / max(1.0, float(MAX_DET))),
        mean_conf,
        max_conf,
        mean_area,
        min(1.0, area_var * 10.0),
        count_ratios[0] if len(count_ratios) > 0 else 0.0,
        count_ratios[1] if len(count_ratios) > 1 else 0.0,
        count_ratios[2] if len(count_ratios) > 2 else 0.0,
        class_mean_conf[0] if len(class_mean_conf) > 0 else 0.0,
        class_mean_conf[1] if len(class_mean_conf) > 1 else 0.0,
        class_mean_conf[2] if len(class_mean_conf) > 2 else 0.0,
        roi_risk,
        center_closeness,
        roi_mean_area,
        roi_count_norm,
        roi_vertical_bias,
    ]

    return [round(float(v), 6) for v in feature_vector], {
        "boxes": box_items,
        "roi": [roi_x1, roi_y1, roi_x2, roi_y2],
        "orig_shape": [int(h), int(w)],
    }


def draw_preview(frame: np.ndarray, detection_meta: dict[str, Any], feature_vector: list[float]) -> np.ndarray:
    """YOLO 박스, ROI, 대표 feature를 한 장의 그림으로 저장한다."""
    preview = frame.copy()
    h, w = preview.shape[:2]

    roi_x1, roi_y1, roi_x2, roi_y2 = detection_meta["roi"]
    cv2.rectangle(preview, (int(roi_x1), int(roi_y1)), (int(roi_x2), int(roi_y2)), (0, 255, 255), 2)
    cv2.putText(preview, "ROI", (int(roi_x1) + 6, int(roi_y1) + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    color_map = {
        "vehicle": (80, 220, 80),
        "person": (80, 160, 255),
        "bike": (255, 180, 80),
    }
    for item in detection_meta["boxes"]:
        x1, y1, x2, y2 = [int(v) for v in item["xyxy"]]
        class_name = str(item["cls_name"])
        color = color_map.get(class_name, (255, 255, 255))
        cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {item['conf']:.2f}"
        cv2.putText(preview, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    panel_w = 430
    canvas = np.full((h, w + panel_w, 3), 245, dtype=np.uint8)
    canvas[:, :w] = preview

    cv2.putText(canvas, "Spatial Feature Example", (w + 18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 2)
    cv2.putText(canvas, "YOLO -> 16D Feature Vector", (w + 18, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (60, 60, 60), 1)

    y = 104
    for key, value in zip(FEATURE_KEYS, feature_vector):
        cv2.putText(canvas, f"{key}: {value:.3f}", (w + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
        y += 24

    cv2.putText(canvas, "Input : one RGB frame", (w + 18, h - 56), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (70, 70, 70), 1)
    cv2.putText(canvas, "Output: 16D meaning-based vector", (w + 18, h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (70, 70, 70), 1)
    return canvas


def main() -> None:
    if not YOLO_WEIGHTS.exists():
        raise FileNotFoundError(f"YOLO 가중치를 찾을 수 없습니다: {YOLO_WEIGHTS}")
    if not INPUT_SOURCE.exists():
        raise FileNotFoundError(f"입력 영상을 찾을 수 없습니다: {INPUT_SOURCE}")

    device = resolve_device()
    print(f"[practice_spatial_encoder] device = {device}")

    print("[practice_spatial_encoder] loading frame...")
    frame = load_video_frame(INPUT_SOURCE, FRAME_INDEX)

    print("[practice_spatial_encoder] loading YOLO...")
    model = YOLO(str(YOLO_WEIGHTS))
    names = getattr(model.model, "names", None) or getattr(model, "names", None)
    if isinstance(names, dict):
        class_names = [str(name) for _, name in sorted(names.items())]
    elif isinstance(names, list):
        class_names = [str(name) for name in names]
    else:
        class_names = ["vehicle", "person", "bike"]

    print("[practice_spatial_encoder] running detection...")
    result = run_yolo(model, frame, device=device)

    print("[practice_spatial_encoder] computing 16D features...")
    feature_vector, detection_meta = compute_feature_vector(result, class_names)

    print("[practice_spatial_encoder] feature vector")
    for key, value in zip(FEATURE_KEYS, feature_vector):
        print(f"  - {key}: {value:.6f}")

    print("[practice_spatial_encoder] saving preview...")
    preview = draw_preview(frame, detection_meta, feature_vector)
    cv2.imwrite(str(OUTPUT_PREVIEW), preview)

    print(f"[practice_spatial_encoder] done: {OUTPUT_PREVIEW}")


if __name__ == "__main__":
    main()
