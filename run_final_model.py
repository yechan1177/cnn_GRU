from __future__ import annotations

"""루트에서 바로 실행하는 최종 모델 확인 스크립트.

상단 설정값만 바꾸면
- mp4 파일 입력
- webcam 입력
둘 다 바로 실행할 수 있다.
"""

import time
from collections import deque
from pathlib import Path
import sys

import torch

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vcp.config import SpatialConfig
from vcp.components.feature_packer import SimpleFeaturePacker
from vcp.components.spatial import YOLOSpatialEncoder
from vcp.schemas import FramePacket
from vcp.tools.final_hybrid_model_ui import (
    draw_series,
    draw_text,
    load_hybrid_params,
    load_model,
    predict_sequence,
)


# =========================
# 사용자 설정
# =========================
USE_WEBCAM = False
WEBCAM_INDEX = 0
VIDEO_SOURCE = str(ROOT_DIR / "data" / "raw" / "videos" / "people_braking.mp4")

YOLO_WEIGHTS = str(ROOT_DIR / "models" / "checkpoints" / "yolo3cls_best.pt")
TEMPORAL_CKPT = str(ROOT_DIR / "models" / "checkpoints" / "temporal_final_best.pt")
HYBRID_MANIFEST = str(ROOT_DIR / "configs" / "hybrid_rule_params.json")

DEVICE = "cuda:0"
CONF_THRESHOLD = 0.65
INPUT_SIZE = 640
MAX_DET = 30
FEATURE_DIM = 16

WINDOW_NAME = "Final Model Live Runner"
SCREENSHOT_PATH = str(ROOT_DIR / "artifacts" / "screenshots" / "run_final_model_capture.jpg")


def _import_cv2_np():
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("실행에는 opencv-python과 numpy가 필요합니다.") from exc
    return cv2, np


def _open_capture(cv2_module):
    if USE_WEBCAM:
        capture = cv2_module.VideoCapture(WEBCAM_INDEX)
        source_name = f"webcam:{WEBCAM_INDEX}"
    else:
        video_path = Path(VIDEO_SOURCE)
        if video_path.exists():
            capture = cv2_module.VideoCapture(str(video_path))
            source_name = str(video_path)
        else:
            capture = cv2_module.VideoCapture(WEBCAM_INDEX)
            source_name = f"webcam:{WEBCAM_INDEX}"
            print(f"[run_final_model] ?? ??? ?? ???? ?????: {video_path}")

    if not capture.isOpened():
        raise RuntimeError(f"??? ?? ?????: {source_name}")
    return capture, source_name


def _make_spatial_encoder() -> YOLOSpatialEncoder:
    cfg = SpatialConfig(
        model_name="yolo_nano",
        feature_dim=FEATURE_DIM,
        weights_path=YOLO_WEIGHTS,
        device=DEVICE,
        conf_threshold=CONF_THRESHOLD,
        max_det=MAX_DET,
        input_size=INPUT_SIZE,
        fallback_to_mock=False,
    )
    return YOLOSpatialEncoder(cfg)


def _draw_detection_boxes(cv2_module, frame, detections: dict):
    boxes = detections.get("boxes", []) if isinstance(detections, dict) else []
    palette = [
        (80, 220, 80),
        (60, 180, 255),
        (255, 180, 60),
        (200, 120, 255),
    ]
    for box in boxes:
        xyxy = box.get("xyxy", [])
        if not isinstance(xyxy, list) or len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cls_id = int(box.get("cls_id", 0))
        label = f"{box.get('cls_name', 'obj')} {float(box.get('conf', 0.0)):.2f}"
        color = palette[cls_id % len(palette)]
        cv2_module.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2_module.putText(
            frame,
            label,
            (x1, max(18, y1 - 8)),
            cv2_module.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2_module.LINE_AA,
        )


def main() -> None:
    cv2, np = _import_cv2_np()

    spatial = _make_spatial_encoder()
    packer = SimpleFeaturePacker()
    model, labels, window_size = load_model(Path(TEMPORAL_CKPT))
    params = load_hybrid_params(Path(HYBRID_MANIFEST))

    capture, source_name = _open_capture(cv2)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 15.0

    feature_buffer: deque[list[float]] = deque(maxlen=window_size)
    pred_history: deque[dict] = deque(maxlen=120)
    paused = False
    frame_id = 0
    last_render = None

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            if not paused:
                ok, frame_bgr = capture.read()
                if not ok:
                    break

                sensor_ts = frame_id / max(1.0, fps)
                pixels = cv2.resize(
                    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY),
                    (8, 4),
                    interpolation=cv2.INTER_AREA,
                ).reshape(-1)
                pixels_norm = [round(float(v) / 255.0, 6) for v in pixels.tolist()]

                packet = FramePacket(
                    frame_id=frame_id,
                    sensor_timestamp=sensor_ts,
                    system_timestamp=time.time(),
                    pixels=pixels_norm,
                    raw_path=f"{source_name}#frame={frame_id}",
                    image=frame_bgr,
                )

                spatial_vector = spatial.encode(packet)
                packed = packer.pack(packet, spatial_vector)
                feature_buffer.append(list(packed.spatial_vector))
                while len(feature_buffer) < window_size:
                    feature_buffer.appendleft([0.0] * len(spatial_vector))

                pred = predict_sequence(model, labels, list(feature_buffer), params)
                pred_history.append(pred)

                canvas = frame_bgr.copy()
                _draw_detection_boxes(cv2, canvas, spatial.get_last_detection())

                h, _ = canvas.shape[:2]
                panel_w = 520
                panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
                panel[:] = (20, 20, 20)

                draw_text(cv2, panel, 18, 30, f"source: {source_name}", (200, 230, 255), 0.5)
                draw_text(cv2, panel, 18, 58, f"frame: {frame_id}", (230, 230, 230), 0.55)
                draw_text(cv2, panel, 18, 90, f"pure model: {pred['pure_label']} ({pred['pure_prob']:.3f})", (120, 255, 160), 0.62)
                final_color = (70, 220, 255) if pred["final_label"] != pred["pure_label"] else (255, 220, 120)
                draw_text(cv2, panel, 18, 120, f"hybrid final: {pred['final_label']}", final_color, 0.72)

                draw_text(cv2, panel, 18, 160, f"boundary: {pred['boundary']:.3f}", (240, 240, 240), 0.55)
                draw_text(cv2, panel, 18, 188, f"warn_prob: {pred['warn_prob']:.3f}", (255, 210, 80), 0.55)
                draw_text(cv2, panel, 18, 216, f"hard_prob: {pred['hard_prob']:.3f}", (255, 140, 110), 0.55)
                draw_text(cv2, panel, 18, 244, f"follow_prob: {pred['follow_prob']:.3f}", (150, 220, 255), 0.55)

                draw_text(cv2, panel, 18, 286, f"roi={pred['roi_risk']:.3f} center={pred['center']:.3f}", (220, 220, 220), 0.5)
                draw_text(cv2, panel, 18, 312, f"looming={pred['looming']:.3f} occlusion={pred['occlusion']:.3f}", (220, 220, 220), 0.5)
                draw_text(cv2, panel, 18, 338, f"motion_delta={pred['motion_delta']:.3f}", (220, 220, 220), 0.5)
                draw_text(cv2, panel, 18, 368, f"warn_rule={int(pred['warn_rule'])} hard_rule={int(pred['hard_rule'])}", (180, 255, 180), 0.55)

                boundary_series = [item["boundary"] for item in pred_history]
                warn_series = [item["warn_prob"] for item in pred_history]
                hard_series = [item["hard_prob"] for item in pred_history]

                draw_series(cv2, np, panel, x=18, y=420, w=460, h=90, values=boundary_series, color=(100, 200, 255), label="boundary", vmin=0.0, vmax=1.0, ref_line=params.warn_boundary_thr)
                draw_series(cv2, np, panel, x=18, y=540, w=460, h=70, values=warn_series, color=(255, 210, 80), label="brake_warning prob", vmin=0.0, vmax=1.0, ref_line=params.warn_prob_thr)
                draw_series(cv2, np, panel, x=18, y=640, w=460, h=70, values=hard_series, color=(255, 140, 110), label="hard_brake_risk prob", vmin=0.0, vmax=1.0, ref_line=params.hard_prob_thr)
                draw_text(cv2, panel, 18, h - 28, "space:pause  s:screenshot  q:quit", (180, 180, 180), 0.48)

                last_render = np.concatenate([canvas, panel], axis=1)
                frame_id += 1

            if last_render is not None:
                cv2.imshow(WINDOW_NAME, last_render)

            key = cv2.waitKey(1 if USE_WEBCAM and not paused else int(1000 / max(1.0, fps))) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
            if key == ord("s") and last_render is not None:
                out = Path(SCREENSHOT_PATH)
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), last_render)

    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
