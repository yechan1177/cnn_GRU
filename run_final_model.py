from __future__ import annotations

"""루트에서 바로 실행하는 최종 모델 확인 스크립트.

상단 기본값을 수정하거나 명령행 인자로 덮어써서 실행한다.

예시::

    # 로컬 GUI (mp4)
    python run_final_model.py --source data/raw/videos/people_braking.mp4

    # 웹캠
    python run_final_model.py --source webcam:0

    # 클라우드/서버(화면 없음): 결과 영상 + 프레임별 예측 JSONL 저장
    python run_final_model.py --source video.mp4 --no-display \\
        --save-video artifacts/screenshots/run.mp4 --output-jsonl artifacts/run.jsonl
"""

import argparse
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vcp.components.feature_packer import SimpleFeaturePacker  # noqa: E402
from vcp.components.spatial import YOLOSpatialEncoder  # noqa: E402
from vcp.config import SpatialConfig  # noqa: E402
from vcp.schemas import FramePacket  # noqa: E402
from vcp.tools.final_hybrid_model_ui import (  # noqa: E402
    draw_series,
    draw_text,
    load_hybrid_params,
    load_model,
    predict_sequence,
)
from vcp.utils.device import resolve_device  # noqa: E402

logger = logging.getLogger("run_final_model")

# =========================
# 기본 설정 (명령행 인자로 덮어쓸 수 있음)
# =========================
DEFAULT_SOURCE = str(ROOT_DIR / "data" / "raw" / "videos" / "people_braking.mp4")
WEBCAM_INDEX = 0

YOLO_WEIGHTS = str(ROOT_DIR / "models" / "checkpoints" / "yolo3cls_best.pt")
TEMPORAL_CKPT = str(ROOT_DIR / "models" / "checkpoints" / "temporal_final_best.pt")
HYBRID_MANIFEST = str(ROOT_DIR / "configs" / "hybrid_rule_params.json")

DEVICE = "auto"  # auto: CUDA가 있으면 cuda:0, 없으면 cpu
# 주의: 배포 체크포인트의 학습 특징은 conf 0.45로 생성되었다(TASK_033).
# 2026-03 실행기는 0.65를 사용해 학습/추론 임계값이 달랐다.
CONF_THRESHOLD = 0.45
INPUT_SIZE = 640
MAX_DET = 30
FEATURE_DIM = 16

WINDOW_NAME = "Final Model Live Runner"
SCREENSHOT_PATH = str(ROOT_DIR / "artifacts" / "screenshots" / "run_final_model_capture.jpg")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="최종 하이브리드 모델 실행기 (GUI/headless 겸용)")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="mp4 경로 또는 webcam:<index>")
    parser.add_argument("--device", default=DEVICE, help="auto | cpu | cuda:0")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="YOLO confidence 임계값")
    parser.add_argument("--yolo-weights", default=YOLO_WEIGHTS)
    parser.add_argument("--temporal-ckpt", default=TEMPORAL_CKPT)
    parser.add_argument("--hybrid-params", default=HYBRID_MANIFEST)
    parser.add_argument("--no-display", action="store_true", help="화면 출력 없이 실행(클라우드/서버)")
    parser.add_argument("--save-video", default="", help="렌더링 결과 mp4 저장 경로")
    parser.add_argument("--output-jsonl", default="", help="프레임별 예측 JSONL 저장 경로")
    parser.add_argument("--max-frames", type=int, default=0, help="0이면 끝까지")
    return parser.parse_args(argv)


def _import_cv2_np():
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("실행에는 opencv-python과 numpy가 필요합니다.") from exc
    return cv2, np


def _open_capture(cv2_module, source: str, allow_webcam_fallback: bool):
    if source.startswith("webcam:"):
        index = int(source.split(":", 1)[1] or WEBCAM_INDEX)
        capture = cv2_module.VideoCapture(index)
        source_name = source
    else:
        video_path = Path(source)
        if video_path.exists():
            capture = cv2_module.VideoCapture(str(video_path))
            source_name = str(video_path)
        elif allow_webcam_fallback:
            capture = cv2_module.VideoCapture(WEBCAM_INDEX)
            source_name = f"webcam:{WEBCAM_INDEX}"
            logger.warning("영상 파일이 없어 웹캠으로 전환합니다: %s", video_path)
        else:
            raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

    if not capture.isOpened():
        raise RuntimeError(f"입력 소스를 열 수 없습니다: {source_name}")
    return capture, source_name


def _make_spatial_encoder(weights: str, device: str, conf: float) -> YOLOSpatialEncoder:
    cfg = SpatialConfig(
        model_name="yolo_nano",
        feature_dim=FEATURE_DIM,
        weights_path=weights,
        device=device,
        conf_threshold=conf,
        max_det=MAX_DET,
        input_size=INPUT_SIZE,
        fallback_to_mock=False,
        feature_version="v1",
    )
    return YOLOSpatialEncoder(cfg)


def _draw_detection_boxes(cv2_module, frame, detections: dict) -> None:
    boxes = detections.get("boxes", []) if isinstance(detections, dict) else []
    palette = [(80, 220, 80), (60, 180, 255), (255, 180, 60), (200, 120, 255)]
    for box in boxes:
        xyxy = box.get("xyxy", [])
        if not isinstance(xyxy, list) or len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cls_id = int(box.get("cls_id", 0))
        label = f"{box.get('cls_name', 'obj')} {float(box.get('conf', 0.0)):.2f}"
        color = palette[cls_id % len(palette)]
        cv2_module.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2_module.putText(frame, label, (x1, max(18, y1 - 8)), cv2_module.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2_module.LINE_AA)


def _render(cv2, np, frame_bgr, spatial, pred, pred_history, params, source_name: str, frame_id: int):
    canvas = frame_bgr.copy()
    _draw_detection_boxes(cv2, canvas, spatial.get_last_detection())
    h, _ = canvas.shape[:2]
    panel = np.zeros((max(h, 740), 520, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)

    draw_text(cv2, panel, 18, 30, f"source: {source_name}"[:60], (200, 230, 255), 0.5)
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

    draw_series(cv2, np, panel, x=18, y=420, w=460, h=90, values=[p["boundary"] for p in pred_history], color=(100, 200, 255), label="boundary", vmin=0.0, vmax=1.0, ref_line=params.warn_boundary_thr)
    draw_series(cv2, np, panel, x=18, y=540, w=460, h=70, values=[p["warn_prob"] for p in pred_history], color=(255, 210, 80), label="brake_warning prob", vmin=0.0, vmax=1.0, ref_line=params.warn_prob_thr)
    draw_series(cv2, np, panel, x=18, y=640, w=460, h=70, values=[p["hard_prob"] for p in pred_history], color=(255, 140, 110), label="hard_brake_risk prob", vmin=0.0, vmax=1.0, ref_line=params.hard_prob_thr)
    draw_text(cv2, panel, 18, panel.shape[0] - 28, "space:pause  s:screenshot  q:quit", (180, 180, 180), 0.48)

    if panel.shape[0] != h:
        panel = cv2.resize(panel, (panel.shape[1], h))
    return np.concatenate([canvas, panel], axis=1)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    cv2, np = _import_cv2_np()

    device = resolve_device(args.device)
    spatial = _make_spatial_encoder(args.yolo_weights, device, args.conf)
    packer = SimpleFeaturePacker()
    model, labels, window_size = load_model(Path(args.temporal_ckpt))
    params = load_hybrid_params(Path(args.hybrid_params))

    capture, source_name = _open_capture(cv2, args.source, allow_webcam_fallback=not args.no_display)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 15.0
    is_webcam = source_name.startswith("webcam:")

    feature_buffer: deque[list[float]] = deque(maxlen=window_size)
    pred_history: deque[dict] = deque(maxlen=120)
    paused = False
    frame_id = 0
    last_render = None
    writer = None
    jsonl_file = None
    latencies: list[float] = []

    if args.output_jsonl:
        Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = Path(args.output_jsonl).open("w", encoding="utf-8")
    if not args.no_display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            if not paused:
                ok, frame_bgr = capture.read()
                if not ok or (args.max_frames and frame_id >= args.max_frames):
                    break

                started = time.perf_counter()
                packet = FramePacket(
                    frame_id=frame_id,
                    sensor_timestamp=frame_id / max(1.0, fps),
                    system_timestamp=time.time(),
                    pixels=[],
                    raw_path=None,
                    image=frame_bgr,
                )
                spatial_vector = spatial.encode(packet)
                packed = packer.pack(packet, spatial_vector)
                feature_buffer.append(list(packed.spatial_vector))
                while len(feature_buffer) < window_size:
                    feature_buffer.appendleft([0.0] * len(spatial_vector))
                pred = predict_sequence(model, labels, list(feature_buffer), params)
                latencies.append(time.perf_counter() - started)
                pred_history.append(pred)

                if jsonl_file is not None:
                    row = {"frame_id": frame_id, "t": round(frame_id / fps, 4)}
                    row.update({k: (round(v, 6) if isinstance(v, float) else v) for k, v in pred.items()})
                    jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")

                if args.save_video or not args.no_display:
                    last_render = _render(cv2, np, frame_bgr, spatial, pred, pred_history, params, source_name, frame_id)
                    if args.save_video:
                        if writer is None:
                            Path(args.save_video).parent.mkdir(parents=True, exist_ok=True)
                            hh, ww = last_render.shape[:2]
                            writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ww, hh))
                        writer.write(last_render)
                frame_id += 1

            if args.no_display:
                continue
            if last_render is not None:
                cv2.imshow(WINDOW_NAME, last_render)
            key = cv2.waitKey(1 if is_webcam and not paused else int(1000 / max(1.0, fps))) & 0xFF
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
        if writer is not None:
            writer.release()
        if jsonl_file is not None:
            jsonl_file.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    if latencies:
        mean_ms = 1000.0 * sum(latencies) / len(latencies)
        logger.info("처리 프레임=%d, device=%s, 평균 지연=%.1f ms/frame", frame_id, device, mean_ms)


if __name__ == "__main__":
    main()
