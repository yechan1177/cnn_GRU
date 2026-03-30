from __future__ import annotations

"""최종 CNN-GRU + 규칙 보정 모델을 즉시 실행하는 루트 스크립트.

이 파일은 저장소를 처음 받은 사람이 가장 먼저 실행하게 되는 진입점이다.
논문 기준 3번 모델(`CNN-GRU + 규칙`)의 실제 동작을 화면에서 바로 확인할 수 있다.
"""

import time
from collections import deque
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vcp.components.event_rules import InstantRuleParams, predict_instant_rule
from vcp.config import SpatialConfig
from vcp.components.feature_packer import SimpleFeaturePacker
from vcp.components.spatial import YOLOSpatialEncoder
from vcp.schemas import FramePacket
from vcp.tools.final_hybrid_model_ui import draw_series, draw_text, load_hybrid_params, load_model, predict_sequence


# 입력 소스는 비디오 파일 또는 웹캠 중 하나를 선택해 사용한다.
USE_WEBCAM = False
WEBCAM_INDEX = 0
VIDEO_SOURCE = str(ROOT_DIR / "data" / "raw" / "videos" / "people_braking.mp4")

# 아래 세 파일만 있으면 논문 재현 브랜치에서 최종 모델을 바로 실행할 수 있다.
YOLO_WEIGHTS = str(ROOT_DIR / "models" / "checkpoints" / "yolo3cls_best.pt")
TEMPORAL_CKPT = str(ROOT_DIR / "models" / "checkpoints" / "temporal_final_best.pt")
HYBRID_MANIFEST = str(ROOT_DIR / "configs" / "hybrid_rule_params.json")

# detector와 temporal 모델이 공통으로 사용하는 실행 설정이다.
DEVICE = "cuda:0"
CONF_THRESHOLD = 0.65
INPUT_SIZE = 640
MAX_DET = 30
FEATURE_DIM = 16

WINDOW_NAME = "Final Model Live Runner"
SCREENSHOT_PATH = str(ROOT_DIR / "artifacts" / "screenshots" / "run_final_model_capture.jpg")


def _import_cv2_np():
    """opencv와 numpy를 지연 로드한다.

    패키지가 없을 때는 필요한 라이브러리를 바로 알 수 있도록 오류 메시지를 바꿔 준다.
    """
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("실행하려면 opencv-python과 numpy가 필요합니다.") from exc
    return cv2, np


def _open_capture(cv2_module):
    """비디오 파일 또는 웹캠 입력을 열고 실제 사용된 입력 이름을 함께 돌려준다."""
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
            print(f"[run_final_model] 영상 파일이 없어 웹캠으로 대체합니다: {video_path}")
    if not capture.isOpened():
        raise RuntimeError(f"입력을 열 수 없습니다: {source_name}")
    return capture, source_name


def _make_spatial_encoder() -> YOLOSpatialEncoder:
    """최종 모델이 공통으로 사용하는 YOLO spatial encoder를 생성한다."""
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
    """YOLO가 검출한 박스를 원본 프레임 위에 그린다."""
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


def main() -> None:
    """최종 하이브리드 모델을 실시간으로 실행한다.

    전체 흐름:
    1. 현재 프레임을 읽는다.
    2. YOLO로 박스와 16차원 순간 feature를 만든다.
    3. 최근 8프레임 시퀀스를 CNN-GRU에 넣는다.
    4. 순간 규칙으로 최종 태그를 보정한다.
    5. 결과를 오른쪽 정보 패널과 함께 보여 준다.
    """
    cv2, np = _import_cv2_np()
    spatial = _make_spatial_encoder()
    packer = SimpleFeaturePacker()
    model, labels, window_size = load_model(Path(TEMPORAL_CKPT))
    params = load_hybrid_params(Path(HYBRID_MANIFEST))
    rule_params = InstantRuleParams()
    capture, source_name = _open_capture(cv2)

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 15.0

    # feature_buffer는 최근 8프레임 feature를 유지하는 핵심 버퍼다.
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

                # detector와 packer가 공통으로 쓰는 입력 프레임 래퍼를 만든다.
                packet = FramePacket(
                    frame_id=frame_id,
                    sensor_timestamp=frame_id / max(1.0, fps),
                    system_timestamp=time.time(),
                    pixels=[],
                    raw_path=f"{source_name}#frame={frame_id}",
                    image=frame_bgr,
                )

                # 1) 현재 프레임 YOLO 검출 + 16차원 순간 feature 생성
                spatial_vector = spatial.encode(packet)
                detection = spatial.get_last_detection()

                # 2) 현재 프레임만으로 계산하는 순간 규칙 결과
                rule_result = predict_instant_rule(detection, rule_params)

                # 3) 이후 temporal 모델 입력으로 쓰기 위해 feature를 고정 형식으로 묶는다.
                packed = packer.pack(packet, spatial_vector)
                feature_buffer.append(list(packed.spatial_vector))

                # 초반 프레임은 길이가 부족하므로 0 벡터로 왼쪽 padding을 채운다.
                while len(feature_buffer) < window_size:
                    feature_buffer.appendleft([0.0] * len(spatial_vector))

                # 4) CNN-GRU 출력과 규칙 결과를 결합해 최종 라벨을 얻는다.
                pred = predict_sequence(model, labels, list(feature_buffer), params, rule_result=rule_result)
                pred_history.append(pred)

                # 5) 왼쪽은 원본 프레임, 오른쪽은 분석 패널로 렌더링한다.
                canvas = frame_bgr.copy()
                _draw_detection_boxes(cv2, canvas, detection)

                h, _ = canvas.shape[:2]
                panel_w = 520
                panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
                panel[:] = (20, 20, 20)

                draw_text(cv2, panel, 18, 30, f"source: {source_name}", (200, 230, 255), 0.5)
                draw_text(cv2, panel, 18, 58, f"frame: {frame_id}", (230, 230, 230), 0.55)
                draw_text(cv2, panel, 18, 90, f"pure model: {pred['pure_label']} ({pred['pure_prob']:.3f})", (120, 255, 160), 0.62)
                final_color = (70, 220, 255) if pred["final_label"] != pred["pure_label"] else (255, 220, 120)
                draw_text(cv2, panel, 18, 120, f"hybrid final: {pred['final_label']} ({pred['final_prob']:.3f})", final_color, 0.72)
                draw_text(cv2, panel, 18, 146, f"decision: {pred['decision_reason']}", (220, 220, 220), 0.48)
                draw_text(cv2, panel, 18, 176, f"rule label: {pred['rule_label']}", (200, 255, 180), 0.55)
                draw_text(cv2, panel, 18, 204, f"rule warn={pred['rule_warn_score']:.3f} hard={pred['rule_hard_score']:.3f}", (200, 255, 180), 0.52)
                draw_text(cv2, panel, 18, 232, f"boundary: {pred['boundary']:.3f}", (240, 240, 240), 0.55)
                draw_text(cv2, panel, 18, 260, f"warn_prob: {pred['warn_prob']:.3f}", (255, 210, 80), 0.55)
                draw_text(cv2, panel, 18, 288, f"hard_prob: {pred['hard_prob']:.3f}", (255, 140, 110), 0.55)
                draw_text(cv2, panel, 18, 316, f"follow_prob: {pred['follow_prob']:.3f}", (150, 220, 255), 0.55)
                draw_text(cv2, panel, 18, 348, f"roi={pred['roi_risk']:.3f} center={pred['center']:.3f}", (220, 220, 220), 0.5)
                draw_text(cv2, panel, 18, 374, f"roi_mean_area={pred['roi_mean_area']:.3f}", (220, 220, 220), 0.5)
                draw_text(cv2, panel, 18, 400, f"roi_count_norm={pred['roi_count_norm']:.3f} roi_y={pred['roi_vertical_bias']:.3f}", (220, 220, 220), 0.5)
                draw_text(cv2, panel, 18, 428, f"warn_reason: {pred['warn_reason']}", (255, 220, 150), 0.48)
                draw_text(cv2, panel, 18, 452, f"hard_reason: {pred['hard_reason']}", (255, 170, 150), 0.48)

                # 최근 예측 추이를 그래프로 그리면 모델이 언제 반응했는지 바로 볼 수 있다.
                boundary_series = [item["boundary"] for item in pred_history]
                warn_series = [item["warn_prob"] for item in pred_history]
                hard_series = [item["hard_prob"] for item in pred_history]

                draw_series(cv2, np, panel, x=18, y=500, w=460, h=80, values=boundary_series, color=(100, 200, 255), label="boundary", vmin=0.0, vmax=1.0, ref_line=params.warn_boundary_thr)
                draw_series(cv2, np, panel, x=18, y=610, w=460, h=65, values=warn_series, color=(255, 210, 80), label="brake_warning prob", vmin=0.0, vmax=1.0, ref_line=params.warn_prob_thr)
                draw_series(cv2, np, panel, x=18, y=700, w=460, h=65, values=hard_series, color=(255, 140, 110), label="hard_brake_risk prob", vmin=0.0, vmax=1.0, ref_line=params.hard_prob_thr)
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
