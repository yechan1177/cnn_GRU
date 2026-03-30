from __future__ import annotations

"""같은 영상에 대해 3개 모델을 동시에 실행하고 비교 산출물을 저장한다."""

import csv
import json
import time
from collections import Counter, deque
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vcp.components.event_rules import InstantRuleParams, predict_instant_rule
from vcp.config import SpatialConfig
from vcp.components.feature_packer import SimpleFeaturePacker
from vcp.components.spatial import YOLOSpatialEncoder
from vcp.schemas import FramePacket
from vcp.tools.final_hybrid_model_ui import load_hybrid_params, load_model, predict_sequence


USE_WEBCAM = False
WEBCAM_INDEX = 0
VIDEO_SOURCE = str(ROOT_DIR / "data" / "raw" / "videos" / "people_braking.mp4")

YOLO_WEIGHTS = str(ROOT_DIR / "models" / "checkpoints" / "yolo3cls_best.pt")
SHARED_TEMPORAL_CKPT = str(ROOT_DIR / "models" / "checkpoints" / "temporal_shared_best.pt")
FINAL_TEMPORAL_CKPT = str(ROOT_DIR / "models" / "checkpoints" / "temporal_final_best.pt")
HYBRID_MANIFEST = str(ROOT_DIR / "configs" / "hybrid_rule_params.json")
DETECTOR_RESULT_SUMMARY = ROOT_DIR / "experiments" / "exp_011_yolo3cls_training" / "result_summary.md"

DEVICE = "cuda:0"
CONF_THRESHOLD = 0.65
INPUT_SIZE = 640
MAX_DET = 30
FEATURE_DIM = 16
MAX_FRAMES = 0
SAVE_ROOT = ROOT_DIR / "artifacts" / "comparisons"
EVENT_LABELS = {"brake_warning", "hard_brake_risk"}
CLIP_PRE_FRAMES = 15
CLIP_POST_FRAMES = 20
MIN_EVENT_GAP = 24


def _import_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("실행하려면 opencv-python이 필요합니다.") from exc
    return cv2


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
            print(f"[run_compare_three_models] 영상 파일이 없어 웹캠으로 대체합니다: {video_path}")
    if not capture.isOpened():
        raise RuntimeError(f"입력을 열 수 없습니다: {source_name}")
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


def _safe_stem(source_name: str) -> str:
    if source_name.startswith("webcam:"):
        return source_name.replace(":", "_")
    return Path(source_name).stem


def _label_index_map() -> dict[str, int]:
    return {
        "normal_drive": 0,
        "front_vehicle_follow": 1,
        "brake_warning": 2,
        "hard_brake_risk": 3,
        "post_brake_recovery": 4,
        "dense_traffic": 5,
    }


def _save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_markdown(path: Path, summary: dict[str, object]) -> None:
    lines = [
        "# 세 모델 비교 요약",
        "",
        f"- 입력 소스: `{summary['source']}`",
        f"- 처리 프레임: `{summary['frame_count']}`",
        f"- 최종 모델 이벤트 수: `{summary['event_count']}`",
        "",
        "## 라벨 분포",
    ]
    for model_name, counts in summary["label_counts"].items():
        lines.append(f"- `{model_name}`: {counts}")
    lines.extend(
        [
            "",
            "## 정량 지표",
            f"- `A_det(mAP50)`: {summary['metrics']['A_det']:.5f}",
            f"- `T_inf_rule(ms/frame, end-to-end)`: {summary['metrics']['T_inf_rule_ms']:.4f}",
            f"- `T_inf_temporal(ms/frame, end-to-end)`: {summary['metrics']['T_inf_temporal_ms']:.4f}",
            f"- `T_inf_hybrid(ms/frame, end-to-end)`: {summary['metrics']['T_inf_hybrid_ms']:.4f}",
            f"- `T_detector_pack(ms/frame)`: {summary['metrics']['T_detector_pack_ms']:.4f}",
            f"- `N_param_rule`: {summary['metrics']['N_param_rule']}",
            f"- `N_param_temporal`: {summary['metrics']['N_param_temporal']}",
            f"- `N_param_hybrid`: {summary['metrics']['N_param_hybrid']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_detector_map50(path: Path) -> float:
    if not path.exists():
        return 0.0
    last_value = 0.0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "mAP50(B)" in line:
            parts = line.split("`")
            for part in parts:
                try:
                    last_value = float(part)
                except ValueError:
                    continue
    return last_value


def _count_params(module: object) -> int:
    if not hasattr(module, "parameters"):
        return 0
    try:
        return int(sum(param.numel() for param in module.parameters()))  # type: ignore[attr-defined]
    except Exception:
        return 0


def _plot_comparison(rows: list[dict[str, object]], path: Path) -> None:
    label_map = _label_index_map()
    frames = [int(row["frame_id"]) for row in rows]
    rule_labels = [label_map[str(row["rule_label"])] for row in rows]
    baseline_labels = [label_map[str(row["baseline_label"])] for row in rows]
    final_labels = [label_map[str(row["final_label"])] for row in rows]

    rule_warn = [float(row["rule_warn_score"]) for row in rows]
    baseline_warn = [float(row["baseline_warn_prob"]) for row in rows]
    final_warn = [float(row["final_warn_prob"]) for row in rows]

    rule_hard = [float(row["rule_hard_score"]) for row in rows]
    baseline_hard = [float(row["baseline_hard_prob"]) for row in rows]
    final_hard = [float(row["final_hard_prob"]) for row in rows]
    final_boundary = [float(row["final_boundary"]) for row in rows]

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(frames, rule_labels, label="1. YOLO+rule", linewidth=1.4)
    axes[0].plot(frames, baseline_labels, label="2. CNN-GRU", linewidth=1.4)
    axes[0].plot(frames, final_labels, label="3. CNN-GRU+rule", linewidth=1.6)
    axes[0].set_ylabel("label idx")
    axes[0].set_yticks(list(label_map.values()), list(label_map.keys()))
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(frames, rule_warn, label="1 warn_score", linewidth=1.4)
    axes[1].plot(frames, baseline_warn, label="2 warn_prob", linewidth=1.4)
    axes[1].plot(frames, final_warn, label="3 warn_prob", linewidth=1.6)
    axes[1].set_ylabel("warn signal")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right")

    axes[2].plot(frames, rule_hard, label="1 hard_score", linewidth=1.4)
    axes[2].plot(frames, baseline_hard, label="2 hard_prob", linewidth=1.4)
    axes[2].plot(frames, final_hard, label="3 hard_prob", linewidth=1.6)
    axes[2].set_ylabel("hard signal")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="upper right")

    axes[3].plot(frames, final_boundary, label="3 boundary", linewidth=1.6)
    axes[3].set_ylabel("boundary")
    axes[3].set_ylim(0.0, 1.0)
    axes[3].set_xlabel("frame")
    axes[3].grid(alpha=0.25)
    axes[3].legend(loc="upper right")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _render_event_frame(cv2_module, frame, frame_id: int, rule_pred: dict[str, object], baseline_pred: dict[str, object], final_pred: dict[str, object]):
    canvas = frame.copy()
    cv2_module.rectangle(canvas, (8, 8), (830, 146), (18, 18, 18), thickness=-1)
    cv2_module.putText(canvas, f"frame={frame_id}", (18, 30), cv2_module.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2_module.LINE_AA)
    cv2_module.putText(canvas, f"1.rule={rule_pred['label']} ({float(rule_pred['warn_score']):.3f}/{float(rule_pred['hard_score']):.3f})", (18, 56), cv2_module.FONT_HERSHEY_SIMPLEX, 0.56, (255, 210, 80), 1, cv2_module.LINE_AA)
    cv2_module.putText(canvas, f"2.temporal={baseline_pred['pure_label']} (b={float(baseline_pred['boundary']):.3f})", (18, 82), cv2_module.FONT_HERSHEY_SIMPLEX, 0.56, (150, 220, 255), 1, cv2_module.LINE_AA)
    cv2_module.putText(canvas, f"3.final={final_pred['final_label']} (w={float(final_pred['warn_prob']):.3f}, h={float(final_pred['hard_prob']):.3f})", (18, 108), cv2_module.FONT_HERSHEY_SIMPLEX, 0.56, (120, 255, 160), 1, cv2_module.LINE_AA)
    cv2_module.putText(canvas, f"decision={final_pred['decision_reason']}", (18, 134), cv2_module.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, cv2_module.LINE_AA)
    return canvas


def _finalize_event(cv2_module, event: dict[str, object], clips_dir: Path) -> dict[str, object]:
    frames = event["frames"]
    if not isinstance(frames, list) or not frames:
        return event
    first_frame = frames[0]["image"]
    height, width = first_frame.shape[:2]
    clip_path = clips_dir / f"{event['event_id']}.mp4"
    fourcc = cv2_module.VideoWriter_fourcc(*"mp4v")
    writer = cv2_module.VideoWriter(str(clip_path), fourcc, float(event["fps"]), (width, height))
    try:
        for item in frames:
            writer.write(item["image"])
    finally:
        writer.release()
    event["clip_path"] = str(clip_path)
    event["start_frame"] = int(frames[0]["frame_id"])
    event["end_frame"] = int(frames[-1]["frame_id"])
    event["frame_count"] = len(frames)
    event.pop("frames", None)
    return event


def main() -> None:
    cv2 = _import_cv2()
    spatial = _make_spatial_encoder()
    packer = SimpleFeaturePacker()
    baseline_model, baseline_labels, baseline_window = load_model(Path(SHARED_TEMPORAL_CKPT))
    final_model, final_labels, final_window = load_model(Path(FINAL_TEMPORAL_CKPT))
    params = load_hybrid_params(Path(HYBRID_MANIFEST))
    rule_params = InstantRuleParams()

    capture, source_name = _open_capture(cv2)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 15.0

    video_stem = _safe_stem(source_name)
    run_dir = SAVE_ROOT / f"{video_stem}_three_model_compare"
    run_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir = run_dir / "screenshots"
    clips_dir = run_dir / "clips"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    baseline_buffer: deque[list[float]] = deque(maxlen=baseline_window)
    final_buffer: deque[list[float]] = deque(maxlen=final_window)
    prebuffer: deque[dict[str, object]] = deque(maxlen=CLIP_PRE_FRAMES)

    rows: list[dict[str, object]] = []
    event_records: list[dict[str, object]] = []
    rule_time_sum = 0.0
    temporal_time_sum = 0.0
    hybrid_time_sum = 0.0
    detector_pack_time_sum = 0.0
    frame_id = 0
    active_event: dict[str, object] | None = None
    last_event_frame = -MIN_EVENT_GAP

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            if MAX_FRAMES > 0 and frame_id >= MAX_FRAMES:
                break

            frame_start = time.perf_counter()

            packet = FramePacket(
                frame_id=frame_id,
                sensor_timestamp=frame_id / max(1.0, fps),
                system_timestamp=time.time(),
                pixels=[],
                raw_path=f"{source_name}#frame={frame_id}",
                image=frame_bgr,
            )

            spatial_vector = spatial.encode(packet)
            detection = spatial.get_last_detection()
            tick = time.perf_counter()
            rule_pred = predict_instant_rule(detection, rule_params)
            packed = packer.pack(packet, spatial_vector)
            detector_and_pack_time = time.perf_counter() - frame_start
            detector_pack_time_sum += detector_and_pack_time
            rule_only_time = time.perf_counter() - tick
            rule_time_sum += detector_and_pack_time + rule_only_time
            feature_vector = list(packed.spatial_vector)

            baseline_buffer.append(feature_vector)
            while len(baseline_buffer) < baseline_window:
                baseline_buffer.appendleft([0.0] * len(feature_vector))
            final_buffer.append(feature_vector)
            while len(final_buffer) < final_window:
                final_buffer.appendleft([0.0] * len(feature_vector))

            tick = time.perf_counter()
            baseline_pred = predict_sequence(baseline_model, baseline_labels, list(baseline_buffer), params, rule_result=None)
            temporal_only_time = time.perf_counter() - tick
            temporal_time_sum += detector_and_pack_time + temporal_only_time
            tick = time.perf_counter()
            final_pred = predict_sequence(final_model, final_labels, list(final_buffer), params, rule_result=rule_pred)
            hybrid_only_time = time.perf_counter() - tick
            hybrid_time_sum += detector_and_pack_time + rule_only_time + hybrid_only_time
            event_frame = _render_event_frame(cv2, frame_bgr, frame_id, rule_pred, baseline_pred, final_pred)
            prebuffer.append({"frame_id": frame_id, "image": event_frame.copy()})

            final_event = str(final_pred["final_label"]) in EVENT_LABELS
            if active_event is None and final_event and (frame_id - last_event_frame) >= MIN_EVENT_GAP:
                event_id = f"event_{len(event_records)+1:03d}_frame_{frame_id:04d}"
                screenshot_path = screenshots_dir / f"{event_id}.jpg"
                cv2.imwrite(str(screenshot_path), event_frame)
                active_event = {
                    "event_id": event_id,
                    "trigger_frame": frame_id,
                    "trigger_models": ["final_hybrid"],
                    "fps": fps,
                    "screenshot_path": str(screenshot_path),
                    "rule_label": str(rule_pred["label"]),
                    "baseline_label": str(baseline_pred["pure_label"]),
                    "final_label": str(final_pred["final_label"]),
                    "final_reason": str(final_pred["decision_reason"]),
                    "frames": list(prebuffer),
                    "remaining_post_frames": CLIP_POST_FRAMES,
                }
                last_event_frame = frame_id

            if active_event is not None:
                active_frames = active_event.get("frames")
                if isinstance(active_frames, list):
                    if not active_frames or int(active_frames[-1]["frame_id"]) != frame_id:
                        active_frames.append({"frame_id": frame_id, "image": event_frame.copy()})
                active_event["remaining_post_frames"] = int(active_event["remaining_post_frames"]) - 1
                if int(active_event["remaining_post_frames"]) <= 0:
                    event_records.append(_finalize_event(cv2, active_event, clips_dir))
                    active_event = None

            rows.append(
                {
                    "frame_id": frame_id,
                    "rule_label": str(rule_pred["label"]),
                    "rule_warn_score": float(rule_pred["warn_score"]),
                    "rule_hard_score": float(rule_pred["hard_score"]),
                    "rule_reason": str(rule_pred["reason"]),
                    "baseline_label": str(baseline_pred["pure_label"]),
                    "baseline_prob": float(baseline_pred["pure_prob"]),
                    "baseline_boundary": float(baseline_pred["boundary"]),
                    "baseline_warn_prob": float(baseline_pred["warn_prob"]),
                    "baseline_hard_prob": float(baseline_pred["hard_prob"]),
                    "baseline_follow_prob": float(baseline_pred["follow_prob"]),
                    "final_label": str(final_pred["final_label"]),
                    "final_prob": float(final_pred["final_prob"]),
                    "final_pure_label": str(final_pred["pure_label"]),
                    "final_pure_prob": float(final_pred["pure_prob"]),
                    "final_boundary": float(final_pred["boundary"]),
                    "final_warn_prob": float(final_pred["warn_prob"]),
                    "final_hard_prob": float(final_pred["hard_prob"]),
                    "final_follow_prob": float(final_pred["follow_prob"]),
                    "final_decision_reason": str(final_pred["decision_reason"]),
                    "final_warn_reason": str(final_pred["warn_reason"]),
                    "final_hard_reason": str(final_pred["hard_reason"]),
                }
            )
            if frame_id % 50 == 0:
                print(f"[compare] processed frame {frame_id}")
            frame_id += 1
    finally:
        capture.release()

    if active_event is not None:
        event_records.append(_finalize_event(cv2, active_event, clips_dir))

    csv_path = run_dir / "frame_comparison.csv"
    json_path = run_dir / "summary.json"
    md_path = run_dir / "summary.md"
    fig_path = run_dir / "comparison_plot.png"
    events_path = run_dir / "event_records.json"

    _save_csv(csv_path, rows)
    _plot_comparison(rows, fig_path)

    summary = {
        "source": source_name,
        "frame_count": len(rows),
        "label_counts": {
            "yolo_rule": dict(Counter(str(row["rule_label"]) for row in rows)),
            "cnn_gru_only": dict(Counter(str(row["baseline_label"]) for row in rows)),
            "cnn_gru_plus_rule": dict(Counter(str(row["final_label"]) for row in rows)),
        },
        "event_count": len(event_records),
        "metrics": {
            "A_det": _parse_detector_map50(DETECTOR_RESULT_SUMMARY),
            "T_inf_rule_ms": round((rule_time_sum / max(1, len(rows))) * 1000.0, 4),
            "T_inf_temporal_ms": round((temporal_time_sum / max(1, len(rows))) * 1000.0, 4),
            "T_inf_hybrid_ms": round((hybrid_time_sum / max(1, len(rows))) * 1000.0, 4),
            "T_detector_pack_ms": round((detector_pack_time_sum / max(1, len(rows))) * 1000.0, 4),
            "N_param_rule": _count_params(spatial._model.model),  # type: ignore[attr-defined]
            "N_param_temporal": _count_params(spatial._model.model) + _count_params(baseline_model),  # type: ignore[attr-defined]
            "N_param_hybrid": _count_params(spatial._model.model) + _count_params(final_model),  # type: ignore[attr-defined]
        },
        "artifacts": {
            "csv": str(csv_path),
            "json": str(json_path),
            "markdown": str(md_path),
            "figure": str(fig_path),
            "events": str(events_path),
            "screenshots_dir": str(screenshots_dir),
            "clips_dir": str(clips_dir),
        },
    }

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _save_markdown(md_path, summary)
    events_path.write_text(json.dumps(event_records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[compare] saved csv: {csv_path}")
    print(f"[compare] saved summary: {json_path}")
    print(f"[compare] saved figure: {fig_path}")
    print(f"[compare] saved events: {events_path}")


if __name__ == "__main__":
    main()
