from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vcp.components.event_rules import InstantRuleParams, predict_instant_rule
from vcp.components.temporal import TemporalGRUNet, build_temporal_channel_groups

ROOT_DIR = Path(__file__).resolve().parents[3]


def _import_cv2_np() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "최종 모델 확인 UI 실행에는 opencv-python과 numpy가 필요합니다."
        ) from exc
    return cv2, np


@dataclass(slots=True)
class UIArgs:
    """최종 하이브리드 모델 UI 인자."""

    run_dir: Path | None
    video: Path | None
    checkpoint: Path
    manifest: Path
    start_frame: int
    play_fps: float
    export_preview: Path | None


@dataclass(slots=True)
class HybridParams:
    """하이브리드 승격 규칙 파라미터."""

    boundary_thr: float
    warn_boundary_thr: float
    hard_boundary_thr: float
    warn_prob_thr: float
    hard_prob_thr: float
    follow_prob_thr: float
    rule_warn_score_thr: float
    rule_hard_score_thr: float
    warn_boost: float
    hard_boost: float


def parse_args() -> UIArgs:
    parser = argparse.ArgumentParser(description="최종 하이브리드 모델 확인 UI")
    parser.add_argument("--run-dir", type=str, default="", help="frame_records.jsonl이 있는 run 디렉터리")
    parser.add_argument("--video", type=str, default="", help="원본 영상 경로")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(ROOT_DIR / "models" / "checkpoints" / "temporal_final_best.pt"),
        help="순수 제안모델 checkpoint",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(ROOT_DIR / "configs" / "hybrid_rule_params.json"),
        help="하이브리드 룰 파라미터가 있는 benchmark manifest 경로",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--play-fps", type=float, default=15.0)
    parser.add_argument(
        "--export-preview",
        type=str,
        default="",
        help="첫 화면을 이미지로 저장하고 종료",
    )
    args = parser.parse_args()
    return UIArgs(
        run_dir=Path(args.run_dir) if args.run_dir else None,
        video=Path(args.video) if args.video else None,
        checkpoint=Path(args.checkpoint),
        manifest=Path(args.manifest),
        start_frame=max(0, int(args.start_frame)),
        play_fps=max(1.0, float(args.play_fps)),
        export_preview=Path(args.export_preview) if args.export_preview else None,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def find_latest_braking_run(outputs_root: Path) -> Path:
    if not outputs_root.exists():
        raise FileNotFoundError(f"outputs/runs가 없습니다: {outputs_root}")
    run_dirs = [
        path
        for path in outputs_root.iterdir()
        if path.is_dir() and "braking" in path.name and (path / "frame_records.jsonl").exists()
    ]
    if not run_dirs:
        raise FileNotFoundError("브레이크 관련 run을 찾지 못했습니다.")
    run_dirs.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return run_dirs[0]


def infer_video_path(frame_rows: list[dict[str, Any]], explicit_video: Path | None) -> Path:
    if explicit_video is not None:
        return explicit_video
    if not frame_rows:
        raise ValueError("frame_records가 비어 있습니다.")
    metadata = frame_rows[0].get("metadata", {})
    source_video = metadata.get("source_video")
    if source_video:
        return Path(str(source_video))
    raw_path = str(frame_rows[0].get("raw_path", ""))
    if "#frame=" in raw_path:
        return Path("data/raw/videos") / raw_path.split("#frame=", 1)[0]
    raise ValueError("원본 영상 경로를 추론하지 못했습니다. --video를 지정하세요.")


def load_hybrid_params(manifest_path: Path) -> HybridParams:
    default = HybridParams(
        boundary_thr=0.50,
        warn_boundary_thr=0.82,
        hard_boundary_thr=0.93,
        warn_prob_thr=0.05,
        hard_prob_thr=0.03,
        follow_prob_thr=0.20,
        rule_warn_score_thr=0.50,
        rule_hard_score_thr=0.65,
        warn_boost=0.22,
        hard_boost=0.30,
    )
    if not manifest_path.exists():
        return default
    data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    params = data.get("hybrid_rule_params", {})
    return HybridParams(
        boundary_thr=float(params.get("boundary_thr", default.boundary_thr)),
        warn_boundary_thr=float(params.get("warn_boundary_thr", default.warn_boundary_thr)),
        hard_boundary_thr=float(params.get("hard_boundary_thr", default.hard_boundary_thr)),
        warn_prob_thr=float(params.get("warn_prob_thr", default.warn_prob_thr)),
        hard_prob_thr=float(params.get("hard_prob_thr", default.hard_prob_thr)),
        follow_prob_thr=float(params.get("follow_prob_thr", default.follow_prob_thr)),
        rule_warn_score_thr=float(params.get("rule_warn_score_thr", default.rule_warn_score_thr)),
        rule_hard_score_thr=float(params.get("rule_hard_score_thr", default.rule_hard_score_thr)),
        warn_boost=float(params.get("warn_boost", default.warn_boost)),
        hard_boost=float(params.get("hard_boost", default.hard_boost)),
    )


def load_model(checkpoint_path: Path) -> tuple[TemporalGRUNet, list[str], int]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint를 찾을 수 없습니다: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint.get("model_config", {})
    labels_map = checkpoint.get("labels", {})
    labels = [""] * len(labels_map)
    for name, index in labels_map.items():
        labels[int(index)] = str(name)
    input_dim = int(model_config.get("input_dim", 16))
    model = TemporalGRUNet(
        input_dim=input_dim,
        hidden_dim=int(model_config.get("hidden_dim", 96)),
        num_contexts=int(model_config.get("num_contexts", len(labels))),
        dropout=float(model_config.get("dropout", 0.1)),
        cnn_channels=int(model_config.get("cnn_channels", 24)),
        channel_groups=model_config.get(
            "channel_groups",
            build_temporal_channel_groups(input_dim),
        ),
        gru_layers=int(model_config.get("gru_layers", 1)),
        head_hidden_dim=int(model_config.get("head_hidden_dim", 0)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    window_size = int(model_config.get("window_size", 8))
    return model, labels, window_size


def draw_text(
    cv2_module: Any,
    image: Any,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int] = (235, 235, 235),
    scale: float = 0.56,
) -> None:
    cv2_module.putText(
        image,
        text,
        (x, y),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2_module.LINE_AA,
    )


def draw_series(
    cv2_module: Any,
    np_module: Any,
    canvas: Any,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    values: list[float],
    color: tuple[int, int, int],
    label: str,
    vmin: float,
    vmax: float,
    ref_line: float | None = None,
) -> None:
    cv2_module.rectangle(canvas, (x, y), (x + w, y + h), (80, 80, 80), 1)
    draw_text(cv2_module, canvas, x + 4, y - 8, label, (200, 200, 200), 0.45)
    if ref_line is not None and vmax > vmin:
        ratio = max(0.0, min(1.0, (ref_line - vmin) / (vmax - vmin)))
        yy = y + h - int(ratio * h)
        cv2_module.line(canvas, (x, yy), (x + w, yy), (120, 170, 255), 1)
    if len(values) < 2:
        return
    points = []
    denom = max(1, len(values) - 1)
    for idx, value in enumerate(values):
        xr = x + int((idx / denom) * w)
        ratio = 0.0 if vmax <= vmin else max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
        yr = y + h - int(ratio * h)
        points.append((xr, yr))
    cv2_module.polylines(canvas, [np_module.array(points, dtype=np_module.int32)], False, color, 2)


def predict_sequence(
    model: TemporalGRUNet,
    labels: list[str],
    feature_window: list[list[float]],
    params: HybridParams,
    rule_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    x = torch.tensor([feature_window], dtype=torch.float32)
    with torch.no_grad():
        output = model(x)
        probs_tensor = torch.softmax(output["context_logits"], dim=-1)[0]
        boundary = float(torch.sigmoid(output["boundary_logit"])[0].item())

    probs = [float(value) for value in probs_tensor.tolist()]
    label_to_index = {name: idx for idx, name in enumerate(labels)}
    follow_idx = label_to_index.get("front_vehicle_follow", 0)
    warn_idx = label_to_index.get("brake_warning", 0)
    hard_idx = label_to_index.get("hard_brake_risk", 0)

    last = feature_window[-1]
    roi_risk = float(last[11]) if len(last) > 11 else 0.0
    center = float(last[12]) if len(last) > 12 else 0.0
    roi_mean_area = float(last[13]) if len(last) > 13 else 0.0
    roi_count_norm = float(last[14]) if len(last) > 14 else 0.0
    roi_vertical_bias = float(last[15]) if len(last) > 15 else 0.0
    warn_signal_reasons: list[str] = []
    hard_signal_reasons: list[str] = []

    warn_rule_signal = (
        boundary >= params.warn_boundary_thr
        and (
            probs[follow_idx] >= params.follow_prob_thr
            or probs[warn_idx] >= params.warn_prob_thr
        )
    )
    if boundary >= params.warn_boundary_thr:
        warn_signal_reasons.append("boundary")
    if probs[follow_idx] >= params.follow_prob_thr:
        warn_signal_reasons.append("follow_prob")
    if probs[warn_idx] >= params.warn_prob_thr:
        warn_signal_reasons.append("warn_prob")

    hard_rule_signal = (
        boundary >= params.hard_boundary_thr
        and (
            probs[follow_idx] >= params.follow_prob_thr
            or probs[warn_idx] >= params.warn_prob_thr
            or probs[hard_idx] >= params.hard_prob_thr
        )
    )
    if boundary >= params.hard_boundary_thr:
        hard_signal_reasons.append("boundary")
    if probs[follow_idx] >= params.follow_prob_thr:
        hard_signal_reasons.append("follow_prob")
    if probs[warn_idx] >= params.warn_prob_thr:
        hard_signal_reasons.append("warn_prob")
    if probs[hard_idx] >= params.hard_prob_thr:
        hard_signal_reasons.append("hard_prob")

    rule_label = str((rule_result or {}).get("label", "normal_drive"))
    rule_warn_score = float((rule_result or {}).get("warn_score", 0.0))
    rule_hard_score = float((rule_result or {}).get("hard_score", 0.0))
    rule_reason = str((rule_result or {}).get("reason", "none"))

    warn_rule = (
        rule_label in {"brake_warning", "hard_brake_risk"}
        and rule_warn_score >= params.rule_warn_score_thr
    ) or warn_rule_signal
    hard_rule = (
        rule_label == "hard_brake_risk"
        and rule_hard_score >= params.rule_hard_score_thr
    ) or hard_rule_signal

    pure_idx = max(range(len(probs)), key=lambda idx: probs[idx])
    final_idx = pure_idx
    decision_reason = f"pure:{labels[pure_idx]}"

    weak_warn_prediction = (
        pure_idx == warn_idx
        and not warn_rule
        and boundary < params.warn_boundary_thr
        and rule_warn_score < params.rule_warn_score_thr
    )
    if weak_warn_prediction:
        if probs[follow_idx] >= probs[0]:
            final_idx = follow_idx
            decision_reason = "demote:weak_warn->follow"
        else:
            final_idx = 0
            decision_reason = "demote:weak_warn->normal"

    if pure_idx != hard_idx and hard_rule and (
        probs[hard_idx] >= params.hard_prob_thr or boundary >= params.hard_boundary_thr
    ):
        final_idx = hard_idx
        if rule_label == "hard_brake_risk" and rule_hard_score >= params.rule_hard_score_thr:
            decision_reason = f"promote:hard_rule({rule_reason})"
        else:
            decision_reason = f"promote:hard_signal({'+'.join(hard_signal_reasons)})"
    elif pure_idx not in {warn_idx, hard_idx} and warn_rule and (
        probs[warn_idx] >= params.warn_prob_thr or boundary >= params.warn_boundary_thr
    ):
        final_idx = warn_idx
        if rule_label in {"brake_warning", "hard_brake_risk"} and rule_warn_score >= params.rule_warn_score_thr:
            decision_reason = f"promote:warn_rule({rule_reason})"
        else:
            decision_reason = f"promote:warn_signal({'+'.join(warn_signal_reasons)})"

    warn_reason = "none"
    hard_reason = "none"
    if rule_label in {"brake_warning", "hard_brake_risk"} and rule_warn_score >= params.rule_warn_score_thr:
        warn_reason = f"rule:{rule_reason}"
    elif warn_rule_signal:
        warn_reason = f"signal:{'+'.join(warn_signal_reasons)}"
    if rule_label == "hard_brake_risk" and rule_hard_score >= params.rule_hard_score_thr:
        hard_reason = f"rule:{rule_reason}"
    elif hard_rule_signal:
        hard_reason = f"signal:{'+'.join(hard_signal_reasons)}"

    return {
        "pure_label": labels[pure_idx],
        "pure_prob": probs[pure_idx],
        "final_label": labels[final_idx],
        "final_prob": probs[final_idx],
        "decision_reason": decision_reason,
        "boundary": boundary,
        "warn_prob": probs[warn_idx],
        "hard_prob": probs[hard_idx],
        "follow_prob": probs[follow_idx],
        "roi_risk": roi_risk,
        "center": center,
        "roi_mean_area": roi_mean_area,
        "roi_count_norm": roi_count_norm,
        "roi_vertical_bias": roi_vertical_bias,
        "warn_rule": warn_rule,
        "hard_rule": hard_rule,
        "warn_reason": warn_reason,
        "hard_reason": hard_reason,
        "rule_label": rule_label,
        "rule_warn_score": rule_warn_score,
        "rule_hard_score": rule_hard_score,
    }


def run_ui(args: UIArgs) -> None:
    cv2, np = _import_cv2_np()
    run_dir = args.run_dir or find_latest_braking_run(Path("outputs") / "runs")
    frame_rows = read_jsonl(run_dir / "frame_records.jsonl")
    frame_rows.sort(key=lambda row: int(row.get("frame_id", 0)))
    video_path = infer_video_path(frame_rows, args.video)

    if not video_path.exists():
        raise FileNotFoundError(f"원본 영상이 없습니다: {video_path}")

    model, labels, window_size = load_model(args.checkpoint)
    params = load_hybrid_params(args.manifest)

    feature_vectors = [
        [float(v) for v in row.get("derived_feature", {}).get("feature_vector", [])]
        for row in frame_rows
    ]
    feature_dim = max((len(vec) for vec in feature_vectors), default=16)
    normalized_vectors: list[list[float]] = []
    for vec in feature_vectors:
        padded = vec[:feature_dim] + [0.0] * max(0, feature_dim - len(vec))
        normalized_vectors.append(padded)

    predictions: list[dict[str, Any]] = []
    zero_vec = [0.0] * feature_dim
    for idx in range(len(frame_rows)):
        start = max(0, idx - window_size + 1)
        window = normalized_vectors[start : idx + 1]
        if len(window) < window_size:
            window = ([zero_vec] * (window_size - len(window))) + window
        predictions.append(predict_sequence(model, labels, window, params))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"비디오를 열 수 없습니다: {video_path}")

    frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    current = min(max(0, args.start_frame), max(0, frame_total - 1))
    playing = False
    delay_ms = max(1, int(1000 / args.play_fps))

    def render(frame_index: int) -> Any:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"프레임을 읽지 못했습니다: {frame_index}")

        row = frame_rows[frame_index]
        pred = predictions[frame_index]
        canvas = frame.copy()
        h, w = canvas.shape[:2]
        panel_w = 520
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        panel[:] = (20, 20, 20)

        draw_text(cv2, panel, 18, 30, f"run: {run_dir.name}", (200, 230, 255), 0.52)
        draw_text(cv2, panel, 18, 58, f"frame: {frame_index}/{max(0, frame_total - 1)}", (230, 230, 230), 0.54)
        draw_text(cv2, panel, 18, 92, f"stored tag: {row.get('context_tag', '-')}", (170, 210, 255), 0.6)
        draw_text(cv2, panel, 18, 122, f"pure model: {pred['pure_label']} ({pred['pure_prob']:.3f})", (120, 255, 160), 0.64)
        final_color = (70, 220, 255) if pred["final_label"] != pred["pure_label"] else (255, 220, 120)
        draw_text(cv2, panel, 18, 152, f"hybrid final: {pred['final_label']} ({pred['final_prob']:.3f})", final_color, 0.72)
        draw_text(cv2, panel, 18, 178, f"decision: {pred['decision_reason']}", (220, 220, 220), 0.48)

        draw_text(cv2, panel, 18, 208, f"boundary: {pred['boundary']:.3f}", (240, 240, 240), 0.56)
        draw_text(cv2, panel, 18, 236, f"warn_prob: {pred['warn_prob']:.3f}", (240, 210, 100), 0.56)
        draw_text(cv2, panel, 18, 264, f"hard_prob: {pred['hard_prob']:.3f}", (255, 150, 120), 0.56)
        draw_text(cv2, panel, 18, 292, f"follow_prob: {pred['follow_prob']:.3f}", (150, 220, 255), 0.56)

        draw_text(cv2, panel, 18, 336, f"roi={pred['roi_risk']:.3f} center={pred['center']:.3f}", (220, 220, 220), 0.52)
        draw_text(cv2, panel, 18, 362, f"roi_mean_area={pred['roi_mean_area']:.3f}", (220, 220, 220), 0.52)
        draw_text(cv2, panel, 18, 388, f"roi_count_norm={pred['roi_count_norm']:.3f} roi_y={pred['roi_vertical_bias']:.3f}", (220, 220, 220), 0.52)
        draw_text(cv2, panel, 18, 418, f"warn_rule={int(pred['warn_rule'])} hard_rule={int(pred['hard_rule'])}", (180, 255, 180), 0.56)
        draw_text(cv2, panel, 18, 444, f"warn_reason: {pred['warn_reason']}", (255, 220, 150), 0.48)
        draw_text(cv2, panel, 18, 468, f"hard_reason: {pred['hard_reason']}", (255, 170, 150), 0.48)

        window_start = max(0, frame_index - 119)
        hist = predictions[window_start : frame_index + 1]
        boundary_series = [item["boundary"] for item in hist]
        warn_series = [item["warn_prob"] for item in hist]
        hard_series = [item["hard_prob"] for item in hist]

        draw_series(
            cv2,
            np,
            panel,
            x=18,
            y=500,
            w=460,
            h=80,
            values=boundary_series,
            color=(100, 200, 255),
            label="boundary",
            vmin=0.0,
            vmax=1.0,
            ref_line=params.warn_boundary_thr,
        )
        draw_series(
            cv2,
            np,
            panel,
            x=18,
            y=610,
            w=460,
            h=65,
            values=warn_series,
            color=(255, 210, 80),
            label="brake_warning prob",
            vmin=0.0,
            vmax=1.0,
            ref_line=params.warn_prob_thr,
        )
        draw_series(
            cv2,
            np,
            panel,
            x=18,
            y=700,
            w=460,
            h=65,
            values=hard_series,
            color=(255, 130, 110),
            label="hard_brake_risk prob",
            vmin=0.0,
            vmax=1.0,
            ref_line=params.hard_prob_thr,
        )

        draw_text(cv2, panel, 18, h - 32, "space:play/pause  a/d:-/+1  j/l:-/+30  q:quit", (180, 180, 180), 0.48)
        return np.concatenate([canvas, panel], axis=1)

    if args.export_preview is not None:
        preview = render(current)
        args.export_preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.export_preview), preview)
        capture.release()
        return

    window_name = "Final Hybrid Model UI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        current = max(0, min(current, frame_total - 1))
        preview = render(current)
        cv2.imshow(window_name, preview)

        key = cv2.waitKey(delay_ms if playing else 0) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            playing = not playing
        elif key == ord("a"):
            current = max(0, current - 1)
            playing = False
        elif key == ord("d"):
            current = min(frame_total - 1, current + 1)
            playing = False
        elif key == ord("j"):
            current = max(0, current - 30)
            playing = False
        elif key == ord("l"):
            current = min(frame_total - 1, current + 30)
            playing = False
        elif playing:
            current = min(frame_total - 1, current + 1)
            if current >= frame_total - 1:
                playing = False

    capture.release()
    cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    run_ui(args)


if __name__ == "__main__":
    main()
