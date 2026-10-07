from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vcp.components.temporal import TemporalGRUNet, build_temporal_channel_groups
from vcp.features.semantic_v1 import V1_KEYS

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

    roi_thr: float
    center_thr: float
    looming_warn_thr: float
    looming_hard_thr: float
    occlusion_thr: float
    motion_stop_thr: float
    boundary_thr: float
    warn_boundary_thr: float
    hard_boundary_thr: float
    warn_prob_thr: float
    hard_prob_thr: float
    follow_prob_thr: float
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
        roi_thr=0.30,
        center_thr=0.72,
        looming_warn_thr=0.006,
        looming_hard_thr=0.014,
        occlusion_thr=0.58,
        motion_stop_thr=0.01,
        boundary_thr=0.50,
        warn_boundary_thr=0.82,
        hard_boundary_thr=0.93,
        warn_prob_thr=0.05,
        hard_prob_thr=0.03,
        follow_prob_thr=0.20,
        warn_boost=0.22,
        hard_boost=0.30,
    )
    if not manifest_path.exists():
        return default
    data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    params = data.get("hybrid_rule_params", {})
    return HybridParams(
        roi_thr=float(params.get("roi_thr", default.roi_thr)),
        center_thr=float(params.get("center_thr", default.center_thr)),
        looming_warn_thr=float(params.get("looming_warn_thr", default.looming_warn_thr)),
        looming_hard_thr=float(params.get("looming_hard_thr", default.looming_hard_thr)),
        occlusion_thr=float(params.get("occlusion_thr", default.occlusion_thr)),
        motion_stop_thr=float(params.get("motion_stop_thr", default.motion_stop_thr)),
        boundary_thr=float(params.get("boundary_thr", default.boundary_thr)),
        warn_boundary_thr=float(params.get("warn_boundary_thr", default.warn_boundary_thr)),
        hard_boundary_thr=float(params.get("hard_boundary_thr", default.hard_boundary_thr)),
        warn_prob_thr=float(params.get("warn_prob_thr", default.warn_prob_thr)),
        hard_prob_thr=float(params.get("hard_prob_thr", default.hard_prob_thr)),
        follow_prob_thr=float(params.get("follow_prob_thr", default.follow_prob_thr)),
        warn_boost=float(params.get("warn_boost", default.warn_boost)),
        hard_boost=float(params.get("hard_boost", default.hard_boost)),
    )


def load_model(checkpoint_path: Path) -> tuple[TemporalGRUNet, list[str], int]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint를 찾을 수 없습니다: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    feature_keys: list[str] | None = None,
) -> dict[str, Any]:
    """순수 모델 예측 + 하이브리드 룰 게이트 최종 예측(v1 특징 기준)."""

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

    keys = list(feature_keys) if feature_keys is not None else list(V1_KEYS)
    last = feature_window[-1]

    def _feat(name: str) -> float:
        # 인덱스 하드코딩 대신 key 이름으로 조회한다(특징 버전 변경 시 오동작 방지).
        idx = keys.index(name) if name in keys else -1
        return float(last[idx]) if 0 <= idx < len(last) else 0.0

    roi_risk = _feat("roi_risk")
    motion_delta = _feat("motion_delta")
    center = _feat("center_closeness")
    looming = _feat("looming_score")
    occlusion = _feat("occlusion_score")

    warn_rule_feat = (
        roi_risk >= params.roi_thr
        and center >= params.center_thr
        and looming >= params.looming_warn_thr
    )
    hard_rule_feat = (
        warn_rule_feat
        and looming >= params.looming_hard_thr
        and (occlusion >= params.occlusion_thr or motion_delta <= params.motion_stop_thr)
    )

    warn_rule_signal = (
        boundary >= params.warn_boundary_thr
        and (
            probs[follow_idx] >= params.follow_prob_thr
            or probs[warn_idx] >= params.warn_prob_thr
        )
    )
    hard_rule_signal = (
        boundary >= params.hard_boundary_thr
        and (
            probs[follow_idx] >= params.follow_prob_thr
            or probs[warn_idx] >= params.warn_prob_thr
            or probs[hard_idx] >= params.hard_prob_thr
        )
    )

    warn_rule = warn_rule_feat or warn_rule_signal
    hard_rule = hard_rule_feat or hard_rule_signal

    boosted = probs[:]
    boosted[warn_idx] += params.warn_boost if warn_rule else 0.0
    boosted[hard_idx] += params.hard_boost if hard_rule else 0.0

    pure_idx = max(range(len(probs)), key=lambda idx: probs[idx])
    final_idx = max(range(len(boosted)), key=lambda idx: boosted[idx])

    if hard_rule and (probs[hard_idx] >= params.hard_prob_thr or boundary >= params.hard_boundary_thr):
        final_idx = hard_idx
    elif (not hard_rule) and warn_rule and (
        probs[warn_idx] >= params.warn_prob_thr or boundary >= params.warn_boundary_thr
    ):
        final_idx = warn_idx

    return {
        "pure_label": labels[pure_idx],
        "pure_prob": probs[pure_idx],
        "final_label": labels[final_idx],
        "boundary": boundary,
        "warn_prob": probs[warn_idx],
        "hard_prob": probs[hard_idx],
        "follow_prob": probs[follow_idx],
        "roi_risk": roi_risk,
        "center": center,
        "looming": looming,
        "occlusion": occlusion,
        "motion_delta": motion_delta,
        "warn_rule": warn_rule,
        "hard_rule": hard_rule,
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
        draw_text(cv2, panel, 18, 152, f"hybrid final: {pred['final_label']}", final_color, 0.72)

        draw_text(cv2, panel, 18, 192, f"boundary: {pred['boundary']:.3f}", (240, 240, 240), 0.56)
        draw_text(cv2, panel, 18, 220, f"warn_prob: {pred['warn_prob']:.3f}", (240, 210, 100), 0.56)
        draw_text(cv2, panel, 18, 248, f"hard_prob: {pred['hard_prob']:.3f}", (255, 150, 120), 0.56)
        draw_text(cv2, panel, 18, 276, f"follow_prob: {pred['follow_prob']:.3f}", (150, 220, 255), 0.56)

        draw_text(cv2, panel, 18, 320, f"roi={pred['roi_risk']:.3f} center={pred['center']:.3f}", (220, 220, 220), 0.52)
        draw_text(cv2, panel, 18, 346, f"looming={pred['looming']:.3f} occlusion={pred['occlusion']:.3f}", (220, 220, 220), 0.52)
        draw_text(cv2, panel, 18, 372, f"motion_delta={pred['motion_delta']:.3f}", (220, 220, 220), 0.52)
        draw_text(cv2, panel, 18, 402, f"warn_rule={int(pred['warn_rule'])} hard_rule={int(pred['hard_rule'])}", (180, 255, 180), 0.56)

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
            y=440,
            w=460,
            h=90,
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
            y=560,
            w=460,
            h=70,
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
            y=660,
            w=460,
            h=70,
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
