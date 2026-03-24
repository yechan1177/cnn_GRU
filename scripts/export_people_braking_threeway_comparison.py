from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ultralytics import YOLO  # type: ignore

from vcp.components.spatial import YOLOSpatialEncoder
from vcp.config import SpatialConfig
from vcp.schemas import FramePacket
from vcp.tools.benchmark_context_models import LiteratureCNNGRUNet
from vcp.tools.final_hybrid_model_ui import load_hybrid_params, load_model, predict_sequence


YOLO_WEIGHTS = ROOT / "experiments/exp_011_yolo3cls_training/runs/yolov8n_3cls_from_pretrained_stage2_noaug/weights/best.pt"
YOLO_RESULT_SUMMARY = ROOT / "experiments/exp_011_yolo3cls_training/result_summary.md"
LITERATURE_CKPT = ROOT / "experiments/exp_017_hybrid_rule_gate_benchmark/runs/context_model_benchmark_hybrid_v2/literature_single_channel_cnn_gru_best.pt"
DEFAULT_PROPOSED_CKPT = ROOT / "experiments/exp_015_temporal_braking_expanded_feat16/runs/mcnn_gru_braking_expanded_feat16_t045_e40_h96_c24/checkpoints/best.pt"
DEFAULT_VIDEO_PATH = ROOT / "data/raw/videos/people_braking.mp4"
DEFAULT_ANNOTATION_PATH = ROOT / "data/annotations/context/people_braking_context_segments.jsonl"
CONF_THRESHOLD = 0.65
INPUT_SIZE = 640
FEATURE_DIM = 16
MAX_DET = 30
WINDOW_SIZE = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="특정 영상 기준 3모델 비교 산출")
    parser.add_argument("--video", type=str, default=str(DEFAULT_VIDEO_PATH))
    parser.add_argument("--annotation", type=str, default=str(DEFAULT_ANNOTATION_PATH))
    parser.add_argument("--proposed-ckpt", type=str, default=str(DEFAULT_PROPOSED_CKPT))
    parser.add_argument("--prefix", type=str, default="people_braking_threeway_comparison")
    return parser.parse_args()


def import_cv2() -> Any:
    import cv2

    return cv2


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def read_detector_map50() -> float:
    text = YOLO_RESULT_SUMMARY.read_text(encoding="utf-8", errors="ignore")
    values = re.findall(r"mAP50\(B\): `([0-9.]+)`", text)
    if not values:
        raise ValueError("YOLO result_summary에서 mAP50 값을 찾지 못했습니다.")
    return float(values[-1])


def count_yolo_params() -> int:
    model = YOLO(str(YOLO_WEIGHTS))
    return int(sum(param.numel() for param in model.model.parameters()))


def load_literature_model() -> tuple[LiteratureCNNGRUNet, list[str]]:
    ckpt = torch.load(LITERATURE_CKPT, map_location="cpu")
    cfg = ckpt["model_config"]
    model = LiteratureCNNGRUNet(
        input_dim=int(cfg["input_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        num_contexts=int(cfg["num_contexts"]),
        dropout=float(cfg["dropout"]),
        cnn_channels=int(cfg["cnn_channels"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    labels_map = ckpt["labels"]
    labels = [""] * len(labels_map)
    for name, index in labels_map.items():
        labels[int(index)] = str(name)
    return model, labels


def count_temporal_params(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def build_spatial() -> YOLOSpatialEncoder:
    cfg = SpatialConfig(
        model_name="yolo_nano",
        feature_dim=FEATURE_DIM,
        weights_path=str(YOLO_WEIGHTS),
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        conf_threshold=CONF_THRESHOLD,
        max_det=MAX_DET,
        input_size=INPUT_SIZE,
        fallback_to_mock=False,
    )
    return YOLOSpatialEncoder(cfg)


def predict_basic_rule(feature_vec: list[float]) -> str:
    vec = feature_vec + [0.0] * max(0, FEATURE_DIM - len(feature_vec))
    (
        det_norm,
        _mean_conf,
        _max_conf,
        mean_area,
        _area_var,
        vehicle_ratio,
        _person_ratio,
        _bike_ratio,
        _vehicle_conf,
        _person_conf,
        _bike_conf,
        roi_risk,
        motion_delta,
        center,
        looming,
        occlusion,
    ) = vec[:FEATURE_DIM]
    pred_label = "normal_drive"
    if det_norm >= 0.28 and vehicle_ratio >= 0.20 and mean_area < 0.01:
        pred_label = "dense_traffic"
    if (
        vehicle_ratio >= 0.05
        and roi_risk >= 0.34
        and center >= 0.76
        and looming >= 0.02
        and (occlusion >= 0.60 or motion_delta <= 0.01)
    ):
        pred_label = "hard_brake_risk"
    elif vehicle_ratio >= 0.05 and roi_risk >= 0.34 and center >= 0.74 and looming >= 0.008:
        pred_label = "brake_warning"
    elif vehicle_ratio >= 0.05 and roi_risk >= 0.28 and center >= 0.70:
        pred_label = "front_vehicle_follow"
    elif occlusion >= 0.60 and looming < 0.006 and roi_risk >= 0.28:
        pred_label = "post_brake_recovery"
    return pred_label


def load_video_frames(video_path: Path) -> tuple[list[tuple[int, Any, float]], int]:
    cv2 = import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"비디오를 열 수 없습니다: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 15.0
    frames: list[tuple[int, Any, float]] = []
    frame_id = 0
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frames.append((frame_id, frame_bgr.copy(), fps))
        frame_id += 1
    capture.release()
    return frames, frame_id


def build_frame_ground_truth(annotation_path: Path, frame_count: int) -> list[str]:
    gt = ["normal_drive"] * frame_count
    for row in read_jsonl(annotation_path):
        label = str(row.get("context_tag", "normal_drive"))
        start = max(0, int(row.get("start_frame", 0)))
        end = min(frame_count - 1, int(row.get("end_frame", start)))
        for frame_idx in range(start, end + 1):
            gt[frame_idx] = label
    return gt


def encode_feature(spatial: YOLOSpatialEncoder, video_path: Path, fid: int, frame_bgr: Any, fps_value: float) -> list[float]:
    cv2 = import_cv2()
    gray = cv2.resize(
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY),
        (8, 4),
        interpolation=cv2.INTER_AREA,
    ).reshape(-1)
    pixels = [round(float(v) / 255.0, 6) for v in gray.tolist()]
    packet = FramePacket(
        frame_id=fid,
        sensor_timestamp=fid / max(1.0, fps_value),
        system_timestamp=time.time(),
        pixels=pixels,
        raw_path=f"{video_path.name}#frame={fid}",
        image=frame_bgr,
    )
    return spatial.encode(packet)


def context_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    return sum(int(gt == pred) for gt, pred in zip(y_true, y_pred, strict=True)) / max(1, len(y_true))


def brake_critical_recall(y_true: list[str], y_pred: list[str]) -> float:
    brake_labels = {"brake_warning", "hard_brake_risk"}
    target_positions = [idx for idx, label in enumerate(y_true) if label in brake_labels]
    if not target_positions:
        return 0.0
    hit = sum(1 for idx in target_positions if y_pred[idx] in brake_labels)
    return hit / len(target_positions)


def summarize_counts(preds: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in preds:
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def evaluate_basic(video_path: Path, frames: list[tuple[int, Any, float]], gt_labels: list[str]) -> dict[str, Any]:
    spatial = build_spatial()
    preds: list[str] = []
    start = time.perf_counter()
    for fid, frame_bgr, fps_value in frames:
        vec = encode_feature(spatial, video_path, fid, frame_bgr, fps_value)
        preds.append(predict_basic_rule(vec))
    elapsed = time.perf_counter() - start
    return {
        "model": "YOLO+rule",
        "context_match_prob": round(context_accuracy(gt_labels, preds), 6),
        "brake_critical_recall": round(brake_critical_recall(gt_labels, preds), 6),
        "inference_ms_per_frame": round((elapsed / max(1, len(frames))) * 1000.0, 4),
        "pred_counts": summarize_counts(preds),
    }


def evaluate_literature(video_path: Path, frames: list[tuple[int, Any, float]], gt_labels: list[str]) -> dict[str, Any]:
    spatial = build_spatial()
    model, labels = load_literature_model()
    buffer: deque[list[float]] = deque(maxlen=WINDOW_SIZE)
    preds: list[str] = []
    start = time.perf_counter()
    for fid, frame_bgr, fps_value in frames:
        vec = encode_feature(spatial, video_path, fid, frame_bgr, fps_value)
        buffer.append(vec)
        while len(buffer) < WINDOW_SIZE:
            buffer.appendleft([0.0] * len(vec))
        x = torch.tensor([list(buffer)], dtype=torch.float32)
        with torch.no_grad():
            output = model(x)
            pred_idx = int(torch.argmax(output["context_logits"], dim=-1)[0].item())
        preds.append(labels[pred_idx])
    elapsed = time.perf_counter() - start
    return {
        "model": "literature_cnn_temporal",
        "context_match_prob": round(context_accuracy(gt_labels, preds), 6),
        "brake_critical_recall": round(brake_critical_recall(gt_labels, preds), 6),
        "inference_ms_per_frame": round((elapsed / max(1, len(frames))) * 1000.0, 4),
        "pred_counts": summarize_counts(preds),
    }


def evaluate_final_hybrid(
    video_path: Path,
    proposed_ckpt: Path,
    frames: list[tuple[int, Any, float]],
    gt_labels: list[str],
) -> dict[str, Any]:
    spatial = build_spatial()
    model, labels, window_size = load_model(proposed_ckpt)
    params = load_hybrid_params(ROOT / "context_model_benchmark_manifest.json")
    buffer: deque[list[float]] = deque(maxlen=window_size)
    preds: list[str] = []
    start = time.perf_counter()
    for fid, frame_bgr, fps_value in frames:
        vec = encode_feature(spatial, video_path, fid, frame_bgr, fps_value)
        buffer.append(vec)
        while len(buffer) < window_size:
            buffer.appendleft([0.0] * len(vec))
        result = predict_sequence(model, labels, list(buffer), params)
        preds.append(str(result["final_label"]))
    elapsed = time.perf_counter() - start
    return {
        "model": "final_ui_hybrid",
        "context_match_prob": round(context_accuracy(gt_labels, preds), 6),
        "brake_critical_recall": round(brake_critical_recall(gt_labels, preds), 6),
        "inference_ms_per_frame": round((elapsed / max(1, len(frames))) * 1000.0, 4),
        "pred_counts": summarize_counts(preds),
    }


def save_markdown(out_md: Path, video_path: Path, annotation_path: Path, rows: list[dict[str, Any]], frame_count: int) -> None:
    headers = [
        "model",
        "context_match_prob",
        "object_detection_map50",
        "inference_ms_per_frame",
        "parameter_count",
        "brake_critical_recall",
    ]
    lines = [
        f"# {video_path.name} 3모델 비교",
        "",
        f"- 대상 영상: `{video_path}`",
        f"- annotation: `{annotation_path}`",
        f"- 프레임 수: `{frame_count}`",
        f"- detector threshold: `{CONF_THRESHOLD}`",
        "- 주의: 객체인식 정확도는 bbox GT가 없어 공통 detector의 validation `mAP50`을 사용했다.",
        "- 주의: 문맥 GT는 `draft` annotation 기준이다.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key in headers) + " |")
    lines += [
        "",
        "## 해석",
        "- `context_match_prob`는 annotation과 프레임 단위로 직접 비교한 값이다.",
        "- `brake_critical_recall`은 `brake_warning`, `hard_brake_risk` 구간을 얼마나 놓치지 않고 잡았는지 의미한다.",
        "- `object_detection_map50`은 세 모델이 같은 YOLO detector를 공유하므로 동일하다.",
        "- 시간은 같은 영상에서 detector를 포함한 end-to-end 평균 시간(ms/frame)이다.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    annotation_path = Path(args.annotation)
    proposed_ckpt = Path(args.proposed_ckpt)
    prefix = str(args.prefix)

    out_csv = ROOT / f"{prefix}.csv"
    out_md = ROOT / f"{prefix}.md"
    out_png = ROOT / f"{prefix}.png"
    out_manifest = ROOT / f"{prefix}_manifest.json"

    frames, frame_count = load_video_frames(video_path)
    gt_labels = build_frame_ground_truth(annotation_path, frame_count)
    detector_map50 = read_detector_map50()
    yolo_params = count_yolo_params()

    literature_model, _ = load_literature_model()
    literature_params = count_temporal_params(literature_model)
    proposed_model, _, _ = load_model(proposed_ckpt)
    proposed_params = count_temporal_params(proposed_model)

    basic_row = evaluate_basic(video_path, frames, gt_labels)
    literature_row = evaluate_literature(video_path, frames, gt_labels)
    final_row = evaluate_final_hybrid(video_path, proposed_ckpt, frames, gt_labels)

    rows = [
        {
            **basic_row,
            "object_detection_map50": round(detector_map50, 6),
            "parameter_count": yolo_params,
        },
        {
            **literature_row,
            "object_detection_map50": round(detector_map50, 6),
            "parameter_count": yolo_params + literature_params,
        },
        {
            **final_row,
            "object_detection_map50": round(detector_map50, 6),
            "parameter_count": yolo_params + proposed_params,
        },
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "context_match_prob",
                "object_detection_map50",
                "inference_ms_per_frame",
                "parameter_count",
                "brake_critical_recall",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    save_markdown(out_md, video_path, annotation_path, rows, frame_count)

    labels = [row["model"] for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes[0, 0].bar(labels, [row["context_match_prob"] for row in rows], color=["#6baed6", "#9ecae1", "#3182bd"])
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].set_title("Context Match Probability")

    axes[0, 1].bar(labels, [row["object_detection_map50"] for row in rows], color=["#74c476", "#74c476", "#74c476"])
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 1].set_title("Object Detection mAP50")

    axes[0, 2].bar(labels, [row["brake_critical_recall"] for row in rows], color=["#fd8d3c", "#fdae6b", "#e6550d"])
    axes[0, 2].set_ylim(0, 1.0)
    axes[0, 2].set_title("Brake-Critical Recall")

    axes[1, 0].bar(labels, [row["inference_ms_per_frame"] for row in rows], color=["#756bb1", "#9e9ac8", "#54278f"])
    axes[1, 0].set_title("Inference Time (ms/frame)")

    axes[1, 1].bar(labels, [row["parameter_count"] / 1_000_000 for row in rows], color=["#969696", "#bdbdbd", "#636363"])
    axes[1, 1].set_title("Parameter Count (M)")

    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.02,
        0.95,
        "\n".join(
            [
                "조건",
                f"- video: {video_path.name}",
                f"- frames: {frame_count}",
                f"- detector conf: {CONF_THRESHOLD}",
                "- detector mAP50는 공통값",
                "- context/brake는 annotation 기준",
                "- annotation 상태: draft",
            ]
        ),
        va="top",
        fontsize=10,
    )
    for ax in axes.ravel():
        if ax.has_data():
            ax.tick_params(axis="x", rotation=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "video_path": str(video_path),
        "annotation_path": str(annotation_path),
        "frame_count": frame_count,
        "detector_conf_threshold": CONF_THRESHOLD,
        "shared_detector_weights": str(YOLO_WEIGHTS),
        "shared_detector_map50": detector_map50,
        "proposed_ckpt": str(proposed_ckpt),
        "yolo_params": yolo_params,
        "literature_temporal_params": literature_params,
        "proposed_temporal_params": proposed_params,
        "rows": rows,
        "prediction_counts": {row["model"]: row["pred_counts"] for row in rows},
        "notes": {
            "context_source": f"{annotation_path.name} annotation",
            "object_metric_source": "YOLO 3클래스 validation mAP50",
            "timing_source": f"{video_path.name} end-to-end 재측정",
        },
    }
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
