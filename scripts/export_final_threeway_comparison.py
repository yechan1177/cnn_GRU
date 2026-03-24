from __future__ import annotations

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
BENCHMARK_CSV = ROOT / "experiments/exp_017_hybrid_rule_gate_benchmark/runs/context_model_benchmark_hybrid_v2/context_model_benchmark.csv"
LITERATURE_CKPT = ROOT / "experiments/exp_017_hybrid_rule_gate_benchmark/runs/context_model_benchmark_hybrid_v2/literature_single_channel_cnn_gru_best.pt"
PROPOSED_CKPT = ROOT / "experiments/exp_015_temporal_braking_expanded_feat16/runs/mcnn_gru_braking_expanded_feat16_t045_e40_h96_c24/checkpoints/best.pt"
HYBRID_MANIFEST = ROOT / "context_model_benchmark_manifest.json"
VIDEO_PATH = ROOT / "data/raw/videos/stopcar.mp4"

OUT_CSV = ROOT / "final_threeway_comparison.csv"
OUT_MD = ROOT / "final_threeway_comparison.md"
OUT_PNG = ROOT / "final_threeway_comparison.png"
OUT_MANIFEST = ROOT / "final_threeway_comparison_manifest.json"


def import_cv2():
    import cv2

    return cv2


def read_context_metrics() -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with BENCHMARK_CSV.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows[str(row["model"])] = {
                "context_acc": float(row["context_acc"]),
                "brake_critical_recall": float(row["brake_critical_recall"]),
            }
    return rows


def read_detector_map50() -> float:
    text = YOLO_RESULT_SUMMARY.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"mAP50\(B\): `([0-9.]+)`", text)
    if not match:
        raise ValueError("YOLO result_summary.md에서 mAP50 값을 찾지 못했습니다.")
    values = re.findall(r"mAP50\(B\): `([0-9.]+)`", text)
    return float(values[-1])


def count_yolo_params() -> int:
    model = YOLO(str(YOLO_WEIGHTS))
    return int(sum(param.numel() for param in model.model.parameters()))


def load_literature_model() -> LiteratureCNNGRUNet:
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
    return model


def count_temporal_params(model: torch.nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def build_spatial() -> YOLOSpatialEncoder:
    cfg = SpatialConfig(
        model_name="yolo_nano",
        feature_dim=16,
        weights_path=str(YOLO_WEIGHTS),
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        conf_threshold=0.45,
        max_det=30,
        input_size=640,
        fallback_to_mock=False,
    )
    return YOLOSpatialEncoder(cfg)


def predict_basic_rule(feature_vec: list[float]) -> str:
    vec = feature_vec + [0.0] * max(0, 16 - len(feature_vec))
    det_norm, _, _, mean_area, _, vehicle_ratio, _, _, _, _, _, roi_risk, motion_delta, center, looming, occlusion = vec[:16]
    pred_label = "normal_drive"
    if det_norm >= 0.28 and vehicle_ratio >= 0.20 and mean_area < 0.01:
        pred_label = "dense_traffic"
    if vehicle_ratio >= 0.05 and roi_risk >= 0.34 and center >= 0.76 and looming >= 0.02 and (occlusion >= 0.60 or motion_delta <= 0.01):
        pred_label = "hard_brake_risk"
    elif vehicle_ratio >= 0.05 and roi_risk >= 0.34 and center >= 0.74 and looming >= 0.008:
        pred_label = "brake_warning"
    elif vehicle_ratio >= 0.05 and roi_risk >= 0.28 and center >= 0.70:
        pred_label = "front_vehicle_follow"
    elif occlusion >= 0.60 and looming < 0.006 and roi_risk >= 0.28:
        pred_label = "post_brake_recovery"
    return pred_label


def measure_ms_per_frame() -> dict[str, float]:
    cv2 = import_cv2()
    spatial = build_spatial()
    proposed_model, proposed_labels, window_size = load_model(PROPOSED_CKPT)
    hybrid_params = load_hybrid_params(HYBRID_MANIFEST)
    literature_model = load_literature_model()

    capture = cv2.VideoCapture(str(VIDEO_PATH))
    if not capture.isOpened():
        raise RuntimeError(f"비디오를 열 수 없습니다: {VIDEO_PATH}")

    frames: list[Any] = []
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_id = 0
    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        frames.append((frame_id, frame_bgr.copy(), fps))
        frame_id += 1
    capture.release()

    def encode_feature(fid: int, frame_bgr: Any, fps_value: float) -> list[float]:
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
            raw_path=f"{VIDEO_PATH.name}#frame={fid}",
            image=frame_bgr,
        )
        return spatial.encode(packet)

    def time_basic() -> float:
        start = time.perf_counter()
        for fid, frame_bgr, fps_value in frames:
            vec = encode_feature(fid, frame_bgr, fps_value)
            _ = predict_basic_rule(vec)
        elapsed = time.perf_counter() - start
        return (elapsed / max(1, len(frames))) * 1000.0

    def time_literature() -> float:
        buffer: deque[list[float]] = deque(maxlen=8)
        start = time.perf_counter()
        for fid, frame_bgr, fps_value in frames:
            vec = encode_feature(fid, frame_bgr, fps_value)
            buffer.append(vec)
            while len(buffer) < 8:
                buffer.appendleft([0.0] * len(vec))
            x = torch.tensor([list(buffer)], dtype=torch.float32)
            with torch.no_grad():
                _ = literature_model(x)
        elapsed = time.perf_counter() - start
        return (elapsed / max(1, len(frames))) * 1000.0

    def time_hybrid() -> float:
        buffer: deque[list[float]] = deque(maxlen=window_size)
        start = time.perf_counter()
        for fid, frame_bgr, fps_value in frames:
            vec = encode_feature(fid, frame_bgr, fps_value)
            buffer.append(vec)
            while len(buffer) < window_size:
                buffer.appendleft([0.0] * len(vec))
            _ = predict_sequence(proposed_model, proposed_labels, list(buffer), hybrid_params)
        elapsed = time.perf_counter() - start
        return (elapsed / max(1, len(frames))) * 1000.0

    return {
        "basic_yolo_rule_ms": round(time_basic(), 4),
        "literature_cnn_temporal_ms": round(time_literature(), 4),
        "final_hybrid_ui_ms": round(time_hybrid(), 4),
        "frame_count": len(frames),
    }


def save_markdown(rows: list[dict[str, Any]]) -> None:
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key in headers) + " |")
    lines += [
        "",
        "## 해석",
        "- 세 모델은 동일 YOLO detector를 공유하므로 `객체인식 정확도(mAP50)`는 같다.",
        "- 최종 UI 하이브리드 모델은 `brake_critical_recall`이 가장 높다.",
        "- 문맥 일치 확률(`context_acc`)은 최종 UI 하이브리드보다 순수 제안모델이 더 높지만, 이번 표의 비교군에는 순수 제안모델을 포함하지 않았다.",
        "- 추론 시간 차이는 detector가 대부분을 차지해서 temporal 구조 차이보다 detector 비용 영향이 크다.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    context_metrics = read_context_metrics()
    detector_map50 = read_detector_map50()
    yolo_params = count_yolo_params()

    literature_model = load_literature_model()
    literature_params = count_temporal_params(literature_model)
    proposed_model, _, _ = load_model(PROPOSED_CKPT)
    proposed_params = count_temporal_params(proposed_model)

    timing = measure_ms_per_frame()

    rows = [
        {
            "model": "YOLO+rule",
            "context_match_prob": round(context_metrics["basic_yolo_rule"]["context_acc"], 6),
            "object_detection_map50": round(detector_map50, 6),
            "inference_ms_per_frame": timing["basic_yolo_rule_ms"],
            "parameter_count": yolo_params,
            "brake_critical_recall": round(context_metrics["basic_yolo_rule"]["brake_critical_recall"], 6),
        },
        {
            "model": "literature_cnn_temporal",
            "context_match_prob": round(context_metrics["single_channel_cnn_gru"]["context_acc"], 6),
            "object_detection_map50": round(detector_map50, 6),
            "inference_ms_per_frame": timing["literature_cnn_temporal_ms"],
            "parameter_count": yolo_params + literature_params,
            "brake_critical_recall": round(context_metrics["single_channel_cnn_gru"]["brake_critical_recall"], 6),
        },
        {
            "model": "final_ui_hybrid",
            "context_match_prob": round(context_metrics["proposed_hybrid_rule_gate"]["context_acc"], 6),
            "object_detection_map50": round(detector_map50, 6),
            "inference_ms_per_frame": timing["final_hybrid_ui_ms"],
            "parameter_count": yolo_params + proposed_params,
            "brake_critical_recall": round(context_metrics["proposed_hybrid_rule_gate"]["brake_critical_recall"], 6),
        },
    ]

    with OUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_markdown(rows)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = [row["model"] for row in rows]

    axes[0, 0].bar(labels, [row["context_match_prob"] for row in rows], color=["#6baed6", "#9ecae1", "#3182bd"])
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].set_title("Context Match Probability")

    axes[0, 1].bar(labels, [row["object_detection_map50"] for row in rows], color=["#74c476", "#74c476", "#74c476"])
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 1].set_title("Object Detection mAP50")

    axes[1, 0].bar(labels, [row["inference_ms_per_frame"] for row in rows], color=["#fd8d3c", "#fdae6b", "#e6550d"])
    axes[1, 0].set_title("Inference Time (ms/frame)")

    axes[1, 1].bar(labels, [row["parameter_count"] / 1_000_000 for row in rows], color=["#756bb1", "#9e9ac8", "#54278f"])
    axes[1, 1].set_title("Parameter Count (M)")

    for ax in axes.ravel():
        ax.tick_params(axis="x", rotation=10)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=160)
    plt.close(fig)

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "video_path": str(VIDEO_PATH),
        "shared_detector_weights": str(YOLO_WEIGHTS),
        "shared_detector_map50": detector_map50,
        "yolo_params": yolo_params,
        "literature_temporal_params": literature_params,
        "proposed_temporal_params": proposed_params,
        "timing": timing,
        "rows": rows,
        "notes": {
            "shared_detector": "세 비교군은 동일 YOLOv8n 3클래스 detector를 공유한다.",
            "context_source": "문맥 지표는 exp_017 benchmark 결과를 사용했다.",
        },
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
