from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[1]
SUMMARY_PATH = (
    ROOT_DIR
    / "artifacts"
    / "comparisons"
    / "people_braking_three_model_compare"
    / "summary.json"
)

OUTPUT_CSV = ROOT_DIR / "manual_hard_brake_range_summary.csv"
OUTPUT_MD = ROOT_DIR / "manual_hard_brake_range_summary.md"
OUTPUT_PNG = ROOT_DIR / "manual_hard_brake_range_summary.png"

TOTAL_FRAMES_DEFAULT = 429
ACTUAL_HARD_MIN = 20
ACTUAL_HARD_MAX = 30
ACTUAL_HARD_MID = (ACTUAL_HARD_MIN + ACTUAL_HARD_MAX) / 2

CLASS_ORDER = [
    "normal_drive",
    "front_vehicle_follow",
    "brake_warning",
    "hard_brake_risk",
    "post_brake_recovery",
    "dense_traffic",
]

MODEL_LABELS = {
    "yolo_rule": "YOLO+rule",
    "cnn_gru_only": "CNN-GRU only",
    "cnn_gru_plus_rule": "CNN-GRU+rule",
}

MODEL_COLORS = {
    "yolo_rule": "#1f77b4",
    "cnn_gru_only": "#ff7f0e",
    "cnn_gru_plus_rule": "#2ca02c",
}

STACK_COLORS = {
    "normal_drive": "#4c78a8",
    "front_vehicle_follow": "#72b7b2",
    "brake_warning": "#f58518",
    "hard_brake_risk": "#e45756",
    "post_brake_recovery": "#54a24b",
    "dense_traffic": "#b279a2",
}


def load_summary() -> dict:
    with SUMMARY_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_rows(summary: dict) -> tuple[int, list[dict[str, object]]]:
    total_frames = int(summary.get("frame_count", TOTAL_FRAMES_DEFAULT))
    label_counts = summary["label_counts"]
    rows: list[dict[str, object]] = []

    for model_key, model_name in MODEL_LABELS.items():
        counts = label_counts.get(model_key, {})
        predicted_hard = int(counts.get("hard_brake_risk", 0))
        predicted_hard_ratio = predicted_hard / total_frames
        frame_error_vs_mid = predicted_hard - ACTUAL_HARD_MID
        ratio_error_vs_mid = predicted_hard_ratio - (ACTUAL_HARD_MID / total_frames)
        rows.append(
            {
                "model_key": model_key,
                "model_name": model_name,
                "total_frames": total_frames,
                "predicted_hard_frames": predicted_hard,
                "predicted_hard_ratio": round(predicted_hard_ratio, 6),
                "actual_hard_min": ACTUAL_HARD_MIN,
                "actual_hard_max": ACTUAL_HARD_MAX,
                "actual_hard_mid": ACTUAL_HARD_MID,
                "actual_hard_ratio_min": round(ACTUAL_HARD_MIN / total_frames, 6),
                "actual_hard_ratio_max": round(ACTUAL_HARD_MAX / total_frames, 6),
                "actual_hard_ratio_mid": round(ACTUAL_HARD_MID / total_frames, 6),
                "frame_error_vs_mid": round(frame_error_vs_mid, 1),
                "ratio_error_vs_mid": round(ratio_error_vs_mid, 6),
                "is_within_actual_range": ACTUAL_HARD_MIN <= predicted_hard <= ACTUAL_HARD_MAX,
            }
        )
    return total_frames, rows


def save_csv(rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0].keys())
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_markdown(total_frames: int, rows: list[dict[str, object]]) -> None:
    lines = [
        "# 수동 하드 브레이크 범위 반영 요약",
        "",
        f"- 기준 비교 파일: `{SUMMARY_PATH}`",
        f"- 전체 프레임 수: `{total_frames}`",
        f"- 사용자 판단 실제 하드 브레이크 범위: `{ACTUAL_HARD_MIN}~{ACTUAL_HARD_MAX}` 프레임",
        f"- 실제 하드 브레이크 중앙값 기준: `{ACTUAL_HARD_MID}` 프레임",
        "",
        "정확한 recall/precision은 프레임 단위 GT가 있어야 계산 가능하므로, 여기서는 하드 브레이크 예측 개수와 총 프레임 대비 비율, 실제 범위 대비 차이만 요약한다.",
        "",
        "| 모델 | 예측 hard_brake 프레임 | 예측 비율 | 실제 비율 범위 | 중앙값 대비 프레임 차이 | 범위 내 포함 여부 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {model_name} | {predicted_hard_frames} | {predicted_hard_ratio:.6f} | {actual_hard_ratio_min:.6f}~{actual_hard_ratio_max:.6f} | {frame_error_vs_mid:+.1f} | {within} |".format(
                model_name=row["model_name"],
                predicted_hard_frames=row["predicted_hard_frames"],
                predicted_hard_ratio=row["predicted_hard_ratio"],
                actual_hard_ratio_min=row["actual_hard_ratio_min"],
                actual_hard_ratio_max=row["actual_hard_ratio_max"],
                frame_error_vs_mid=row["frame_error_vs_mid"],
                within="예" if row["is_within_actual_range"] else "아니오",
            )
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(summary: dict, total_frames: int, rows: list[dict[str, object]]) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: stacked class distribution
    ax0 = axes[0]
    x = list(range(len(MODEL_LABELS)))
    model_keys = list(MODEL_LABELS.keys())
    bottoms = [0] * len(model_keys)

    for class_name in CLASS_ORDER:
        values = [int(summary["label_counts"].get(model_key, {}).get(class_name, 0)) for model_key in model_keys]
        ax0.bar(
            x,
            values,
            bottom=bottoms,
            color=STACK_COLORS[class_name],
            label=class_name,
            alpha=0.9,
        )
        bottoms = [b + v for b, v in zip(bottoms, values)]

    ax0.set_title("모델별 클래스 분포")
    ax0.set_ylabel("프레임 수")
    ax0.set_xticks(x)
    ax0.set_xticklabels([MODEL_LABELS[k] for k in model_keys], rotation=10)
    ax0.legend(fontsize=8, loc="upper right")

    # Right: hard-brake predicted vs actual range
    ax1 = axes[1]
    predicted = [int(row["predicted_hard_frames"]) for row in rows]
    labels = [str(row["model_name"]) for row in rows]
    bars = ax1.bar(labels, predicted, color=[MODEL_COLORS[r["model_key"]] for r in rows], alpha=0.9)
    ax1.axhspan(ACTUAL_HARD_MIN, ACTUAL_HARD_MAX, color="#d62728", alpha=0.15, label="실제 하드 브레이크 범위 (20~30프레임)")
    ax1.axhline(ACTUAL_HARD_MID, color="#d62728", linestyle="--", linewidth=1.5, label="실제 중앙값 25프레임")
    for bar in bars:
        height = int(bar.get_height())
        ax1.text(bar.get_x() + bar.get_width() / 2, height + 0.8, str(height), ha="center", va="bottom")
    ax1.set_title("모델별 hard_brake 예측 프레임 수")
    ax1.set_ylabel("프레임 수")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("people_braking 수동 하드 브레이크 범위 반영 종합 비교", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"요약 파일이 없습니다: {SUMMARY_PATH}")

    summary = load_summary()
    total_frames, rows = build_rows(summary)
    save_csv(rows)
    save_markdown(total_frames, rows)
    plot(summary, total_frames, rows)

    print(f"[saved] csv: {OUTPUT_CSV}")
    print(f"[saved] md : {OUTPUT_MD}")
    print(f"[saved] png: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
