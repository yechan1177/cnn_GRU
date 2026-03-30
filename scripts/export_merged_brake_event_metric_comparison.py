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

MERGED_CLASS_CSV = ROOT_DIR / "merged_brake_class_count_comparison.csv"
MERGED_CLASS_MD = ROOT_DIR / "merged_brake_class_count_comparison.md"
MERGED_CLASS_PNG = ROOT_DIR / "merged_brake_class_count_comparison.png"

METRIC_CSV = ROOT_DIR / "paper_metric_comparison_with_event_score.csv"
METRIC_MD = ROOT_DIR / "paper_metric_comparison_with_event_score.md"
METRIC_PNG = ROOT_DIR / "paper_metric_comparison_with_event_score.png"

ACTUAL_EVENT_FRAMES = 30

MODEL_LABELS = {
    "yolo_rule": "YOLO+rule",
    "cnn_gru_only": "CNN-GRU only",
    "cnn_gru_plus_rule": "CNN-GRU+rule",
}

MODEL_METRIC_KEYS = {
    "yolo_rule": ("A_det", "T_inf_rule_ms", "N_param_rule"),
    "cnn_gru_only": ("A_det", "T_inf_temporal_ms", "N_param_temporal"),
    "cnn_gru_plus_rule": ("A_det", "T_inf_hybrid_ms", "N_param_hybrid"),
}

CLASS_ORDER_MERGED = [
    "normal_drive",
    "front_vehicle_follow",
    "brake_warning",
    "post_brake_recovery",
    "dense_traffic",
]

MODEL_COLORS = {
    "yolo_rule": "#1f77b4",
    "cnn_gru_only": "#ff7f0e",
    "cnn_gru_plus_rule": "#2ca02c",
}


def load_summary() -> dict:
    with SUMMARY_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_merged_class_rows(summary: dict) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for class_name in CLASS_ORDER_MERGED:
        row: dict[str, int | str] = {"class_name": class_name}
        for model_key in MODEL_LABELS:
            counts = summary["label_counts"].get(model_key, {})
            if class_name == "brake_warning":
                value = int(counts.get("brake_warning", 0)) + int(counts.get("hard_brake_risk", 0))
            else:
                value = int(counts.get(class_name, 0))
            row[model_key] = value
        rows.append(row)
    return rows


def event_score(predicted_event_frames: int, actual_event_frames: int) -> float:
    return max(0.0, 1.0 - abs(predicted_event_frames - actual_event_frames) / actual_event_frames)


def build_metric_rows(summary: dict) -> list[dict[str, object]]:
    metrics = summary["metrics"]
    rows: list[dict[str, object]] = []
    for model_key, model_name in MODEL_LABELS.items():
        counts = summary["label_counts"].get(model_key, {})
        predicted_event_frames = int(counts.get("brake_warning", 0)) + int(counts.get("hard_brake_risk", 0))
        a_det_key, t_inf_key, n_param_key = MODEL_METRIC_KEYS[model_key]
        rows.append(
            {
                "model_name": model_name,
                "A_det": float(metrics[a_det_key]),
                "T_inf": float(metrics[t_inf_key]),
                "N_param": int(metrics[n_param_key]),
                "N_pred_evt": predicted_event_frames,
                "N_gt_evt": ACTUAL_EVENT_FRAMES,
                "S_evt": round(event_score(predicted_event_frames, ACTUAL_EVENT_FRAMES), 6),
            }
        )
    return rows


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_markdown(path: Path, title: str, rows: list[dict[str, object]]) -> None:
    lines = [f"# {title}", ""]
    if "class_name" in rows[0]:
        lines.extend(
            [
                "| 클래스 | YOLO+rule | CNN-GRU only | CNN-GRU+rule |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['class_name']} | {row['yolo_rule']} | {row['cnn_gru_only']} | {row['cnn_gru_plus_rule']} |"
            )
    else:
        lines.extend(
            [
                f"- 실제 이벤트 프레임 수 기준: `{ACTUAL_EVENT_FRAMES}`",
                "- `S_evt = max(0, 1 - |N_pred_evt - N_gt_evt| / N_gt_evt)`",
                "",
                "| 모델 | A_det | T_inf | N_param | N_pred_evt | N_gt_evt | S_evt |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['model_name']} | {row['A_det']:.5f} | {row['T_inf']:.4f} | {row['N_param']} | {row['N_pred_evt']} | {row['N_gt_evt']} | {row['S_evt']:.6f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_merged_class_counts(rows: list[dict[str, object]]) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(13, 6))
    x = list(range(len(CLASS_ORDER_MERGED)))
    width = 0.25
    offsets = {
        "yolo_rule": -width,
        "cnn_gru_only": 0.0,
        "cnn_gru_plus_rule": width,
    }

    for model_key, label in MODEL_LABELS.items():
        values = [int(row[model_key]) for row in rows]
        positions = [idx + offsets[model_key] for idx in x]
        bars = ax.bar(positions, values, width=width, label=label, color=MODEL_COLORS[model_key], alpha=0.9)
        for bar in bars:
            height = int(bar.get_height())
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, str(height), ha="center", va="bottom", fontsize=9)

    ax.set_title("브레이크 이벤트 통합 클래스 분포 비교")
    ax.set_xlabel("클래스")
    ax.set_ylabel("프레임 수")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER_MERGED, rotation=15, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(MERGED_CLASS_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metric_rows(rows: list[dict[str, object]]) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    model_names = [str(row["model_name"]) for row in rows]
    colors = [MODEL_COLORS[key] for key in MODEL_LABELS]

    metric_specs = [
        ("A_det", "객체인식 정확도 A_det"),
        ("T_inf", "평균 추론 시간 T_inf (ms/frame)"),
        ("N_param", "전체 파라미터 수 N_param"),
        ("S_evt", "이벤트 추출 적합도 S_evt"),
    ]

    for ax, (metric_key, title) in zip(axes, metric_specs):
        values = [float(row[metric_key]) for row in rows]
        bars = ax.bar(model_names, values, color=colors, alpha=0.9)
        for bar, value in zip(bars, values):
            if metric_key in {"A_det", "S_evt"}:
                label = f"{value:.6f}"
            elif metric_key == "T_inf":
                label = f"{value:.4f}"
            else:
                label = f"{int(value)}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("논문용 핵심 지표 비교")
    fig.tight_layout()
    fig.savefig(METRIC_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"요약 파일이 없습니다: {SUMMARY_PATH}")

    summary = load_summary()
    merged_rows = build_merged_class_rows(summary)
    metric_rows = build_metric_rows(summary)

    save_csv(MERGED_CLASS_CSV, merged_rows)
    save_markdown(MERGED_CLASS_MD, "브레이크 이벤트 통합 클래스 분포 비교", merged_rows)
    plot_merged_class_counts(merged_rows)

    save_csv(METRIC_CSV, metric_rows)
    save_markdown(METRIC_MD, "논문용 핵심 지표 비교", metric_rows)
    plot_metric_rows(metric_rows)

    print(f"[saved] {MERGED_CLASS_CSV}")
    print(f"[saved] {MERGED_CLASS_MD}")
    print(f"[saved] {MERGED_CLASS_PNG}")
    print(f"[saved] {METRIC_CSV}")
    print(f"[saved] {METRIC_MD}")
    print(f"[saved] {METRIC_PNG}")


if __name__ == "__main__":
    main()
