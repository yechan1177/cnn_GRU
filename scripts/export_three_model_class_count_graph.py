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
OUTPUT_CSV = ROOT_DIR / "three_model_class_count_comparison.csv"
OUTPUT_MD = ROOT_DIR / "three_model_class_count_comparison.md"
OUTPUT_PNG = ROOT_DIR / "three_model_class_count_comparison.png"

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


def load_counts(summary_path: Path) -> list[dict[str, int | str]]:
    with summary_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    label_counts: dict[str, dict[str, int]] = payload["label_counts"]
    rows: list[dict[str, int | str]] = []
    for class_name in CLASS_ORDER:
        rows.append(
            {
                "class_name": class_name,
                "yolo_rule": int(label_counts.get("yolo_rule", {}).get(class_name, 0)),
                "cnn_gru_only": int(label_counts.get("cnn_gru_only", {}).get(class_name, 0)),
                "cnn_gru_plus_rule": int(label_counts.get("cnn_gru_plus_rule", {}).get(class_name, 0)),
            }
        )
    return rows


def save_csv(rows: list[dict[str, int | str]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["class_name", "yolo_rule", "cnn_gru_only", "cnn_gru_plus_rule"],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_markdown(rows: list[dict[str, int | str]], path: Path) -> None:
    lines = [
        "# 세 모델 클래스별 개수 비교",
        "",
        "기준 파일:",
        f"- `{SUMMARY_PATH}`",
        "",
        "| 클래스 | YOLO+rule | CNN-GRU only | CNN-GRU+rule |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['class_name']} | {row['yolo_rule']} | {row['cnn_gru_only']} | {row['cnn_gru_plus_rule']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_counts(rows: list[dict[str, int | str]], path: Path) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(14, 7))

    x = list(range(len(CLASS_ORDER)))
    width = 0.25
    offsets = {
        "yolo_rule": -width,
        "cnn_gru_only": 0.0,
        "cnn_gru_plus_rule": width,
    }

    for model_key, label in MODEL_LABELS.items():
        values = [int(row[model_key]) for row in rows]
        positions = [v + offsets[model_key] for v in x]
        bars = ax.bar(
            positions,
            values,
            width=width,
            label=label,
            color=MODEL_COLORS[model_key],
            alpha=0.9,
        )
        for bar in bars:
            height = int(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                str(height),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_title("세 모델 클래스별 예측 개수 비교 (people_braking)")
    ax.set_xlabel("클래스")
    ax.set_ylabel("프레임 수")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER, rotation=20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"요약 파일이 없습니다: {SUMMARY_PATH}")

    rows = load_counts(SUMMARY_PATH)
    save_csv(rows, OUTPUT_CSV)
    save_markdown(rows, OUTPUT_MD)
    plot_counts(rows, OUTPUT_PNG)

    print(f"[saved] csv: {OUTPUT_CSV}")
    print(f"[saved] md : {OUTPUT_MD}")
    print(f"[saved] png: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
