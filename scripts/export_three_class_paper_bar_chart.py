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

OUTPUT_CSV = ROOT_DIR / "three_class_paper_bar_chart.csv"
OUTPUT_MD = ROOT_DIR / "three_class_paper_bar_chart.md"
OUTPUT_PNG = ROOT_DIR / "three_class_paper_bar_chart.png"

CLASS_ITEMS = [
    ("normal_drive", "일반 주행\n(normal_drive)"),
    ("front_vehicle_follow", "차량 추종\n(vehicle_follow)"),
    ("brake_warning", "브레이크 경고\n(brake_warning)"),
]

MODEL_LABELS = {
    "yolo_rule": "YOLO+rule",
    "cnn_gru_only": "CNN-GRU only",
    "cnn_gru_plus_rule": "CNN-GRU+rule",
}

MODEL_COLORS = {
    "yolo_rule": "#4C78A8",
    "cnn_gru_only": "#F58518",
    "cnn_gru_plus_rule": "#54A24B",
}


def load_summary() -> dict:
    with SUMMARY_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def build_rows(summary: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    label_counts = summary["label_counts"]
    for class_key, class_label in CLASS_ITEMS:
        rows.append(
            {
                "class_key": class_key,
                "class_label": class_label,
                "yolo_rule": int(label_counts.get("yolo_rule", {}).get(class_key, 0)),
                "cnn_gru_only": int(label_counts.get("cnn_gru_only", {}).get(class_key, 0)),
                "cnn_gru_plus_rule": int(label_counts.get("cnn_gru_plus_rule", {}).get(class_key, 0)),
            }
        )
    return rows


def save_csv(rows: list[dict[str, object]]) -> None:
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["class_key", "class_label", "yolo_rule", "cnn_gru_only", "cnn_gru_plus_rule"],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_markdown(rows: list[dict[str, object]]) -> None:
    lines = [
        "# 주요 3개 상황 분포 비교",
        "",
        f"- 기준 파일: `{SUMMARY_PATH}`",
        "",
        "| 상황 | YOLO+rule | CNN-GRU only | CNN-GRU+rule |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['class_label'].replace(chr(10), ' ')} | {row['yolo_rule']} | {row['cnn_gru_only']} | {row['cnn_gru_plus_rule']} |"
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_chart(rows: list[dict[str, object]]) -> None:
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = list(range(len(rows)))
    width = 0.22
    offsets = {
        "yolo_rule": -width,
        "cnn_gru_only": 0.0,
        "cnn_gru_plus_rule": width,
    }

    for model_key, model_label in MODEL_LABELS.items():
        values = [int(row[model_key]) for row in rows]
        positions = [idx + offsets[model_key] for idx in x]
        bars = ax.bar(
            positions,
            values,
            width=width,
            label=model_label,
            color=MODEL_COLORS[model_key],
            edgecolor="#2f2f2f",
            linewidth=0.6,
            alpha=0.95,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f"{value}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#1f2937",
            )

    ax.set_title("주요 3개 상황 분포 비교", fontsize=12, pad=10)
    ax.set_ylabel("프레임 수", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([str(row["class_label"]) for row in rows], fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", linestyle=(0, (3, 3)), linewidth=0.6, alpha=0.45)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.legend(loc="upper right", frameon=True, edgecolor="#cbd5e1", fontsize=9)

    ymax = max(max(int(row["yolo_rule"]), int(row["cnn_gru_only"]), int(row["cnn_gru_plus_rule"])) for row in rows)
    ax.set_ylim(0, ymax * 1.18)

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"요약 파일이 없습니다: {SUMMARY_PATH}")

    summary = load_summary()
    rows = build_rows(summary)
    save_csv(rows)
    save_markdown(rows)
    plot_chart(rows)

    print(f"[saved] csv: {OUTPUT_CSV}")
    print(f"[saved] md : {OUTPUT_MD}")
    print(f"[saved] png: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
