from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PNG = ROOT_DIR / "paper_model_illustration.png"
OUTPUT_SVG = ROOT_DIR / "paper_model_illustration.svg"


def add_stage(ax, x: float, y: float, w: float, h: float, title: str, subtitle: str, color: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.6,
        edgecolor=color,
        facecolor="#ffffff",
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.67, title, ha="center", va="center", fontsize=12, weight="bold", color="#1f2937")
    ax.text(x + w / 2, y + h * 0.33, subtitle, ha="center", va="center", fontsize=9, color="#4b5563")


def add_arrow(ax, x1: float, y1: float, x2: float, y2: float, color: str = "#64748b") -> None:
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.6,
        color=color,
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


def add_input_panel(ax) -> None:
    frame = FancyBboxPatch(
        (0.04, 0.60),
        0.14,
        0.20,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.6,
        edgecolor="#2563eb",
        facecolor="#eff6ff",
    )
    ax.add_patch(frame)
    ax.add_patch(FancyBboxPatch((0.065, 0.645), 0.09, 0.08, boxstyle="round,pad=0.01,rounding_size=0.02", linewidth=1.0, edgecolor="#2563eb", facecolor="#dbeafe"))
    ax.add_patch(Circle((0.085, 0.685), 0.008, color="#ef4444"))
    ax.add_patch(Circle((0.112, 0.685), 0.008, color="#22c55e"))
    ax.add_patch(Circle((0.139, 0.685), 0.008, color="#f59e0b"))
    ax.text(0.11, 0.615, "Input Video", ha="center", va="center", fontsize=11, weight="bold", color="#1e3a8a")
    ax.text(0.11, 0.575, "driving frames", ha="center", va="center", fontsize=8.5, color="#1e40af")


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_input_panel(ax)

    add_stage(ax, 0.23, 0.58, 0.14, 0.22, "YOLO", "person / vehicle / bike", "#0f766e")
    add_stage(ax, 0.42, 0.58, 0.16, 0.22, "16D Feature", "ROI, center, area, ratio", "#7c3aed")
    add_stage(ax, 0.63, 0.58, 0.14, 0.22, "CNN-GRU", "8-frame temporal encoder", "#ea580c")
    add_stage(ax, 0.82, 0.58, 0.14, 0.22, "Rule Gate", "current-frame correction", "#be123c")

    add_arrow(ax, 0.18, 0.69, 0.23, 0.69)
    add_arrow(ax, 0.37, 0.69, 0.42, 0.69)
    add_arrow(ax, 0.58, 0.69, 0.63, 0.69)
    add_arrow(ax, 0.77, 0.69, 0.82, 0.69)

    # lower supporting cues
    support = FancyBboxPatch(
        (0.39, 0.18),
        0.36,
        0.22,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.2,
        edgecolor="#94a3b8",
        facecolor="#f8fafc",
        linestyle="--",
    )
    ax.add_patch(support)
    ax.text(0.57, 0.33, "Temporal Context Cues", ha="center", va="center", fontsize=11, weight="bold", color="#334155")
    ax.text(0.57, 0.25, "normal_drive  |  follow  |  brake_warning  |  recovery", ha="center", va="center", fontsize=8.5, color="#475569")

    add_arrow(ax, 0.50, 0.58, 0.50, 0.40, "#94a3b8")
    add_arrow(ax, 0.70, 0.58, 0.64, 0.40, "#94a3b8")

    # output panel
    out = FancyBboxPatch(
        (0.82, 0.18),
        0.14,
        0.22,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.6,
        edgecolor="#1d4ed8",
        facecolor="#eff6ff",
    )
    ax.add_patch(out)
    ax.text(0.89, 0.31, "Final Output", ha="center", va="center", fontsize=11, weight="bold", color="#1e3a8a")
    ax.text(0.89, 0.24, "context label + event trigger", ha="center", va="center", fontsize=8.5, color="#1e40af")
    add_arrow(ax, 0.89, 0.58, 0.89, 0.40, "#1d4ed8")

    fig.tight_layout(pad=0.4)
    fig.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_SVG, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] png: {OUTPUT_PNG}")
    print(f"[saved] svg: {OUTPUT_SVG}")


if __name__ == "__main__":
    main()
