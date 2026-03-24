from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def add_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    subtitle: str,
    color: str,
) -> None:
    """박스를 그리고 제목/설명을 배치한다."""
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.8,
        edgecolor="#1f2937",
        facecolor=color,
    )
    ax.add_patch(patch)
    cx = xy[0] + width / 2
    cy = xy[1] + height / 2
    ax.text(
        cx,
        cy + 0.03,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        cx,
        cy - 0.03,
        subtitle,
        ha="center",
        va="center",
        fontsize=9,
        color="#1f2937",
    )


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    """단계 간 흐름 화살표를 추가한다."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=2.0, color="#374151"),
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_path = root / "project_architecture.png"

    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    ax.text(
        0.5,
        0.96,
        "Vision-Context On-Device Pipeline Architecture",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0.5,
        0.925,
        "proposal_v2 aligned | RTX3080Ti(Train) / Jetson Orin Nano 8GB(Deploy)",
        ha="center",
        va="center",
        fontsize=11,
        color="#334155",
    )

    y = 0.60
    w = 0.135
    h = 0.18
    xs = [0.03, 0.185, 0.34, 0.495, 0.65, 0.805]

    add_box(
        ax,
        (xs[0], y),
        w,
        h,
        "1) Input Stream",
        "camera/video frames",
        "#dbeafe",
    )
    add_box(
        ax,
        (xs[1], y),
        w,
        h,
        "2) Spatial Encoder",
        "YOLO-nano / MobileNet",
        "#dcfce7",
    )
    add_box(
        ax,
        (xs[2], y),
        w,
        h,
        "3) Feature Packing",
        "vector + metadata",
        "#fef3c7",
    )
    add_box(
        ax,
        (xs[3], y),
        w,
        h,
        "4) Temporal Context",
        "GRU / TSM+GRU",
        "#fae8ff",
    )
    add_box(
        ax,
        (xs[4], y),
        w,
        h,
        "5) Scoring + Curation",
        "context/boundary/uncertainty",
        "#fee2e2",
    )
    add_box(
        ax,
        (xs[5], y),
        w,
        h,
        "6) Aligned Export",
        "frame/event JSONL",
        "#e0f2fe",
    )

    for i in range(len(xs) - 1):
        add_arrow(
            ax,
            (xs[i] + w, y + h / 2),
            (xs[i + 1], y + h / 2),
        )

    add_box(
        ax,
        (0.06, 0.26),
        0.40,
        0.20,
        "Train/Experiment (RTX 3080 Ti)",
        "YOLO train, Temporal GRU train, threshold tuning",
        "#ede9fe",
    )
    add_box(
        ax,
        (0.54, 0.26),
        0.40,
        0.20,
        "Deploy/Inference (Jetson Orin Nano 8GB)",
        "lightweight runtime, event-driven storage",
        "#cffafe",
    )
    add_arrow(ax, (0.46, 0.36), (0.54, 0.36))

    add_box(
        ax,
        (0.03, 0.05),
        0.30,
        0.14,
        "Storage Policy",
        "high: clip | mid: keyframe | low: discard/summary",
        "#fde68a",
    )
    add_box(
        ax,
        (0.35, 0.05),
        0.30,
        0.14,
        "Schema",
        "sensor_ts + system_ts, frame/event separation",
        "#bae6fd",
    )
    add_box(
        ax,
        (0.67, 0.05),
        0.30,
        0.14,
        "Research Artifacts",
        "figures/tables/logs/proposal-ready outputs",
        "#bbf7d0",
    )

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
