from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]

ENV_HEADERS = ["구분", "항목", "설정"]
ENV_ROWS = [
    ["학습 환경", "GPU", "RTX 3080 Ti 12GB"],
    ["학습 환경", "메모리", "RAM 32GB"],
    ["학습 환경", "CUDA", "11.8"],
    ["학습 환경", "객체 검출 프레임워크", "Ultralytics YOLOv8"],
    ["학습 환경", "시계열 학습 프레임워크", "PyTorch"],
    ["객체 검출 학습", "모델", "YOLOv8n (3 classes: person, vehicle, bike)"],
    ["객체 검출 학습", "입력 해상도", "640"],
    ["객체 검출 학습", "배치 크기", "16"],
    ["객체 검출 학습", "최적화", "Adam"],
    ["시계열 학습", "입력 길이", "최근 8프레임 시퀀스"],
    ["시계열 학습", "입력 특징", "16차원 의미 기반 feature vector"],
    ["시계열 학습", "모델", "멀티채널 CNN-GRU + Hybrid Rule Gate"],
    ["시계열 학습", "hidden dimension", "96"],
    ["시계열 학습", "CNN channels", "24"],
    ["시계열 학습", "dropout", "0.1"],
    ["배포 타깃", "디바이스", "Jetson Orin Nano 8GB"],
    ["배포 타깃", "구조 원칙", "학습/배포 경로 분리, 온디바이스 추론 지향"],
]

METRIC_HEADERS = ["지표", "의미", "설명"]
METRIC_ROWS = [
    ["Context Match Probability", "문맥 일치 확률", "프레임 단위 예측 맥락이 annotation과 일치한 비율"],
    ["Brake-Critical Recall", "브레이크 민감도", "`brake_warning`, `hard_brake_risk` 구간을 놓치지 않고 잡은 비율"],
    ["Object Detection mAP50", "객체 검출 정확도", "공통 YOLO detector의 validation mAP50"],
    ["Inference Time (ms/frame)", "추론 시간", "같은 영상에서 detector 포함 end-to-end 평균 처리 시간"],
    ["Parameter Count", "파라미터 수", "detector와 temporal 모듈을 포함한 전체 파라미터 수"],
]


def write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(rows)


def write_md(path: Path, title: str, headers: list[str], rows: list[list[str]]) -> None:
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def add_box(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.6,
        edgecolor="#1f2937",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=11, fontweight="bold", color="#111827")
    ax.text(x + w / 2, y + h * 0.35, body, ha="center", va="center", fontsize=9, color="#1f2937")


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=2.0, color="#334155"),
    )


def export_model_diagram(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    ax.text(
        0.5,
        0.95,
        "Proposed On-Device Brake Context Model",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        0.5,
        0.915,
        "YOLOv8n + Semantic Feature Vector + Multichannel CNN-GRU + Hybrid Rule Gate",
        ha="center",
        va="center",
        fontsize=11,
        color="#334155",
    )

    y = 0.60
    w = 0.13
    h = 0.18
    xs = [0.03, 0.18, 0.33, 0.48, 0.63, 0.78]

    add_box(ax, xs[0], y, w, h, "Input Video", "driving video / camera stream", "#dbeafe")
    add_box(ax, xs[1], y, w, h, "YOLOv8n", "person / vehicle / bike detection", "#dcfce7")
    add_box(ax, xs[2], y, w, h, "Semantic Vector", "16D feature\nROI / looming / motion", "#fef3c7")
    add_box(ax, xs[3], y, w, h, "Multichannel CNN", "channel-wise short-term pattern extraction", "#fae8ff")
    add_box(ax, xs[4], y, w, h, "GRU", "temporal context modeling", "#fee2e2")
    add_box(ax, xs[5], y, w, h, "Hybrid Rule Gate", "context refinement\nbrake warning / hard brake", "#e0f2fe")

    for idx in range(len(xs) - 1):
        add_arrow(ax, (xs[idx] + w, y + h / 2), (xs[idx + 1], y + h / 2))

    add_box(ax, 0.13, 0.28, 0.28, 0.15, "Context Heads", "normal_drive / follow / brake_warning /\nhard_brake_risk / recovery / dense_traffic", "#ede9fe")
    add_box(ax, 0.46, 0.28, 0.18, 0.15, "Boundary Head", "event transition score", "#cffafe")
    add_box(ax, 0.68, 0.28, 0.20, 0.15, "Output", "context tag + event selection +\non-device data curation", "#bbf7d0")

    add_arrow(ax, (xs[4] + w / 2, y), (0.27, 0.43))
    add_arrow(ax, (xs[4] + w / 2, y), (0.55, 0.43))
    add_arrow(ax, (xs[5] + w / 2, y), (0.78, 0.43))

    add_box(ax, 0.05, 0.06, 0.40, 0.12, "Key Feature Set", "det_norm, mean_area, vehicle/person/bike ratio,\nroi_risk, motion_delta, center_closeness, looming_score, occlusion_score", "#fde68a")
    add_box(ax, 0.55, 0.06, 0.35, 0.12, "On-Device Property", "lightweight detector + fixed-size vector + small temporal model\nfor future Jetson Orin Nano deployment", "#bae6fd")

    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    env_csv = ROOT / "paper_experiment_environment_table.csv"
    env_md = ROOT / "paper_experiment_environment_table.md"
    metric_csv = ROOT / "paper_evaluation_metrics_table.csv"
    metric_md = ROOT / "paper_evaluation_metrics_table.md"
    diagram_png = ROOT / "paper_model_diagram.png"

    write_csv(env_csv, ENV_HEADERS, ENV_ROWS)
    write_md(env_md, "실험 환경 표", ENV_HEADERS, ENV_ROWS)
    write_csv(metric_csv, METRIC_HEADERS, METRIC_ROWS)
    write_md(metric_md, "평가 지표 표", METRIC_HEADERS, METRIC_ROWS)
    export_model_diagram(diagram_png)

    print(str(env_csv))
    print(str(env_md))
    print(str(metric_csv))
    print(str(metric_md))
    print(str(diagram_png))


if __name__ == "__main__":
    main()
