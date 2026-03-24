from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

ACTUAL_EXPERIMENTS: list[dict[str, str]] = [
    {
        "experiment_id": "exp_013",
        "label": "Baseline64+t0.45",
        "summary_path": str(
            ROOT
            / "experiments"
            / "exp_013_temporal_manual_context_yolo3cls_t045"
            / "runs"
            / "mcnn_gru_manual_context_yolo3cls_t045_e20_h96_c24"
            / "train_summary.json"
        ),
        "run_dir": str(ROOT / "outputs" / "runs" / "stopcar_yolo3cls_mctx_t045_v3_demo_20260324_104339"),
    },
    {
        "experiment_id": "exp_014",
        "label": "Feat16+BrakeLabel",
        "summary_path": str(
            ROOT
            / "experiments"
            / "exp_014_temporal_brake_transition_feat16"
            / "runs"
            / "mcnn_gru_brake_transition_feat16_t045_e30_h96_c24"
            / "train_summary.json"
        ),
        "run_dir": str(ROOT / "outputs" / "runs" / "stopcar_brake_transition_t045_demo_20260324_105245"),
    },
    {
        "experiment_id": "exp_015",
        "label": "Feat16+ExpandedBrake",
        "summary_path": str(
            ROOT
            / "experiments"
            / "exp_015_temporal_braking_expanded_feat16"
            / "runs"
            / "mcnn_gru_braking_expanded_feat16_t045_e40_h96_c24"
            / "train_summary.json"
        ),
        "run_dir": str(ROOT / "outputs" / "runs" / "stopcar_braking_expanded_t045_demo_20260324_115331"),
    },
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def configure_font() -> None:
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


def summarize_actual_experiment(item: dict[str, str]) -> dict[str, Any]:
    summary = read_json(Path(item["summary_path"]))
    rows = read_jsonl(Path(item["run_dir"]) / "frame_records.jsonl")
    counts = Counter(str(row.get("context_tag", "unknown")) for row in rows)
    brake_rows = [row for row in rows if 333 <= int(row.get("frame_id", 0)) <= 350]
    brake_counter = Counter(str(row.get("context_tag", "unknown")) for row in brake_rows)
    return {
        "experiment_id": item["experiment_id"],
        "label": item["label"],
        "feature_dim": int(summary["feature_dim"]),
        "best_epoch": int(summary["best_epoch"]),
        "val_context_acc": float(summary["best_metrics"]["val_context_acc"]),
        "val_boundary_f1": float(summary["best_metrics"]["val_boundary_f1"]),
        "val_boundary_recall": float(summary["best_metrics"]["val_boundary_recall"]),
        "stopcar_total_frames": len(rows),
        "stopcar_brake_window_frames": len(brake_rows),
        "stopcar_brake_warning_frames": int(brake_counter.get("brake_warning", 0)),
        "stopcar_hard_brake_frames": int(brake_counter.get("hard_brake_risk", 0)),
        "stopcar_follow_frames": int(brake_counter.get("front_vehicle_follow", 0)),
        "stopcar_normal_frames": int(brake_counter.get("normal_drive", 0)),
        "context_counts": dict(counts),
        "run_dir": item["run_dir"],
        "summary_path": item["summary_path"],
    }


def write_actual_table(rows: list[dict[str, Any]]) -> None:
    csv_path = ROOT / "braking_experiment_actual_table.csv"
    md_path = ROOT / "braking_experiment_actual_table.md"
    fields = [
        "experiment_id",
        "label",
        "feature_dim",
        "best_epoch",
        "val_context_acc",
        "val_boundary_f1",
        "val_boundary_recall",
        "stopcar_brake_warning_frames",
        "stopcar_hard_brake_frames",
        "stopcar_follow_frames",
        "stopcar_normal_frames",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})

    md_lines = [
        "# 브레이크 실험 실제 비교표",
        "",
        "| 실험 | 설정 | feature_dim | best_epoch | val_context_acc | val_boundary_f1 | val_boundary_recall | stopcar brake_warning | stopcar hard_brake | stopcar follow | stopcar normal |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['experiment_id']} | {row['label']} | {row['feature_dim']} | {row['best_epoch']} | "
            f"{row['val_context_acc']:.4f} | {row['val_boundary_f1']:.4f} | {row['val_boundary_recall']:.4f} | "
            f"{row['stopcar_brake_warning_frames']} | {row['stopcar_hard_brake_frames']} | "
            f"{row['stopcar_follow_frames']} | {row['stopcar_normal_frames']} |"
        )
    md_lines.extend(
        [
            "",
            "## 출처",
            "- exp_013 train summary / stopcar run",
            "- exp_014 train summary / stopcar run",
            "- exp_015 train summary / stopcar run",
        ]
    )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def write_assumed_comparison_table() -> None:
    csv_path = ROOT / "braking_experiment_assumed_comparison.csv"
    md_path = ROOT / "braking_experiment_assumed_comparison.md"
    rows = [
        {
            "model": "가정 비교군 A",
            "type": "가정",
            "detector": "YOLO only",
            "temporal": "없음",
            "feature": "검출 박스/클래스만 사용",
            "brake_warning_expected": "낮음",
            "hard_brake_expected": "거의 불가",
            "edge_cost": "낮음",
            "note": "단일 프레임 검출만으로 전환 맥락 분리 어려움",
        },
        {
            "model": "가정 비교군 B",
            "type": "가정",
            "detector": "YOLO + 단일 통계 GRU",
            "temporal": "GRU",
            "feature": "ROI/looming 없음",
            "brake_warning_expected": "중간 이하",
            "hard_brake_expected": "낮음",
            "edge_cost": "중간",
            "note": "전방 급팽창/정지 신호 부족",
        },
        {
            "model": "제안 모델(exp_015)",
            "type": "실측",
            "detector": "YOLO 3cls + feat16",
            "temporal": "멀티채널 CNN-GRU",
            "feature": "ROI+looming+occlusion",
            "brake_warning_expected": "일부 검출",
            "hard_brake_expected": "아직 부족",
            "edge_cost": "중간",
            "note": "현재 draft annotation 기준 stopcar에서 brake_warning 3 frame 출력",
        },
    ]
    fields = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# 브레이크 실험 가정 비교군 표",
        "",
        "> 아래 표의 `가정` 행은 실제 측정치가 아니라 구조 비교용 정성 비교이다.",
        "",
        "| 모델 | 구분 | detector | temporal | feature | brake_warning 기대 | hard_brake 기대 | 엣지 비용 | 비고 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['model']} | {row['type']} | {row['detector']} | {row['temporal']} | "
            f"{row['feature']} | {row['brake_warning_expected']} | {row['hard_brake_expected']} | "
            f"{row['edge_cost']} | {row['note']} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def export_metric_chart(rows: list[dict[str, Any]]) -> None:
    configure_font()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    labels = [row["label"] for row in rows]
    context = [row["val_context_acc"] for row in rows]
    boundary = [row["val_boundary_f1"] for row in rows]
    warnings = [row["stopcar_brake_warning_frames"] for row in rows]

    axes[0].bar(labels, context, color=["#8fb339", "#f4a259", "#5b8e7d"])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Validation Context Accuracy")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(labels, boundary, color=["#577590", "#f3722c", "#277da1"])
    axes[1].set_ylim(0, max(0.06, max(boundary) + 0.01))
    axes[1].set_title("Validation Boundary F1")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(labels, warnings, color=["#adb5bd", "#ffb703", "#2a9d8f"])
    axes[2].set_title("Stopcar brake_warning frames (333-350)")
    axes[2].tick_params(axis="x", rotation=20)

    fig.suptitle("Braking experiment comparison (measured)")
    fig.tight_layout()
    fig.savefig(ROOT / "braking_experiment_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def export_stopcar_timeline(rows: list[dict[str, Any]]) -> None:
    configure_font()
    context_order = [
        "normal_drive",
        "front_vehicle_follow",
        "brake_warning",
        "hard_brake_risk",
        "post_brake_recovery",
        "dense_traffic",
    ]
    context_index = {name: idx for idx, name in enumerate(context_order)}

    fig, axes = plt.subplots(len(rows), 1, figsize=(14, 7), sharex=True)
    if len(rows) == 1:
        axes = [axes]

    for axis, row in zip(axes, rows, strict=True):
        frame_rows = read_jsonl(Path(row["run_dir"]) / "frame_records.jsonl")
        window_rows = [item for item in frame_rows if 320 <= int(item["frame_id"]) <= 360]
        xs = [int(item["frame_id"]) for item in window_rows]
        ys = [context_index.get(str(item.get("context_tag", "normal_drive")), 0) for item in window_rows]
        boundary_vals = [float(item.get("scores", {}).get("boundary", 0.0)) for item in window_rows]
        axis.step(xs, ys, where="mid", color="#1d3557", linewidth=2)
        axis2 = axis.twinx()
        axis2.plot(xs, boundary_vals, color="#e76f51", linewidth=1.5, alpha=0.8)
        axis.set_yticks(range(len(context_order)))
        axis.set_yticklabels(context_order)
        axis.set_ylim(-0.5, len(context_order) - 0.5)
        axis.set_title(row["label"])
        axis.grid(alpha=0.2)
        axis2.set_ylim(0, 1)
        axis2.set_ylabel("boundary", color="#e76f51")

    axes[-1].set_xlabel("frame id")
    fig.suptitle("Stopcar context timeline around brake window (320-360)")
    fig.tight_layout()
    fig.savefig(ROOT / "braking_stopcar_timeline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def export_source_manifest(rows: list[dict[str, Any]]) -> None:
    manifest = {
        "actual_experiments": rows,
        "root_outputs": {
            "actual_table_csv": str(ROOT / "braking_experiment_actual_table.csv"),
            "actual_table_md": str(ROOT / "braking_experiment_actual_table.md"),
            "assumed_table_csv": str(ROOT / "braking_experiment_assumed_comparison.csv"),
            "assumed_table_md": str(ROOT / "braking_experiment_assumed_comparison.md"),
            "metrics_chart": str(ROOT / "braking_experiment_metrics.png"),
            "timeline_chart": str(ROOT / "braking_stopcar_timeline.png"),
        },
        "note": "가정 비교군 표는 구조 비교용 정성 비교이며 실제 측정치가 아니다.",
    }
    (ROOT / "braking_experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    rows = [summarize_actual_experiment(item) for item in ACTUAL_EXPERIMENTS]
    write_actual_table(rows)
    write_assumed_comparison_table()
    export_metric_chart(rows)
    export_stopcar_timeline(rows)
    export_source_manifest(rows)
    print(
        json.dumps(
            {
                "actual_rows": rows,
                "outputs": [
                    "braking_experiment_actual_table.csv",
                    "braking_experiment_actual_table.md",
                    "braking_experiment_assumed_comparison.csv",
                    "braking_experiment_assumed_comparison.md",
                    "braking_experiment_metrics.png",
                    "braking_stopcar_timeline.png",
                    "braking_experiment_manifest.json",
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
