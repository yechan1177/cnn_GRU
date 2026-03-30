from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


LABEL_ORDER = [
    "normal_drive",
    "front_vehicle_follow",
    "brake_warning",
    "hard_brake_risk",
    "post_brake_recovery",
    "dense_traffic",
]


@dataclass(slots=True)
class RuleDatasetArgs:
    """순간 feature sequence 기반 pseudo-label 데이터셋 생성 인자."""

    runs_root: Path
    output_root: Path
    dataset_name: str
    run_glob: str
    window_size: int
    val_ratio: float
    baseline_window: int


@dataclass(slots=True)
class RuleState:
    """브레이크 hold 상태를 위한 내부 상태."""

    hard_hold: int = 0
    warn_hold: int = 0
    recovery_hold: int = 0


def parse_args() -> RuleDatasetArgs:
    parser = argparse.ArgumentParser(description="순간 feature sequence 규칙 기반 temporal dataset 생성")
    parser.add_argument("--runs-root", type=str, default="outputs/runs")
    parser.add_argument("--output-root", type=str, default="data/processed")
    parser.add_argument("--dataset-name", type=str, default="rule_context_instant_feat16")
    parser.add_argument("--run-glob", type=str, default="*_feat16instant_demo_*")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--baseline-window", type=int, default=8)
    args = parser.parse_args()
    return RuleDatasetArgs(
        runs_root=Path(args.runs_root),
        output_root=Path(args.output_root),
        dataset_name=str(args.dataset_name),
        run_glob=str(args.run_glob),
        window_size=max(2, int(args.window_size)),
        val_ratio=max(0.0, min(0.5, float(args.val_ratio))),
        baseline_window=max(3, int(args.baseline_window)),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False))
            file.write("\n")


def load_run_frames(runs_root: Path, run_glob: str) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: list[tuple[str, list[dict[str, Any]]]] = []
    for run_dir in sorted(runs_root.glob(run_glob)):
        if not run_dir.is_dir():
            continue
        frame_path = run_dir / "frame_records.jsonl"
        if not frame_path.exists():
            continue
        rows = read_jsonl(frame_path)
        if rows:
            rows = sorted(rows, key=lambda item: int(item.get("frame_id", 0)))
            groups.append((run_dir.name, rows))
    if not groups:
        raise FileNotFoundError(f"조건에 맞는 run이 없습니다: root={runs_root}, glob={run_glob}")
    return groups


def split_groups(groups: list[tuple[str, list[dict[str, Any]]]], val_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(groups) <= 1:
        rows = [row for _, group_rows in groups for row in group_rows]
        return rows, []
    val_count = max(1, int(round(len(groups) * val_ratio)))
    if val_count >= len(groups):
        val_count = len(groups) - 1
    train_rows = [row for _, group_rows in groups[:-val_count] for row in group_rows]
    val_rows = [row for _, group_rows in groups[-val_count:] for row in group_rows]
    return train_rows, val_rows


def analyze_samples(rows: list[dict[str, Any]]) -> dict[str, Any]:
    context_distribution: dict[str, int] = {}
    boundary_positive_count = 0
    event_active_count = 0
    feature_dim = 0
    for row in rows:
        target = row.get("target", {})
        context_tag = str(target.get("context_tag", "unknown"))
        context_distribution[context_tag] = context_distribution.get(context_tag, 0) + 1
        boundary_positive_count += int(target.get("boundary_label", 0))
        event_active_count += int(target.get("event_active_label", 0))
        for vector in row.get("features", {}).get("vision_feature_seq", []):
            if isinstance(vector, list):
                feature_dim = max(feature_dim, len(vector))
    return {
        "sample_count": len(rows),
        "feature_dim": feature_dim,
        "context_distribution": context_distribution,
        "boundary_positive_count": boundary_positive_count,
        "event_active_count": event_active_count,
    }


def build_feature_map(frame: dict[str, Any]) -> dict[str, float]:
    derived = frame.get("derived_feature", {})
    vector = derived.get("feature_vector", [])
    keys = derived.get("feature_keys", [])
    mapping: dict[str, float] = {}
    if isinstance(vector, list) and isinstance(keys, list):
        for idx, key in enumerate(keys):
            if idx >= len(vector):
                break
            try:
                mapping[str(key)] = float(vector[idx])
            except (TypeError, ValueError):
                mapping[str(key)] = 0.0
    return mapping


def compute_baseline(history: list[dict[str, float]], baseline_window: int) -> dict[str, float]:
    if not history:
        return {
            "det_norm": 0.0,
            "mean_area": 0.0,
            "vehicle_ratio": 0.0,
            "person_ratio": 0.0,
            "bike_ratio": 0.0,
            "roi_risk": 0.0,
            "center_closeness": 0.0,
            "roi_mean_area": 0.0,
            "roi_count_norm": 0.0,
        }
    subset = history[-baseline_window:]
    count = float(len(subset))
    return {
        "det_norm": sum(item.get("det_norm", 0.0) for item in subset) / count,
        "mean_area": sum(item.get("mean_area", 0.0) for item in subset) / count,
        "vehicle_ratio": sum(item.get("vehicle_ratio", 0.0) for item in subset) / count,
        "person_ratio": sum(item.get("person_ratio", 0.0) for item in subset) / count,
        "bike_ratio": sum(item.get("bike_ratio", 0.0) for item in subset) / count,
        "roi_risk": sum(item.get("roi_risk", 0.0) for item in subset) / count,
        "center_closeness": sum(item.get("center_closeness", 0.0) for item in subset) / count,
        "roi_mean_area": sum(item.get("roi_mean_area", 0.0) for item in subset) / count,
        "roi_count_norm": sum(item.get("roi_count_norm", 0.0) for item in subset) / count,
    }


def label_frame(feature: dict[str, float], baseline: dict[str, float], state: RuleState) -> tuple[str, int, bool, dict[str, float]]:
    det_norm = float(feature.get("det_norm", 0.0))
    mean_area = float(feature.get("mean_area", 0.0))
    vehicle_ratio = float(feature.get("vehicle_ratio", 0.0))
    person_ratio = float(feature.get("person_ratio", 0.0))
    bike_ratio = float(feature.get("bike_ratio", 0.0))
    roi_risk = float(feature.get("roi_risk", 0.0))
    center = float(feature.get("center_closeness", 0.0))
    roi_mean_area = float(feature.get("roi_mean_area", 0.0))
    roi_count_norm = float(feature.get("roi_count_norm", 0.0))

    vru_ratio = person_ratio + bike_ratio
    baseline_roi_area = max(1e-4, baseline.get("roi_mean_area", 0.0))
    roi_area_delta = roi_mean_area - baseline.get("roi_mean_area", 0.0)
    roi_risk_delta = roi_risk - baseline.get("roi_risk", 0.0)
    center_delta = center - baseline.get("center_closeness", 0.0)
    area_jump = roi_mean_area / baseline_roi_area

    stable_follow = vehicle_ratio >= 0.12 and roi_risk >= 0.20 and center >= 0.55 and roi_mean_area >= 0.002
    dense_cond = det_norm >= 0.12 and vehicle_ratio >= 0.28 and roi_risk < 0.18
    warn_cond = (
        (vru_ratio >= 0.06 and roi_risk >= 0.20 and center >= 0.60 and roi_mean_area >= 0.004)
        or (stable_follow and roi_area_delta >= 0.010 and area_jump >= 1.35)
        or (stable_follow and roi_risk_delta >= 0.12 and center_delta >= 0.08)
        or (center >= 0.72 and area_jump >= 2.10 and roi_mean_area >= 0.008)
    )
    hard_cond = (
        (vru_ratio >= 0.08 and roi_risk >= 0.30 and center >= 0.76 and roi_mean_area >= 0.014)
        or (stable_follow and roi_area_delta >= 0.018 and area_jump >= 1.80)
        or (center >= 0.78 and area_jump >= 2.60 and roi_mean_area >= 0.016)
        or (roi_risk >= 0.48 and center >= 0.84 and roi_mean_area >= 0.022)
    )

    boundary = 0
    if state.hard_hold > 0:
        state.hard_hold -= 1
        label = "hard_brake_risk"
    elif hard_cond:
        state.hard_hold = 2
        state.warn_hold = 0
        state.recovery_hold = 3
        label = "hard_brake_risk"
        boundary = 1
    elif state.warn_hold > 0:
        state.warn_hold -= 1
        label = "brake_warning"
    elif warn_cond:
        state.warn_hold = 2
        state.recovery_hold = max(state.recovery_hold, 3)
        label = "brake_warning"
        boundary = 1
    elif state.recovery_hold > 0:
        state.recovery_hold -= 1
        label = "post_brake_recovery"
    elif dense_cond:
        label = "dense_traffic"
    elif stable_follow:
        label = "front_vehicle_follow"
    else:
        label = "normal_drive"

    event_active = label in {"brake_warning", "hard_brake_risk", "post_brake_recovery"}
    debug = {
        "roi_area_delta": round(roi_area_delta, 6),
        "roi_risk_delta": round(roi_risk_delta, 6),
        "center_delta": round(center_delta, 6),
        "area_jump": round(area_jump, 6),
        "warn_cond": 1.0 if warn_cond else 0.0,
        "hard_cond": 1.0 if hard_cond else 0.0,
        "roi_count_norm": round(roi_count_norm, 6),
        "mean_area": round(mean_area, 6),
    }
    return label, boundary, event_active, debug


def build_samples_for_run(run_id: str, frames: list[dict[str, Any]], window_size: int, baseline_window: int, label_map: dict[str, int]) -> list[dict[str, Any]]:
    feature_history: list[dict[str, float]] = []
    labels: list[dict[str, Any]] = []
    feature_keys: list[str] | None = None
    state = RuleState()

    for frame in frames:
        feature_map = build_feature_map(frame)
        if feature_keys is None:
            keys = frame.get("derived_feature", {}).get("feature_keys")
            if isinstance(keys, list) and keys:
                feature_keys = [str(item) for item in keys]
        baseline = compute_baseline(feature_history, baseline_window)
        label, boundary, event_active, debug = label_frame(feature_map, baseline, state)
        labels.append(
            {
                "frame_id": int(frame.get("frame_id", 0)),
                "context_tag": label,
                "boundary": boundary,
                "event_active": event_active,
                "debug": debug,
            }
        )
        feature_history.append(feature_map)

    samples: list[dict[str, Any]] = []
    frame_by_id = {int(frame.get("frame_id", 0)): frame for frame in frames}
    for idx, target in enumerate(labels):
        frame_id = int(target["frame_id"])
        start_idx = max(0, idx - window_size + 1)
        window_labels = labels[start_idx : idx + 1]
        window_frames = [frame_by_id[int(item["frame_id"])] for item in window_labels]
        feature_seq = [frame.get("derived_feature", {}).get("feature_vector", []) for frame in window_frames]
        feature_seq = [[float(item) for item in vector] for vector in feature_seq]
        target_frame = window_frames[-1]

        importance_target = 0.0
        scores = target_frame.get("scores", {})
        if isinstance(scores, dict):
            try:
                importance_target = float(scores.get("total_importance", 0.0))
            except (TypeError, ValueError):
                importance_target = 0.0

        samples.append(
            {
                "sample_id": f"{run_id}_rule_f{frame_id:06d}",
                "run_id": run_id,
                "window_size": window_size,
                "frame_ids": [int(item["frame_id"]) for item in window_frames],
                "sensor_timestamps": [float(frame.get("sensor_timestamp", 0.0)) for frame in window_frames],
                "features": {
                    "vision_feature_seq": feature_seq,
                    "feature_dim": len(feature_seq[-1]) if feature_seq and feature_seq[-1] else 0,
                    "feature_keys": feature_keys or [],
                },
                "target": {
                    "context_tag": str(target["context_tag"]),
                    "context_index": int(label_map[str(target["context_tag"])]),
                    "boundary_label": int(target["boundary"]),
                    "event_active_label": 1 if bool(target["event_active"]) else 0,
                    "importance_target": importance_target,
                },
                "metadata": {
                    "source_video": str(target_frame.get("metadata", {}).get("source_video", "")),
                    "annotation_source": "instant_feature_rule_pseudolabel",
                    "debug": target["debug"],
                },
            }
        )
    return samples


def build_rule_context_dataset(args: RuleDatasetArgs) -> dict[str, Any]:
    label_map = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    run_groups = load_run_frames(args.runs_root, args.run_glob)
    grouped_samples: list[tuple[str, list[dict[str, Any]]]] = []
    for run_id, frames in run_groups:
        grouped_samples.append((run_id, build_samples_for_run(run_id, frames, args.window_size, args.baseline_window, label_map)))

    all_samples = [row for _, rows in grouped_samples for row in rows]
    train_rows, val_rows = split_groups(grouped_samples, args.val_ratio)

    dataset_id = f"dataset_{args.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset_dir = args.output_root / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(dataset_dir / "train.jsonl", train_rows)
    write_jsonl(dataset_dir / "val.jsonl", val_rows)
    write_jsonl(dataset_dir / "samples_all.jsonl", all_samples)
    (dataset_dir / "label_map.json").write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")

    analysis = {
        "dataset_id": dataset_id,
        "train": analyze_samples(train_rows),
        "val": analyze_samples(val_rows),
        "all": analyze_samples(all_samples),
        "runs": [run_id for run_id, _ in run_groups],
        "rule_config": {
            "window_size": args.window_size,
            "baseline_window": args.baseline_window,
            "run_glob": args.run_glob,
        },
    }
    (dataset_dir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "dataset_id": dataset_id,
        "dataset_dir": str(dataset_dir.resolve()),
        "train_path": str((dataset_dir / "train.jsonl").resolve()),
        "val_path": str((dataset_dir / "val.jsonl").resolve()),
        "sample_count": len(all_samples),
        "run_count": len(run_groups),
        "label_map": label_map,
        "run_glob": args.run_glob,
        "window_size": args.window_size,
        "baseline_window": args.baseline_window,
    }
    (dataset_dir / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_rule_context_dataset(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
