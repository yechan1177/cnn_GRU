from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class BuildContextDatasetArgs:
    """수동 맥락 annotation을 processed dataset으로 변환할 때 사용하는 인자."""

    annotations_root: Path
    runs_root: Path
    output_root: Path
    dataset_name: str
    window_size: int
    val_ratio: float


def parse_args() -> BuildContextDatasetArgs:
    parser = argparse.ArgumentParser(description="수동 맥락 annotation 기반 Temporal 학습 데이터셋 생성")
    parser.add_argument("--annotations-root", type=str, default="data/annotations/context")
    parser.add_argument("--runs-root", type=str, default="outputs/runs")
    parser.add_argument("--output-root", type=str, default="data/processed")
    parser.add_argument("--dataset-name", type=str, default="manual_context_bootstrap")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()
    return BuildContextDatasetArgs(
        annotations_root=Path(args.annotations_root),
        runs_root=Path(args.runs_root),
        output_root=Path(args.output_root),
        dataset_name=str(args.dataset_name),
        window_size=max(2, int(args.window_size)),
        val_ratio=max(0.0, min(0.5, float(args.val_ratio))),
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
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


def load_annotation_segments(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*_context_segments.jsonl")):
        for row in read_jsonl(path):
            row["_source_path"] = str(path)
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"annotation 파일이 없습니다: {root}")
    return rows


def load_run_frames(runs_root: Path, run_id: str) -> list[dict[str, Any]]:
    path = runs_root / run_id / "frame_records.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"run frame_records 파일이 없습니다: {path}")
    rows = read_jsonl(path)
    return sorted(rows, key=lambda item: int(item.get("frame_id", 0)))


def build_label_map(segments: list[dict[str, Any]]) -> dict[str, int]:
    labels = sorted({str(segment["context_tag"]) for segment in segments})
    return {label: idx for idx, label in enumerate(labels)}


def extract_feature_vector(frame: dict[str, Any]) -> list[float]:
    derived = frame.get("derived_feature", {})
    sample = derived.get("feature_vector")
    if not isinstance(sample, list):
        sample = derived.get("feature_sample", [])
    values: list[float] = []
    if not isinstance(sample, list):
        return values
    for item in sample:
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            values.append(0.0)
    return values


def split_groups(
    groups: list[tuple[str, list[dict[str, Any]]]], val_ratio: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not groups:
        return [], []
    if len(groups) == 1:
        return groups[0][1], []

    buckets: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {}
    for group_id, rows in groups:
        context_tag = "unknown"
        if rows:
            context_tag = str(rows[0].get("target", {}).get("context_tag", "unknown"))
        buckets.setdefault(context_tag, []).append((group_id, rows))

    train_groups: list[tuple[str, list[dict[str, Any]]]] = []
    val_groups: list[tuple[str, list[dict[str, Any]]]] = []

    for _, bucket in sorted(buckets.items(), key=lambda item: item[0]):
        if len(bucket) == 1:
            train_groups.extend(bucket)
            continue

        val_count = max(1, int(round(len(bucket) * val_ratio)))
        if val_count >= len(bucket):
            val_count = len(bucket) - 1
        train_groups.extend(bucket[:-val_count])
        val_groups.extend(bucket[-val_count:])

    if not val_groups and train_groups:
        val_groups.append(train_groups.pop())

    train_rows = [row for _, rows in train_groups for row in rows]
    val_rows = [row for _, rows in val_groups for row in rows]
    return train_rows, val_rows


def analyze_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    context_distribution: dict[str, int] = {}
    boundary_count = 0
    event_active_count = 0

    for row in samples:
        target = row.get("target", {})
        context_tag = str(target.get("context_tag", "unknown"))
        context_distribution[context_tag] = context_distribution.get(context_tag, 0) + 1
        boundary_count += int(target.get("boundary_label", 0))
        event_active_count += int(target.get("event_active_label", 0))

    feature_dim = 0
    for row in samples:
        seq = row.get("features", {}).get("vision_feature_seq", [])
        for vector in seq:
            if isinstance(vector, list):
                feature_dim = max(feature_dim, len(vector))

    return {
        "sample_count": len(samples),
        "feature_dim": feature_dim,
        "context_distribution": context_distribution,
        "boundary_positive_count": boundary_count,
        "event_active_count": event_active_count,
    }


def build_context_dataset(args: BuildContextDatasetArgs) -> dict[str, Any]:
    segments = load_annotation_segments(args.annotations_root)
    label_map = build_label_map(segments)
    feature_keys: list[str] | None = None

    run_frames_cache: dict[str, list[dict[str, Any]]] = {}
    grouped_samples: list[tuple[str, list[dict[str, Any]]]] = []

    for segment in sorted(segments, key=lambda item: (str(item["run_id"]), int(item["start_frame"]))):
        run_id = str(segment["run_id"])
        if run_id not in run_frames_cache:
            run_frames_cache[run_id] = load_run_frames(args.runs_root, run_id)
        frames = run_frames_cache[run_id]

        start_frame = int(segment["start_frame"])
        end_frame = int(segment["end_frame"])
        if end_frame < start_frame:
            raise ValueError(f"segment frame 범위가 잘못되었습니다: {segment['segment_id']}")

        boundary_frames = {int(item) for item in segment.get("boundary_frames", [])}
        context_tag = str(segment["context_tag"])
        event_active = bool(segment.get("event_active", False))
        segment_samples: list[dict[str, Any]] = []

        for target_frame_id in range(start_frame, end_frame + 1):
            start_idx = max(0, target_frame_id - args.window_size + 1)
            window = frames[start_idx : target_frame_id + 1]
            if not window:
                continue

            feature_seq = [extract_feature_vector(frame) for frame in window]
            if feature_keys is None and feature_seq and feature_seq[0]:
                feature_keys = [f"feature_{idx}" for idx in range(len(feature_seq[0]))]

            target_frame = window[-1]
            scores = target_frame.get("scores", {})
            importance_target = float(scores.get("total_importance", 0.0)) if isinstance(scores, dict) else 0.0

            segment_samples.append(
                {
                    "sample_id": f"{run_id}_{segment['segment_id']}_f{target_frame_id:06d}",
                    "run_id": run_id,
                    "window_size": args.window_size,
                    "frame_ids": [int(frame.get("frame_id", 0)) for frame in window],
                    "sensor_timestamps": [float(frame.get("sensor_timestamp", 0.0)) for frame in window],
                    "features": {
                        "vision_feature_seq": feature_seq,
                        "feature_dim": len(feature_seq[0]) if feature_seq and feature_seq[0] else 0,
                        "feature_keys": feature_keys or [],
                    },
                    "target": {
                        "context_tag": context_tag,
                        "context_index": label_map[context_tag],
                        "boundary_label": 1 if target_frame_id in boundary_frames else 0,
                        "event_active_label": 1 if event_active else 0,
                        "importance_target": importance_target,
                    },
                    "metadata": {
                        "segment_id": str(segment["segment_id"]),
                        "video_path": str(segment["video_path"]),
                        "status": str(segment.get("status", "draft")),
                        "confidence": str(segment.get("confidence", "unknown")),
                        "notes": str(segment.get("notes", "")),
                        "annotation_source": str(segment.get("_source_path", "")),
                    },
                }
            )

        grouped_samples.append((str(segment["segment_id"]), segment_samples))

    all_samples = [row for _, rows in grouped_samples for row in rows]
    train_rows, val_rows = split_groups(grouped_samples, args.val_ratio)

    dataset_id = f"dataset_{args.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset_dir = args.output_root / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_path = dataset_dir / "train.jsonl"
    val_path = dataset_dir / "val.jsonl"
    all_path = dataset_dir / "samples_all.jsonl"
    label_map_path = dataset_dir / "label_map.json"
    analysis_path = dataset_dir / "analysis.json"
    manifest_path = dataset_dir / "dataset_manifest.json"
    annotation_copy_path = dataset_dir / "source_annotations.jsonl"

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_jsonl(all_path, all_samples)
    write_jsonl(annotation_copy_path, segments)
    label_map_path.write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")

    analysis = {
        "dataset_id": dataset_id,
        "train": analyze_samples(train_rows),
        "val": analyze_samples(val_rows),
        "all": analyze_samples(all_samples),
    }
    analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "dataset_id": dataset_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "annotations_root": str(args.annotations_root),
            "runs_root": str(args.runs_root),
            "run_ids": sorted(run_frames_cache.keys()),
        },
        "config": {
            "dataset_name": args.dataset_name,
            "window_size": args.window_size,
            "val_ratio": args.val_ratio,
            "split_strategy": "context_stratified_segment_validation",
            "feature_source": "frame_records.derived_feature.feature_vector_or_feature_sample",
        },
        "counts": {
            "segments": len(segments),
            "samples_all": len(all_samples),
            "samples_train": len(train_rows),
            "samples_val": len(val_rows),
        },
        "paths": {
            "dataset_dir": str(dataset_dir),
            "samples_all": str(all_path),
            "train": str(train_path),
            "val": str(val_path),
            "label_map": str(label_map_path),
            "analysis": str(analysis_path),
            "source_annotations": str(annotation_copy_path),
        },
        "notes": {
            "annotation_status": "draft",
            "limitation": "현재 annotation 수가 적고 draft 상태이며, ROI/추적 기반 추가 feature는 아직 포함되지 않음",
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_context_dataset(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
