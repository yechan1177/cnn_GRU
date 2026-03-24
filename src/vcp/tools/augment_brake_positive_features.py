from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BRAKE_WARNING_LABEL = "brake_warning"
HARD_BRAKE_LABEL = "hard_brake_risk"


@dataclass(slots=True)
class AugmentArgs:
    dataset_dir: Path
    output_dir: Path
    warn_factor: int
    hard_factor: int
    seed: int


def parse_args() -> AugmentArgs:
    parser = argparse.ArgumentParser(description="브레이크 positive feature 시퀀스 합성 증강")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--warn-factor", type=int, default=6)
    parser.add_argument("--hard-factor", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    return AugmentArgs(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        warn_factor=max(1, int(args.warn_factor)),
        hard_factor=max(1, int(args.hard_factor)),
        seed=int(args.seed),
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
    with path.open("w", encoding="utf-8", newline="") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_ratios(vec: list[float]) -> None:
    vehicle = clamp01(vec[5])
    person = clamp01(vec[6])
    bike = clamp01(vec[7])
    total = vehicle + person + bike
    if total <= 1e-8:
        vec[5], vec[6], vec[7] = 0.0, 0.0, 0.0
        return
    vec[5] = vehicle / total
    vec[6] = person / total
    vec[7] = bike / total


def augment_sequence(seq: list[list[float]], label: str, rng: random.Random) -> list[list[float]]:
    output: list[list[float]] = []
    length = max(1, len(seq))
    for idx, base_vec in enumerate(seq):
        vec = [float(v) for v in base_vec[:16]]
        if len(vec) < 16:
            vec.extend([0.0] * (16 - len(vec)))

        for dim in range(16):
            vec[dim] = clamp01(vec[dim] + rng.uniform(-0.01, 0.01))

        phase = idx / max(1, length - 1)
        end_weight = 0.35 + (0.65 * phase)

        vehicle_boost = rng.uniform(0.03, 0.10) * end_weight
        person_drop = vehicle_boost * rng.uniform(0.45, 0.85)
        vec[5] = clamp01(vec[5] + vehicle_boost)
        vec[6] = clamp01(vec[6] - person_drop)
        normalize_ratios(vec)

        if label == BRAKE_WARNING_LABEL:
            vec[11] = clamp01(vec[11] + rng.uniform(0.03, 0.09) * end_weight)
            vec[13] = clamp01(vec[13] + rng.uniform(0.02, 0.06) * end_weight)
            vec[14] = clamp01(vec[14] + rng.uniform(0.01, 0.05) * end_weight)
            vec[12] = clamp01(vec[12] + rng.uniform(0.005, 0.03) * phase)
            vec[15] = clamp01(vec[15] + rng.uniform(0.0, 0.05) * phase)
        elif label == HARD_BRAKE_LABEL:
            vec[11] = clamp01(vec[11] + rng.uniform(0.05, 0.12) * end_weight)
            vec[13] = clamp01(vec[13] + rng.uniform(0.03, 0.08) * end_weight)
            vec[14] = clamp01(vec[14] + rng.uniform(0.03, 0.08) * end_weight)
            vec[15] = clamp01(vec[15] + rng.uniform(0.08, 0.20) * end_weight)
            vec[12] = clamp01((vec[12] * rng.uniform(0.1, 0.5)) - rng.uniform(0.0, 0.01))

        vec[1] = clamp01(vec[1] + rng.uniform(-0.01, 0.03))
        vec[2] = clamp01(max(vec[2], vec[1]))
        vec[8] = clamp01(vec[8] + rng.uniform(0.01, 0.04) * end_weight)
        output.append([round(v, 6) for v in vec])
    return output


def summarize_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        tag = str(row.get("target", {}).get("context_tag", "unknown"))
        counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def build_augmented_train(
    train_rows: list[dict[str, Any]],
    warn_factor: int,
    hard_factor: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    augmented = list(train_rows)
    stats = {
        "warning_original": 0,
        "hard_original": 0,
        "warning_synthetic": 0,
        "hard_synthetic": 0,
    }
    for row in train_rows:
        label = str(row.get("target", {}).get("context_tag", ""))
        if label not in {BRAKE_WARNING_LABEL, HARD_BRAKE_LABEL}:
            continue
        factor = warn_factor if label == BRAKE_WARNING_LABEL else hard_factor
        if label == BRAKE_WARNING_LABEL:
            stats["warning_original"] += 1
        else:
            stats["hard_original"] += 1

        seq = row.get("features", {}).get("vision_feature_seq", [])
        if not isinstance(seq, list) or not seq:
            continue

        for aug_idx in range(factor):
            new_row = copy.deepcopy(row)
            new_row["sample_id"] = f"{row.get('sample_id', 'sample')}_aug{aug_idx:03d}"
            new_row["features"]["vision_feature_seq"] = augment_sequence(seq, label, rng)
            new_row.setdefault("metadata", {})
            new_row["metadata"]["synthetic_augmentation"] = True
            new_row["metadata"]["augmentation_source"] = str(row.get("sample_id", "unknown"))
            new_row["metadata"]["augmentation_type"] = "brake_positive_feature_jitter"
            importance = float(new_row.get("target", {}).get("importance_target", 0.0))
            new_row["target"]["importance_target"] = round(clamp01(importance + rng.uniform(0.01, 0.05)), 6)
            augmented.append(new_row)
            if label == BRAKE_WARNING_LABEL:
                stats["warning_synthetic"] += 1
            else:
                stats["hard_synthetic"] += 1
    return augmented, stats


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    train_path = args.dataset_dir / "train.jsonl"
    val_path = args.dataset_dir / "val.jsonl"
    label_map_path = args.dataset_dir / "label_map.json"
    manifest_path = args.dataset_dir / "dataset_manifest.json"
    source_annotations_path = args.dataset_dir / "source_annotations.jsonl"

    for path in [train_path, val_path, label_map_path]:
        if not path.exists():
            raise FileNotFoundError(f"필수 파일이 없습니다: {path}")

    train_rows = read_jsonl(train_path)
    val_rows = read_jsonl(val_path)
    augmented_train, stats = build_augmented_train(train_rows, args.warn_factor, args.hard_factor, rng)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", augmented_train)
    write_jsonl(args.output_dir / "val.jsonl", val_rows)
    write_jsonl(args.output_dir / "samples_all.jsonl", augmented_train + val_rows)

    for src in [label_map_path, source_annotations_path]:
        if src.exists():
            (args.output_dir / src.name).write_text(src.read_text(encoding="utf-8-sig"), encoding="utf-8")

    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    manifest["dataset_name"] = args.output_dir.name
    manifest["source_dataset_dir"] = str(args.dataset_dir)
    manifest["augmentation"] = {
        "type": "synthetic_brake_positive_feature_jitter",
        "warn_factor": args.warn_factor,
        "hard_factor": args.hard_factor,
        "seed": args.seed,
        "stats": stats,
    }
    manifest["counts"] = {
        "train": len(augmented_train),
        "val": len(val_rows),
        "all": len(augmented_train) + len(val_rows),
    }
    (args.output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    analysis = {
        "train_counts": summarize_counts(augmented_train),
        "val_counts": summarize_counts(val_rows),
        "augmentation_stats": stats,
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "analysis": analysis}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
