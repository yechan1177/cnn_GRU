from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


TARGET_NAMES = ["person", "vehicle", "bike"]
TARGET_INDEX = {name: idx for idx, name in enumerate(TARGET_NAMES)}

CLASS_MAP = {
    "person": "person",
    "pedestrian": "person",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "cng": "vehicle",
    "other-vehicle": "vehicle",
    "other_vehicle": "vehicle",
    "rickshaw": "vehicle",
    "bicycle": "bike",
    "motorcycle": "bike",
    "bike": "bike",
    "biker": "bike",
}


@dataclass(slots=True)
class RemapArgs:
    dataset_root: Path
    output_dir: Path
    seed: int
    train_ratio: float
    val_ratio: float
    link_mode: str


def parse_args() -> RemapArgs:
    parser = argparse.ArgumentParser(description="YOLO 데이터셋을 person/vehicle/bike 3클래스로 재매핑")
    parser.add_argument("--dataset-root", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="dataset/yolov8_3cls_merged")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--link-mode",
        type=str,
        default="hardlink",
        choices=["hardlink", "copy"],
        help="이미지/라벨 파일 이동 방식",
    )
    args = parser.parse_args()
    return RemapArgs(
        dataset_root=Path(args.dataset_root),
        output_dir=Path(args.output_dir),
        seed=int(args.seed),
        train_ratio=max(0.5, min(0.95, float(args.train_ratio))),
        val_ratio=max(0.01, min(0.3, float(args.val_ratio))),
        link_mode=str(args.link_mode),
    )


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def safe_link_or_copy(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy"
    shutil.copy2(src, dst)
    return "copy"


def read_label_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def remap_label_lines(lines: list[str], class_names: list[str]) -> tuple[list[str], dict[str, int]]:
    output: list[str] = []
    stats = {
        "boxes_input": 0,
        "boxes_kept": 0,
        "boxes_dropped": 0,
    }
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls_idx = int(float(parts[0]))
        except ValueError:
            continue
        if not (0 <= cls_idx < len(class_names)):
            continue
        original_name = str(class_names[cls_idx]).strip().lower()
        stats["boxes_input"] += 1
        mapped_name = CLASS_MAP.get(original_name)
        if mapped_name is None:
            stats["boxes_dropped"] += 1
            continue
        mapped_idx = TARGET_INDEX[mapped_name]
        output.append(" ".join([str(mapped_idx), *parts[1:]]))
        stats["boxes_kept"] += 1
    return output, stats


def collect_structured_samples(dataset_dir: Path) -> list[dict[str, Any]]:
    yaml_cfg = load_yaml(dataset_dir / "data.yaml")
    class_names = [str(item) for item in yaml_cfg.get("names", [])]
    rows: list[dict[str, Any]] = []
    for split in ("train", "valid", "test"):
        image_dir = dataset_dir / split / "images"
        label_dir = dataset_dir / split / "labels"
        if not image_dir.exists():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file():
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            rows.append(
                {
                    "source_dataset": dataset_dir.name,
                    "source_mode": "structured",
                    "split": split,
                    "image_path": image_path,
                    "label_path": label_path,
                    "class_names": class_names,
                }
            )
    return rows


def collect_flat_samples(dataset_dir: Path, seed: int, train_ratio: float, val_ratio: float) -> list[dict[str, Any]]:
    yaml_cfg = load_yaml(dataset_dir / "data.yaml")
    class_names = [str(item) for item in yaml_cfg.get("names", [])]
    export_root = dataset_dir / "export"
    image_dir = export_root / "images"
    label_dir = export_root / "labels"
    image_paths = sorted([path for path in image_dir.iterdir() if path.is_file()])
    rng = random.Random(seed)
    rng.shuffle(image_paths)

    total = len(image_paths)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    rows: list[dict[str, Any]] = []
    for idx, image_path in enumerate(image_paths):
        if idx < train_end:
            split = "train"
        elif idx < val_end:
            split = "valid"
        else:
            split = "test"
        label_path = label_dir / f"{image_path.stem}.txt"
        rows.append(
            {
                "source_dataset": dataset_dir.name,
                "source_mode": "flat_export",
                "split": split,
                "image_path": image_path,
                "label_path": label_path,
                "class_names": class_names,
            }
        )
    return rows


def build_dataset(args: RemapArgs) -> dict[str, Any]:
    source_roots = [
        args.dataset_root / "P2_Dhaka_Dataset.v29i.yolov8",
        args.dataset_root / "Self Driving Car.v3-fixed-small.yolov8",
    ]

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    samples: list[dict[str, Any]] = []
    samples.extend(collect_structured_samples(source_roots[0]))
    samples.extend(
        collect_flat_samples(
            source_roots[1],
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
    )

    stats: dict[str, Any] = {
        "images_total": 0,
        "labels_total": 0,
        "boxes_input": 0,
        "boxes_kept": 0,
        "boxes_dropped": 0,
        "link_mode_requested": args.link_mode,
        "copy_operations": 0,
        "hardlink_operations": 0,
        "split_counts": {"train": 0, "valid": 0, "test": 0},
        "class_box_counts": {name: 0 for name in TARGET_NAMES},
        "source_dataset_counts": {},
    }

    for split in ("train", "valid", "test"):
        (args.output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (args.output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    for row_idx, row in enumerate(samples):
        split = str(row["split"])
        image_path = Path(row["image_path"])
        label_path = Path(row["label_path"])
        class_names = list(row["class_names"])
        source_dataset = str(row["source_dataset"])

        source_count = stats["source_dataset_counts"].setdefault(
            source_dataset, {"images": 0, "boxes_kept": 0, "boxes_dropped": 0}
        )
        source_count["images"] += 1

        out_name = f"{source_dataset}__{image_path.name}"
        out_image_path = args.output_dir / split / "images" / out_name
        out_label_path = args.output_dir / split / "labels" / f"{Path(out_name).stem}.txt"

        link_result = safe_link_or_copy(image_path, out_image_path, args.link_mode)
        if link_result == "hardlink":
            stats["hardlink_operations"] += 1
        else:
            stats["copy_operations"] += 1

        remapped_lines, line_stats = remap_label_lines(read_label_lines(label_path), class_names)
        out_label_path.write_text("\n".join(remapped_lines) + ("\n" if remapped_lines else ""), encoding="utf-8")

        for remapped_line in remapped_lines:
            mapped_idx = int(remapped_line.split()[0])
            stats["class_box_counts"][TARGET_NAMES[mapped_idx]] += 1

        stats["images_total"] += 1
        stats["labels_total"] += 1
        stats["boxes_input"] += line_stats["boxes_input"]
        stats["boxes_kept"] += line_stats["boxes_kept"]
        stats["boxes_dropped"] += line_stats["boxes_dropped"]
        stats["split_counts"][split] += 1
        source_count["boxes_kept"] += line_stats["boxes_kept"]
        source_count["boxes_dropped"] += line_stats["boxes_dropped"]

    yaml_text = {
        "path": str(args.output_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 3,
        "names": TARGET_NAMES,
    }
    (args.output_dir / "data.yaml").write_text(
        yaml.safe_dump(yaml_text, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (args.output_dir / "class_mapping.json").write_text(
        json.dumps(
            {
                "target_names": TARGET_NAMES,
                "source_to_target": CLASS_MAP,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(args.output_dir),
        "sources": [str(path) for path in source_roots],
        "seed": args.seed,
        "train_ratio_for_flat_export": args.train_ratio,
        "val_ratio_for_flat_export": args.val_ratio,
        "stats": stats,
    }
    (args.output_dir / "remap_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_dataset(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
