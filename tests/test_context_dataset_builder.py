from __future__ import annotations

import json
from pathlib import Path

from vcp.tools.build_context_dataset import BuildContextDatasetArgs, build_context_dataset


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False))
            file.write("\n")


def test_build_context_dataset(tmp_path: Path) -> None:
    annotations_root = tmp_path / "data" / "annotations" / "context"
    runs_root = tmp_path / "outputs" / "runs"
    output_root = tmp_path / "data" / "processed"

    frame_rows = []
    for idx in range(12):
        frame_rows.append(
            {
                "frame_id": idx,
                "sensor_timestamp": idx * 0.1,
                "scores": {"total_importance": 0.6 if idx >= 6 else 0.2},
                "derived_feature": {"feature_sample": [0.1, 0.2, 0.3, 0.4]},
            }
        )

    _write_jsonl(runs_root / "demo_run_001" / "frame_records.jsonl", frame_rows)
    _write_jsonl(
        annotations_root / "demo_context_segments.jsonl",
        [
            {
                "segment_id": "seg_001",
                "run_id": "demo_run_001",
                "video_path": "data/raw/videos/demo.mp4",
                "start_frame": 0,
                "end_frame": 5,
                "context_tag": "normal_drive",
                "boundary_frames": [],
                "event_active": False,
                "status": "draft",
                "confidence": "medium",
                "notes": "초반 일반 주행",
            },
            {
                "segment_id": "seg_002",
                "run_id": "demo_run_001",
                "video_path": "data/raw/videos/demo.mp4",
                "start_frame": 6,
                "end_frame": 11,
                "context_tag": "hard_brake_risk",
                "boundary_frames": [6],
                "event_active": True,
                "status": "draft",
                "confidence": "medium",
                "notes": "후반 이벤트",
            },
        ],
    )

    args = BuildContextDatasetArgs(
        annotations_root=annotations_root,
        runs_root=runs_root,
        output_root=output_root,
        dataset_name="unit_context",
        window_size=4,
        val_ratio=0.5,
    )
    manifest = build_context_dataset(args)

    dataset_dir = Path(manifest["paths"]["dataset_dir"])
    assert dataset_dir.exists()
    assert (dataset_dir / "train.jsonl").exists()
    assert (dataset_dir / "val.jsonl").exists()
    assert (dataset_dir / "label_map.json").exists()
    assert manifest["counts"]["segments"] == 2
    assert manifest["counts"]["samples_all"] == 12
