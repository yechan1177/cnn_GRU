from __future__ import annotations

import json
from pathlib import Path

import yaml

from vcp.config import load_config
from vcp.pipeline import VisionContextPipeline


def test_load_default_config() -> None:
    cfg = load_config("configs/default.yaml")
    assert cfg.runtime.max_frames == 120
    assert cfg.temporal.window_size == 8
    assert cfg.storage.base_dir == "outputs"


def test_pipeline_smoke(tmp_path: Path) -> None:
    cfg = {
        "runtime": {"profile": "test_smoke", "max_frames": 24, "seed": 3},
        "camera": {"fps": 15, "width": 224, "height": 224, "channels": 3},
        "spatial": {"model_name": "mock_spatial", "feature_dim": 32},
        "temporal": {"model_name": "gru_mock", "window_size": 6, "num_contexts": 4, "hidden_dim": 16},
        "labels": ["idle", "approach", "manipulate", "handover"],
        "scoring": {
            "context_weight": 0.35,
            "boundary_weight": 0.30,
            "uncertainty_weight": 0.20,
            "novelty_weight": 0.15,
            "high_threshold": 0.45,
            "mid_threshold": 0.30,
        },
        "curation": {"ring_buffer_size": 64, "pre_event_frames": 4, "post_event_frames": 4},
        "storage": {
            "base_dir": str(tmp_path / "outputs"),
            "run_name": "test_run",
            "save_raw_frames": False,
        },
    }
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    pipeline = VisionContextPipeline(config_path=config_path)
    summary = pipeline.run()

    assert summary["frames"] == 24
    assert summary["events"] >= 1

    run_dir = Path(summary["run_dir"])
    exports_dir = Path(summary["exports_dir"])
    assert (run_dir / "frame_records.jsonl").exists()
    assert (run_dir / "event_records.jsonl").exists()
    assert (exports_dir / "aligned_dataset.jsonl").exists()

    frame_lines = (run_dir / "frame_records.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(frame_lines) == 24

    first_frame = json.loads(frame_lines[0])
    assert "sensor_timestamp" in first_frame
    assert "system_timestamp" in first_frame
    assert "storage_policy" in first_frame
