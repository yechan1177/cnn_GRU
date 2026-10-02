from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vcp.experiments.vla_suite import future_action
from vcp.vla import VLAEpisode, export_lerobot_like, narrate, select_clips
from vcp.vla.curation import event_coverage


def test_narration_korean_and_english() -> None:
    ko, en = narrate("hard_brake_risk", speed=12.0, ttc=1.2, confidence=0.9)
    assert "제동" in ko and "brake" in en and "1.2" in ko


def test_future_action_within_group() -> None:
    a = np.array([0, 1, 2, 3, 10, 11], dtype=np.float32)
    g = np.array([0, 0, 0, 0, 1, 1])
    out = future_action(a, g, fps=2.0, horizons=(0.5, 1.0))
    assert out[:, 0].tolist() == [1, 2, 3, 3, 11, 11]
    assert out[:, 1].tolist() == [2, 3, 3, 3, 11, 11]


def test_select_clips_prefers_high_score() -> None:
    score = np.zeros(100)
    score[40:45] = 1.0
    group = np.zeros(100, dtype=int)
    mask = select_clips(score, group, clip_len=10, budget_ratio=0.1)
    assert mask[40:45].all() and mask.sum() == 10
    cov = event_coverage(mask, score > 0, group)
    assert cov["event_coverage"] == 1.0


def test_export_lerobot_like(tmp_path: Path) -> None:
    T = 5
    ep = VLAEpisode(
        episode_index=0, task_ko="과업", task_en="task", fps=10.0, source="unit",
        timestamp=np.arange(T) / 10.0, state=np.zeros((T, 2)), features=np.zeros((T, 16)),
        context_probs=np.full((T, 2), 0.5), action=np.zeros((T, 2)), context=["a"] * T,
        narration_ko=["가"] * T, narration_en=["a"] * T, event_score=np.zeros(T), uncertainty=np.zeros(T),
    )
    out = export_lerobot_like([ep], tmp_path / "ds", "unit", ["a", "b"])
    info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["total_frames"] == T and info["total_episodes"] == 1
    files = list((out / "data" / "chunk-000").iterdir())
    assert len(files) == 1
