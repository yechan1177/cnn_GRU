from __future__ import annotations

import numpy as np
import pytest

from vcp.experiments.data import blocked_folds, gather_windows, split_groups, window_index
from vcp.experiments.metrics import auroc, boundary_f1_tolerance, event_metrics, macro_f1
from vcp.sim import CONTEXT_LABELS, simulate_episode
from vcp.sim.world import ROBOT_SCENARIO_TYPES, SCENARIO_TYPES


@pytest.mark.parametrize("scenario", list(SCENARIO_TYPES) + list(ROBOT_SCENARIO_TYPES))
def test_simulator_runs_and_is_deterministic(scenario: str) -> None:
    a = simulate_episode(scenario, seed=11, duration_s=8.0)
    b = simulate_episode(scenario, seed=11, duration_s=8.0)
    assert len(a.frames) == len(a.labels) == 120
    assert a.labels == b.labels
    assert all(0 <= y < len(CONTEXT_LABELS) for y in a.labels)


def test_lead_brake_produces_brake_labels() -> None:
    labels = []
    for seed in range(8):
        labels += simulate_episode("lead_brake", seed=seed).labels
    assert CONTEXT_LABELS.index("brake_warning") in labels or CONTEXT_LABELS.index("hard_brake_risk") in labels


def test_window_index_respects_groups() -> None:
    group = np.array([0, 0, 0, 1, 1])
    idx = window_index(group, 3)
    assert idx.tolist() == [[-1, -1, 0], [-1, 0, 1], [0, 1, 2], [-1, -1, 3], [-1, 3, 4]]
    X = np.arange(10, dtype=np.float32).reshape(5, 2)
    w = gather_windows(X, idx)
    assert w.shape == (5, 3, 2) and float(w[0, 0].sum()) == 0.0


def test_split_groups_disjoint() -> None:
    s = split_groups(np.arange(100), 0.15, 0.15, seed=1)
    assert not set(s["train"]) & set(s["test"]) and not set(s["val"]) & set(s["test"])
    assert len(s["train"]) + len(s["val"]) + len(s["test"]) == 100


def test_blocked_folds_purge() -> None:
    block, folds = blocked_folds(1000, 10, purge=5)
    assert block.max() == 9
    assert not folds[0]["purge_mask"][95:105].any()


def test_metrics_basic() -> None:
    y = np.array([0, 0, 2, 2, 2, 0, 0, 0])
    p = np.array([0, 0, 0, 2, 2, 0, 2, 2])
    g = np.zeros(8, dtype=int)
    t = np.arange(8) / 10.0
    ev = event_metrics(y, p, g, t, (2,), fa_tol_s=0.0)
    assert ev.n_events == 1 and ev.detected == 1 and ev.false_alarms == 1
    assert abs(macro_f1(y, y, 3) - 1.0) < 1e-9
    assert abs(auroc(np.array([0.1, 0.9, 0.8, 0.2]), np.array([0, 1, 1, 0])) - 1.0) < 1e-9
    b = boundary_f1_tolerance(np.array([0, 0, 1, 0, 0]), np.array([0, 0, 0, 0.9, 0]), np.zeros(5, int), 0.5, tol=1)
    assert b["f1"] == 1.0
