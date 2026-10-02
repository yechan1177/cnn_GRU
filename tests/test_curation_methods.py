"""`vcp.vla.curation`의 10개 선별법·선별 품질 지표 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from vcp.vla.curation import METHODS, clip_starts, select, selection_stats

CLIP = 10


def _synthetic(seed: int = 0) -> dict[str, np.ndarray]:
    """길이가 다른 그룹 6개, 일부 그룹에만 위험 구간이 있는 합성 신호."""

    rng = np.random.default_rng(seed)
    lengths = [95, 120, 60, 143, 100, 82]
    group = np.repeat(np.arange(len(lengths)), lengths)
    n = len(group)
    hazard = np.zeros(n, dtype=bool)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    # 그룹 1, 3, 4에 위험 구간을 둔다(클립 격자 위에 정확히 놓이게 그룹 시작 + 20..39).
    for gi in (1, 3, 4):
        hazard[offsets[gi] + 20 : offsets[gi] + 40] = True
    labels = np.where(hazard, 2, rng.integers(0, 2, n)).astype(np.int16)
    return {
        "group": group,
        "labels": labels,
        "hazard": hazard,
        "event_score": np.where(hazard, 0.9, 0.1) + 0.05 * rng.random(n),
        "entropy": rng.random(n),
        "features": rng.normal(size=(n, 5)).astype(np.float32),
        "action": np.where(hazard, -3.0, 0.0) + 0.1 * rng.normal(size=n),
        "ittc": np.where(hazard, 0.8, 0.0) + 0.01 * rng.random(n),
        "loss": rng.random(n),
    }


def _kwargs(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "event_score": d["event_score"],
        "entropy": d["entropy"],
        "features": d["features"],
        "action": d["action"],
        "ittc": d["ittc"],
        "oracle": d["hazard"],
        "loss": d["loss"],
    }


def test_clip_starts_respect_groups() -> None:
    group = np.repeat([0, 1, 2], [25, 9, 31])
    starts = clip_starts(group, CLIP)
    assert starts.tolist() == [0, 10, 34, 44, 54]
    for s in starts:
        assert group[s] == group[s + CLIP - 1]
    with pytest.raises(ValueError):
        clip_starts(group, 0)


@pytest.mark.parametrize("budget", [0.05, 0.1, 0.2])
def test_all_methods_select_same_frame_count_within_groups(budget: float) -> None:
    d = _synthetic()
    n = len(d["group"])
    k = max(1, int(round(budget * n / CLIP)))
    starts = set(clip_starts(d["group"], CLIP).tolist())
    for method in METHODS:
        mask = select(method, budget, d["group"], CLIP, np.random.default_rng(3), **_kwargs(d))
        assert mask.dtype == bool and mask.shape == (n,)
        assert mask.sum() == k * CLIP, method
        # 선택된 연속 구간은 클립 격자 위에서 시작하고 그룹 경계를 넘지 않는다.
        sel = np.flatnonzero(mask)
        run_starts = sel[np.concatenate([[True], np.diff(sel) > 1])]
        for rs in run_starts:
            assert rs in starts, (method, rs)
        clip_ids = [s for s in starts if mask[s]]
        assert len(clip_ids) == k, method
        for s in clip_ids:
            assert mask[s : s + CLIP].all()
            assert d["group"][s] == d["group"][s + CLIP - 1]


def test_missing_input_raises() -> None:
    d = _synthetic()
    rng = np.random.default_rng(0)
    for method, need in [
        ("event", "event_score"),
        ("uncertainty", "entropy"),
        ("coreset", "features"),
        ("action_trigger", "action"),
        ("rule_ittc", "ittc"),
        ("oracle", "oracle"),
        ("offline_loss", "loss"),
        ("ours", "entropy"),
    ]:
        kw = _kwargs(d)
        kw[need] = None
        with pytest.raises(ValueError):
            select(method, 0.1, d["group"], CLIP, rng, **kw)
    with pytest.raises(ValueError):
        select("unknown", 0.1, d["group"], CLIP, rng)
    with pytest.raises(ValueError):
        select("event", 0.1, d["group"], CLIP, rng, event_score=d["event_score"][:-1])


@pytest.mark.parametrize("method", ["oracle", "event", "action_trigger", "rule_ittc"])
def test_signal_methods_pick_hazard_segments(method: str) -> None:
    d = _synthetic()
    n = len(d["group"])
    budget = 6 * CLIP / n  # 위험 클립 6개와 정확히 같은 예산
    mask = select(method, budget, d["group"], CLIP, np.random.default_rng(0), **_kwargs(d))
    assert np.array_equal(mask, d["hazard"]), method


def test_oracle_ties_are_broken_randomly() -> None:
    d = _synthetic()
    n = len(d["group"])
    budget = 2 * CLIP / n  # 위험 클립 6개 중 2개만 고를 수 있다(모두 비율 1.0으로 동률)
    picks = set()
    for seed in range(8):
        mask = select("oracle", budget, d["group"], CLIP, np.random.default_rng(seed), oracle=d["hazard"])
        assert (mask & d["hazard"]).sum() == 2 * CLIP
        picks.add(tuple(np.flatnonzero(mask)[::CLIP].tolist()))
    assert len(picks) > 1


def test_ours_reservoir_fraction() -> None:
    d = _synthetic()
    n = len(d["group"])
    # 이벤트 점수만 의미 있게: 위험 클립 6개가 점수 상위.
    budget = 10 * CLIP / n
    for reservoir, n_top in [(0.0, 10), (0.4, 6), (1.0, 0)]:
        kw = _kwargs(d)
        kw["entropy"] = np.zeros(n)
        mask = select("ours", budget, d["group"], CLIP, np.random.default_rng(1), reservoir=reservoir, **kw)
        assert mask.sum() == 10 * CLIP
        if n_top >= 6:
            # 점수 상위 몫이 위험 클립 6개를 모두 덮는다.
            assert (mask & d["hazard"]).sum() == d["hazard"].sum()
    # reservoir=0.4 → 점수 상위 6개 = 위험 클립 6개, 나머지 4개는 무작위(위험 아님)
    kw = _kwargs(d)
    kw["entropy"] = np.zeros(n)
    mask = select("ours", budget, d["group"], CLIP, np.random.default_rng(2), reservoir=0.4, **kw)
    assert (mask & d["hazard"]).sum() == 6 * CLIP
    assert (mask & ~d["hazard"]).sum() == 4 * CLIP


def test_ours_per_group_cap() -> None:
    d = _synthetic()
    n = len(d["group"])
    kw = _kwargs(d)
    kw["entropy"] = np.zeros(n)
    budget = 6 * CLIP / n
    mask = select("ours", budget, d["group"], CLIP, np.random.default_rng(0), reservoir=0.0, per_group_cap=1, **kw)
    assert mask.sum() == 6 * CLIP
    per_group = np.bincount(d["group"][mask], minlength=6) // CLIP
    assert per_group.max() == 1
    assert (per_group > 0).sum() == 6


def test_uniform_spreads_over_groups() -> None:
    d = _synthetic()
    n = len(d["group"])
    budget = 12 * CLIP / n
    mask = select("uniform", budget, d["group"], CLIP, np.random.default_rng(0))
    per_group = np.bincount(d["group"][mask], minlength=6) // CLIP
    assert per_group.sum() == 12
    assert (per_group > 0).all()
    lengths = np.bincount(d["group"])
    expected = 12 * lengths / lengths.sum()
    assert np.all(np.abs(per_group - expected) < 1.0 + 1e-9)


def test_coreset_deterministic_and_diverse() -> None:
    d = _synthetic()
    m1 = select("coreset", 0.1, d["group"], CLIP, np.random.default_rng(7), features=d["features"])
    m2 = select("coreset", 0.1, d["group"], CLIP, np.random.default_rng(7), features=d["features"])
    assert np.array_equal(m1, m2)
    # 특징이 두 군집이면 2개 클립 예산에서 두 군집을 하나씩 고른다.
    n = len(d["group"])
    feats = np.zeros((n, 3), dtype=np.float32)
    feats[d["hazard"]] = 10.0
    mask = select("coreset", 2 * CLIP / n, d["group"], CLIP, np.random.default_rng(0), features=feats)
    assert (mask & d["hazard"]).sum() == CLIP
    assert (mask & ~d["hazard"]).sum() == CLIP


def test_selection_stats_values() -> None:
    group = np.repeat([0, 1, 2, 3], 10)
    labels = np.zeros(40, dtype=np.int64)
    labels[2:5] = 2  # 그룹 0 위험 구간
    labels[12:14] = 3  # 그룹 1 위험 구간
    labels[30:32] = 1
    mask = np.zeros(40, dtype=bool)
    mask[0:10] = True  # 그룹 0 전체
    mask[30:35] = True  # 그룹 3 일부
    s = selection_stats(mask, labels, group, n_classes=4, hazard_ids=(2, 3))
    assert s["n_selected"] == 15
    assert s["selected_ratio"] == pytest.approx(15 / 40)
    # 선택 라벨: 0이 10개, 1이 2개, 2가 3개
    assert s["class_hist"] == pytest.approx([10 / 15, 2 / 15, 3 / 15, 0.0])
    p = np.array([10, 2, 3]) / 15
    assert s["label_entropy"] == pytest.approx(float(-(p * np.log(p)).sum()))
    assert s["label_entropy_norm"] == pytest.approx(s["label_entropy"] / np.log(4))
    assert s["hazard_frame_recall"] == pytest.approx(3 / 5)
    assert s["n_hazard_events"] == 2
    assert s["hazard_event_recall"] == pytest.approx(0.5)
    assert s["group_coverage"] == pytest.approx(0.5)
    empty = selection_stats(np.zeros(40, dtype=bool), labels, group, 4, (2,))
    assert empty["n_selected"] == 0 and empty["label_entropy"] == 0.0 and empty["group_coverage"] == 0.0
