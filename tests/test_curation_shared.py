"""v3 공유 저장소 선별(`vcp.vla.curation.select_shared`, docs/33 M1) 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from vcp.vla.curation import clip_starts, select, select_shared, shared_reservoir_mask

CLIP = 10


def _group() -> np.ndarray:
    """길이가 다른 6개 그룹 + 같은 id가 떨어져 다시 나오는 구간(경계 검사용)."""

    lengths = [95, 120, 60, 143, 100, 82]
    ids = [0, 1, 2, 3, 1, 4]  # id 1이 두 번(떨어진 구간) 나온다
    return np.repeat(np.asarray(ids), lengths)


def _clip_set(mask: np.ndarray, starts: np.ndarray) -> set[int]:
    return {int(s) for s in starts if mask[s]}


@pytest.mark.parametrize("reservoir", [0.5, 0.8, 0.9])
def test_reservoir_identical_across_scores(reservoir: float) -> None:
    group = _group()
    n = len(group)
    rng0 = np.random.default_rng(0)
    scores = {
        "a": rng0.random(n),
        "b": -rng0.random(n),
        "c": np.zeros(n),  # 전부 동률
        "d": np.where(np.arange(n) % 37 < 5, 1.0, 0.0),
    }
    starts = clip_starts(group, CLIP)
    res = shared_reservoir_mask(0.2, group, CLIP, np.random.default_rng(42), reservoir)
    res_clips = _clip_set(res, starts)
    k = max(1, int(round(0.2 * n / CLIP)))
    assert len(res_clips) == int(round(reservoir * k))
    masks = {}
    for name, sc in scores.items():
        masks[name] = select_shared(
            sc, 0.2, group, CLIP, np.random.default_rng(42), reservoir, entropy=rng0.random(n), lam=0.5
        )
        assert (masks[name] & res).sum() == res.sum(), name  # 저장소는 모든 방법에 그대로 포함된다
    masks["random"] = select_shared(None, 0.2, group, CLIP, np.random.default_rng(42), reservoir)
    assert (masks["random"] & res).sum() == res.sum()
    # 점수 몫은 방법마다 달라야 한다(점수 a와 b는 거의 반대 순서).
    assert not np.array_equal(masks["a"], masks["b"])
    # 다른 시드면 저장소가 달라진다.
    other = shared_reservoir_mask(0.2, group, CLIP, np.random.default_rng(43), reservoir)
    assert not np.array_equal(res, other)


def test_score_share_is_top_of_non_reservoir() -> None:
    group = _group()
    n = len(group)
    starts = clip_starts(group, CLIP)
    score = np.random.default_rng(5).random(n)
    mask = select_shared(score, 0.1, group, CLIP, np.random.default_rng(7), 0.5)
    res = shared_reservoir_mask(0.1, group, CLIP, np.random.default_rng(7), 0.5)
    fill = _clip_set(mask & ~res, starts)
    clip_score = {int(s): float(score[s : s + CLIP].max()) for s in starts}
    res_clips = _clip_set(res, starts)
    others = [s for s in clip_score if s not in res_clips and s not in fill]
    assert len(fill) > 0
    assert min(clip_score[s] for s in fill) >= max(clip_score[s] for s in others)


def test_entropy_term_and_lam() -> None:
    group = np.repeat(np.arange(4), 50)
    n = len(group)
    starts = clip_starts(group, CLIP)
    score = np.zeros(n)
    entropy = np.zeros(n)
    entropy[starts[7] : starts[7] + CLIP] = 1.0  # 엔트로피만 높은 클립
    m0 = select_shared(score, 0.05, group, CLIP, np.random.default_rng(1), 0.0, entropy=entropy, lam=0.0)
    m1 = select_shared(score, 0.05, group, CLIP, np.random.default_rng(1), 0.0, entropy=entropy, lam=1.0)
    assert m1[starts[7]] and m1.sum() == m0.sum() == CLIP
    with pytest.raises(ValueError):
        select_shared(score, 0.05, group, CLIP, np.random.default_rng(1), 0.0, lam=0.5)  # entropy 누락


@pytest.mark.parametrize("budget", [0.02, 0.1, 0.3])
@pytest.mark.parametrize("reservoir", [0.0, 0.5, 0.95, 1.0])
def test_selected_amount_equal_and_group_boundaries(budget: float, reservoir: float) -> None:
    group = _group()
    n = len(group)
    starts = clip_starts(group, CLIP)
    start_set = set(starts.tolist())
    k = max(1, int(round(budget * n / CLIP)))
    rng0 = np.random.default_rng(9)
    for sc in (None, rng0.random(n), np.zeros(n), -np.arange(n, dtype=float)):
        mask = select_shared(sc, budget, group, CLIP, np.random.default_rng(3), reservoir)
        assert mask.dtype == bool and mask.shape == (n,)
        assert int(mask.sum()) == k * CLIP
        clips = [s for s in starts if mask[s]]
        assert len(clips) == k
        sel = np.flatnonzero(mask)
        run_starts = sel[np.concatenate([[True], np.diff(sel) > 1])]
        for rs in run_starts:
            assert int(rs) in start_set
        for s in clips:
            assert mask[s : s + CLIP].all()
            # 클립이 같은 연속 구간 안에 있다(떨어진 같은 id 구간을 잇지 않는다).
            assert len(np.unique(group[s : s + CLIP])) == 1
            assert not np.any(group[s + 1 : s + CLIP] != group[s : s + CLIP - 1])


def test_none_baseline_is_uniform_random() -> None:
    group = _group()
    n = len(group)
    starts = clip_starts(group, CLIP)
    # 같은 rng 상태의 select("random")과 같은 클립 집합(저장소 비율과 무관).
    ref = select("random", 0.1, group, CLIP, np.random.default_rng(11))
    for rho in (0.0, 0.5, 0.9, 1.0):
        m = select_shared(None, 0.1, group, CLIP, np.random.default_rng(11), rho)
        assert np.array_equal(m, ref), rho
    # 여러 시드에서 선택 빈도가 클립 위치·그룹과 무관하게 균일하다(K/|C|에 가깝다).
    counts = np.zeros(len(starts))
    n_rep = 600
    for seed in range(n_rep):
        m = select_shared(None, 0.1, group, CLIP, np.random.default_rng(seed), 0.9)
        counts += m[starts]
    k = max(1, int(round(0.1 * n / CLIP)))
    freq = counts / n_rep
    expect = k / len(starts)
    assert abs(freq.mean() - expect) < 1e-9
    sd = np.sqrt(expect * (1 - expect) / n_rep)
    assert np.all(np.abs(freq - expect) < 5 * sd)
    # 앞쪽 절반·뒤쪽 절반 클립의 평균 선택률 차이가 작다(위치 편향 없음).
    half = len(starts) // 2
    assert abs(freq[:half].mean() - freq[half:].mean()) < 0.03


def test_input_validation() -> None:
    group = _group()
    n = len(group)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        select_shared(np.zeros(n - 1), 0.1, group, CLIP, rng, 0.5)
    with pytest.raises(ValueError):
        select_shared(np.zeros((n, 2)), 0.1, group, CLIP, rng, 0.5)
    with pytest.raises(ValueError):
        select_shared(None, 0.0, group, CLIP, rng, 0.5)
    with pytest.raises(ValueError):
        select_shared(None, 0.1, group, CLIP, rng, 1.5)
    # 유효 클립이 없으면 빈 마스크.
    tiny = np.arange(30)  # 모든 그룹 길이 1
    assert select_shared(None, 0.5, tiny, CLIP, rng, 0.5).sum() == 0
