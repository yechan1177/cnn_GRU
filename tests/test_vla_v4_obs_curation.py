"""A15 v4 관측·선별 테스트: 특징 이력 일반화(FeatureHistory H·s, 폐루프 전달)와 S1 점수 몫 에피소드 상한.

계약: docs/36 6절, docs/28a "v4 추가 계약(A15)", docs/29 10절, docs/30 7절.
폐루프 테스트는 CPU 부담을 줄이려고 작은 사양(3 에피소드 × 2초)만 쓴다.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vcp.sim.env import SimEnv
from vcp.vla.closed_loop import make_test_specs, resolve_feature_history, run_closed_loop
from vcp.vla.curation import clip_starts, select_shared, shared_reservoir_mask
from vcp.vla.instructions import styles_for_domain
from vcp.vla.obs import FEATURE_DIM, FeatureHistory, OnlineFeatureTracker, feature_history_offsets, stack_feature_history
from vcp.vla.pool import PoolConfig, generate_pool, load_pool, style_for_seed

CLIP = 10


def _run(env: SimEnv, cmd: float | None = None) -> list:
    infos = [env.reset()]
    while not env.done:
        infos.append(env.step(cmd))
    return infos


def _v3_stack(feats: np.ndarray, t: int) -> np.ndarray:
    """v3 규칙을 직접 적은 기대값: (feats[t], feats[max(t−2, 0)])."""

    return np.stack([feats[t], feats[max(t - 2, 0)]]).astype(np.float32)


def _train_side_history(x_all: np.ndarray, ep: np.ndarray, gidx: int, history: int, stride: int) -> np.ndarray:
    """학습 쪽 규칙(PolicyData.observation(idx, history, stride), A14)을 풀 전체 배열 인덱스로 직접 구현한 기대값.

    인덱스 k = max(g − k·s, 에피소드 첫 프레임). 풀은 에피소드가 연속 구간으로 저장된다.
    """

    first = int(np.flatnonzero(ep == ep[gidx])[0])
    rows = [max(gidx - k * stride, first) for k in range(history)]
    return x_all[rows].astype(np.float32)


# ---------------------------------------------------------------------------
# 특징 이력 일반화
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def small_pool(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """2 에피소드 × 3초 풀과, 같은 사양으로 재현한 에피소드별 검출 프레임."""

    root = tmp_path_factory.mktemp("pool")
    cfg = PoolConfig(n_episodes=2, seed_base=100000, duration_s=3.0, workers=1)
    generate_pool(root / "pool", cfg)
    data, meta = load_pool(root / "pool")
    frames_per_ep: list[list] = []
    for ep_meta in meta["episodes"]:
        style, _ = style_for_seed(ep_meta["seed"], "driving")
        env = SimEnv(ep_meta["scenario"], ep_meta["seed"], duration_s=cfg.duration_s, style=style)
        frames_per_ep.append([i.frame for i in _run(env, None)])
    return {"data": data, "frames": frames_per_ep}


def test_offsets_and_stack_shapes() -> None:
    assert feature_history_offsets(8, 2).tolist() == [0, 2, 4, 6, 8, 10, 12, 14]
    assert feature_history_offsets().tolist() == [0, 2]
    feats = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)
    one = stack_feature_history(feats, 5, history=8, stride=2)
    assert one.shape == (8, 3) and one.dtype == np.float32
    # 프레임 5, 3, 1, 그다음은 에피소드 시작(0)에서 잘라 낸다
    assert [int(r[0] // 3) for r in one] == [5, 3, 1, 0, 0, 0, 0, 0]
    many = stack_feature_history(feats, np.arange(20), history=4, stride=3)
    assert many.shape == (20, 4, 3)
    for t in range(20):
        assert np.array_equal(many[t], stack_feature_history(feats, t, 4, 3))
    # 기본값은 v3 규칙과 같다
    for t in range(20):
        assert np.array_equal(stack_feature_history(feats, t), _v3_stack(feats, t))
    with pytest.raises(ValueError):
        stack_feature_history(feats, 3, history=0)
    with pytest.raises(ValueError):
        FeatureHistory(2, history=2, stride=0)


def test_feature_history_h8_matches_pool_and_train_rule(small_pool: dict[str, Any]) -> None:
    """FeatureHistory(H=8, s=2)의 형태·초기 프레임 잘라 냄, 풀 X_v1v2로 만든 학습 쪽 [H] 이력과의 동일성."""

    data, frames_per_ep = small_pool["data"], small_pool["frames"]
    h, s = 8, 2
    hist = FeatureHistory(len(frames_per_ep), history=h, stride=s)
    assert hist._slots == (h - 1) * s + 1
    x_all, ep = data["X_v1v2"], data["ep"]
    pool_feats = [x_all[ep == k] for k in range(len(frames_per_ep))]
    offsets = [int(np.flatnonzero(ep == k)[0]) for k in range(len(frames_per_ep))]
    prev_obs: np.ndarray | None = None
    prev_snapshot = np.zeros(0)
    for t in range(len(frames_per_ep[0])):
        obs = hist.update([frames[t] for frames in frames_per_ep])
        assert obs.shape == (2, h, FEATURE_DIM) and obs.dtype == np.float32 and obs.flags.c_contiguous
        # 에피소드 안 규칙
        expected = np.stack([stack_feature_history(f, t, h, s) for f in pool_feats])
        assert np.array_equal(obs, expected), t
        # 학습 쪽 규칙(풀 전체 인덱스, 에피소드 시작에서 잘라 냄)
        for i in range(len(frames_per_ep)):
            assert np.array_equal(obs[i], _train_side_history(x_all, ep, offsets[i] + t, h, s)), (t, i)
        # 초기 프레임: t − k·s < 0인 칸은 모두 프레임 0
        frame0 = np.stack([f[0] for f in pool_feats])
        for k in range(t // s + 1, h):
            assert np.array_equal(obs[:, k], frame0), (t, k)
        # 이전 반환값을 링 버퍼가 덮어쓰지 않는다(복사본 반환)
        if prev_obs is not None:
            assert np.array_equal(prev_obs, prev_snapshot)
        prev_obs, prev_snapshot = obs, obs.copy()


def test_feature_history_default_equals_v3(small_pool: dict[str, Any]) -> None:
    """기본값(H=2, s=2)은 v3의 [t, max(t−2,0)] 배열과 같다(명시 2/2와도 같다)."""

    data, frames_per_ep = small_pool["data"], small_pool["frames"]
    a = FeatureHistory(len(frames_per_ep))
    b = FeatureHistory(len(frames_per_ep), history=2, stride=2)
    assert a._slots == 3
    pool_feats = [data["X_v1v2"][data["ep"] == k] for k in range(len(frames_per_ep))]
    for t in range(len(frames_per_ep[0])):
        frames = [f[t] for f in frames_per_ep]
        oa, ob = a.update(frames), b.update(frames)
        assert oa.shape == (2, 2, FEATURE_DIM)
        assert np.array_equal(oa, ob)
        assert np.array_equal(oa, np.stack([_v3_stack(f, t) for f in pool_feats])), t


# ---------------------------------------------------------------------------
# 폐루프 전달
# ---------------------------------------------------------------------------
def _specs() -> list:
    return make_test_specs("driving", 1, 850000, counterfactual=True)[:3]


def _replay_features(spec: Any, duration_s: float) -> np.ndarray:
    """같은 명령(0)으로 SimEnv를 재현해 에피소드 특징 [T,32]를 만든다."""

    style = styles_for_domain("driving")[spec.style]
    env = SimEnv(spec.scenario, spec.seed, duration_s=duration_s, style=style)
    tracker = OnlineFeatureTracker()
    return np.stack([tracker.update_vector(info.frame) for info in _run(env, 0.0)])


def test_resolve_feature_history() -> None:
    def plain(obs: dict[str, np.ndarray]) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)

    def tagged(obs: dict[str, np.ndarray]) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)

    tagged.feature_history = 8  # type: ignore[attr-defined]
    tagged.feature_stride = 3  # type: ignore[attr-defined]
    assert resolve_feature_history(None) == (2, 2)
    assert resolve_feature_history(plain) == (2, 2)
    assert resolve_feature_history(tagged) == (8, 3)
    assert resolve_feature_history(tagged, 4, None) == (4, 3)  # 명시 인자가 우선, 나머지는 속성
    assert resolve_feature_history(plain, None, 5) == (2, 5)
    with pytest.raises(ValueError):
        resolve_feature_history(plain, 0, None)


def test_closed_loop_feature_history_attr_h8() -> None:
    """policy_fn 속성 feature_history=8로 obs["features"]가 [B,8,32]이고, 재현한 특징 이력과 같다."""

    specs = _specs()
    duration = 2.0
    seen: list[np.ndarray] = []

    def zero_policy(obs: dict[str, np.ndarray]) -> np.ndarray:
        seen.append(obs["features"].copy())
        return np.zeros(len(obs["proprio"]), dtype=np.float32)

    zero_policy.feature_history = 8  # type: ignore[attr-defined]
    zero_policy.feature_stride = 2  # type: ignore[attr-defined]
    res = run_closed_loop(zero_policy, specs, duration_s=duration)
    assert res["config"]["feature_history"] == 8 and res["config"]["feature_stride"] == 2
    assert len(seen) == 29 and seen[0].shape == (3, 8, FEATURE_DIM) and seen[0].dtype == np.float32
    for i, spec in enumerate(specs):
        feats = _replay_features(spec, duration)
        for t, obs in enumerate(seen):
            assert np.array_equal(obs[i], stack_feature_history(feats, t, 8, 2)), (i, t)
            # 앞 두 칸은 v3 [t, t−2]와 같다
            assert np.array_equal(obs[i][:2], _v3_stack(feats, t))
    # 명시 인자가 속성보다 우선한다
    seen.clear()
    res2 = run_closed_loop(zero_policy, specs[:1], duration_s=1.0, feature_history=3, feature_stride=1)
    assert seen[0].shape == (1, 3, FEATURE_DIM)
    assert res2["config"]["feature_history"] == 3 and res2["config"]["feature_stride"] == 1


def test_closed_loop_default_equals_v3() -> None:
    """속성·인자가 없으면 2/2이고, 특징이 v3 규칙과 같으며 특징을 쓰는 정책의 결과도 명시 2/2와 같다."""

    specs = _specs()
    duration = 2.0
    seen: list[np.ndarray] = []

    def feat_policy(obs: dict[str, np.ndarray]) -> np.ndarray:
        # 특징에 의존하는 결정적 명령(특징 경로가 달라지면 궤적이 달라진다)
        seen.append(obs["features"].copy())
        f = obs["features"]
        return np.tanh(f[:, 0, :4].sum(1) - f[:, 1, :4].sum(1)).astype(np.float32)

    res = run_closed_loop(feat_policy, specs, duration_s=duration)
    first_seen = list(seen)
    seen.clear()
    res_explicit = run_closed_loop(feat_policy, specs, duration_s=duration, feature_history=2, feature_stride=2)
    assert res["config"]["feature_history"] == 2 and res["config"]["feature_stride"] == 2
    assert res["episodes"] == res_explicit["episodes"] and res["overall"] == res_explicit["overall"]
    assert all(np.array_equal(a, b) for a, b in zip(first_seen, seen))
    assert first_seen[0].shape == (3, 2, FEATURE_DIM)
    # 0 명령 정책으로 v3 규칙을 직접 대조(재현 경로가 같은 명령이어야 하므로)
    zero_seen: list[np.ndarray] = []

    def zero_policy(obs: dict[str, np.ndarray]) -> np.ndarray:
        zero_seen.append(obs["features"].copy())
        return np.zeros(len(obs["proprio"]), dtype=np.float32)

    run_closed_loop(zero_policy, specs[:2], duration_s=duration)
    for i, spec in enumerate(specs[:2]):
        feats = _replay_features(spec, duration)
        for t, obs in enumerate(zero_seen):
            assert np.array_equal(obs[i], _v3_stack(feats, t))
    # 전문가 참조도 config에 기본값을 기록한다
    expert = run_closed_loop(None, specs[:1], duration_s=1.0)
    assert expert["config"]["feature_history"] == 2 and expert["timing"]["features_s"] == 0.0


# ---------------------------------------------------------------------------
# S1 점수 몫 에피소드 상한
# ---------------------------------------------------------------------------
def _many_groups(n_groups: int = 40, length: int = 100) -> np.ndarray:
    return np.repeat(np.arange(n_groups), length)


def _concentrated_score(group: np.ndarray, hot: tuple[int, ...] = (3, 17)) -> np.ndarray:
    """일부 에피소드에 높은 점수가 몰린 프레임 점수(동률 없음)."""

    rng = np.random.default_rng(123)
    score = rng.random(len(group)) * 0.1
    score[np.isin(group, hot)] += 1.0
    return score


def _clip_groups(mask: np.ndarray, starts: np.ndarray, group: np.ndarray) -> list[int]:
    return [int(group[s]) for s in starts if mask[s]]


@pytest.mark.parametrize("reservoir", [0.5, 0.9])
def test_cap1_distinct_episodes_same_amount_same_reservoir(reservoir: float) -> None:
    group = _many_groups()
    n = len(group)
    starts = clip_starts(group, CLIP)
    score = _concentrated_score(group)
    entropy = np.random.default_rng(4).random(n)
    budget = 0.1
    res = shared_reservoir_mask(budget, group, CLIP, np.random.default_rng(42), reservoir)
    base_info: dict[str, Any] = {}
    base = select_shared(score, budget, group, CLIP, np.random.default_rng(42), reservoir, entropy=entropy, lam=0.5, info=base_info)
    info: dict[str, Any] = {}
    capped = select_shared(
        score, budget, group, CLIP, np.random.default_rng(42), reservoir, entropy=entropy, lam=0.5, per_group_cap=1, info=info
    )
    k = max(1, int(round(budget * n / CLIP)))
    assert capped.sum() == base.sum() == k * CLIP  # 선택량 불변
    assert (capped & res).sum() == res.sum() == (base & res).sum()  # 저장소는 c와 무관
    share = _clip_groups(capped & ~res, starts, group)
    assert len(share) == info["n_score"] == k - int(round(reservoir * k))
    assert len(set(share)) == len(share)  # 점수 몫 클립의 에피소드가 서로 다르다
    assert info["n_cap_overflow"] == 0 and info["n_score_groups"] == len(share) and info["per_group_cap"] == 1
    # v3(None)는 점수가 몰린 에피소드에서 여러 클립을 고른다(상한이 실제로 작동했는지 대조)
    base_share = _clip_groups(base & ~res, starts, group)
    assert len(set(base_share)) < len(base_share)
    assert base_info["n_score_groups"] == len(set(base_share)) and base_info["per_group_cap"] is None
    # 상한 c=2: 에피소드당 최대 2개
    c2 = select_shared(score, budget, group, CLIP, np.random.default_rng(42), reservoir, entropy=entropy, lam=0.5, per_group_cap=2)
    share2 = _clip_groups(c2 & ~res, starts, group)
    assert max(share2.count(g) for g in set(share2)) <= 2 and c2.sum() == k * CLIP


def test_cap1_picks_best_clip_per_episode() -> None:
    """c=1이면 점수 몫은 '각 에피소드의 저장소 밖 최고 점수 클립' 중 점수 상위다(동률 없는 점수)."""

    group = _many_groups(30, 80)
    starts = clip_starts(group, CLIP)
    score = np.random.default_rng(8).random(len(group))
    budget, rho = 0.1, 0.5
    res = shared_reservoir_mask(budget, group, CLIP, np.random.default_rng(3), rho)
    mask = select_shared(score, budget, group, CLIP, np.random.default_rng(3), rho, per_group_cap=1)
    res_set = {int(s) for s in starts if res[s]}
    clip_score = {int(s): float(score[s : s + CLIP].max()) for s in starts if int(s) not in res_set}
    best: dict[int, tuple[float, int]] = {}
    for s, v in clip_score.items():
        g = int(group[s])
        if g not in best or v > best[g][0]:
            best[g] = (v, s)
    n_share = int((mask & ~res).sum()) // CLIP
    expect = {s for _, s in sorted(best.values(), reverse=True)[:n_share]}
    assert {int(s) for s in starts if mask[s] and int(s) not in res_set} == expect


def test_none_equals_v3_and_large_cap() -> None:
    """per_group_cap=None은 인자 생략(v3)과 같고, 충분히 큰 상한도 v3와 같다."""

    group = _many_groups(12, 150)
    n = len(group)
    score = _concentrated_score(group, hot=(5,))
    entropy = np.random.default_rng(1).random(n)
    for seed in range(5):
        v3 = select_shared(score, 0.2, group, CLIP, np.random.default_rng(seed), 0.5, entropy=entropy, lam=0.5)
        none = select_shared(score, 0.2, group, CLIP, np.random.default_rng(seed), 0.5, entropy=entropy, lam=0.5, per_group_cap=None)
        big = select_shared(score, 0.2, group, CLIP, np.random.default_rng(seed), 0.5, entropy=entropy, lam=0.5, per_group_cap=10**6)
        assert np.array_equal(v3, none) and np.array_equal(v3, big)


def test_cap_shortfall_fills_by_score() -> None:
    """에피소드 수가 점수 몫보다 적으면 상한 내 클립을 먼저 넣고 나머지를 상한 무시 점수 순으로 채운다."""

    group = np.repeat(np.arange(3), 200)  # 3 에피소드, 클립 60개
    starts = clip_starts(group, CLIP)
    score = np.random.default_rng(2).random(len(group))
    budget, rho = 0.2, 0.25  # K = 12, R = 3, 점수 몫 9 > 에피소드 3
    info: dict[str, Any] = {}
    mask = select_shared(score, budget, group, CLIP, np.random.default_rng(6), rho, per_group_cap=1, info=info)
    base = select_shared(score, budget, group, CLIP, np.random.default_rng(6), rho)
    res = shared_reservoir_mask(budget, group, CLIP, np.random.default_rng(6), rho)
    assert info["k"] == 12 and info["n_reservoir"] == 3 and info["n_score"] == 9
    assert info["n_cap_overflow"] == 6 and info["n_score_groups"] == 3
    assert mask.sum() == base.sum() == 12 * CLIP and (mask & res).sum() == res.sum()
    # 채움 결과: 각 에피소드 최고 클립 3개 + 나머지 중 점수 상위 6개 = 저장소 밖 점수 상위 9개(동률 없음)
    res_set = {int(s) for s in starts if res[s]}
    clip_score = {int(s): float(score[s : s + CLIP].max()) for s in starts if int(s) not in res_set}
    best_per_group = {}
    for s, v in clip_score.items():
        g = int(group[s])
        if g not in best_per_group or v > clip_score[best_per_group[g]]:
            best_per_group[g] = s
    others = sorted((s for s in clip_score if s not in best_per_group.values()), key=lambda s: -clip_score[s])
    expect = set(best_per_group.values()) | set(others[:6])
    assert {int(s) for s in starts if mask[s] and int(s) not in res_set} == expect


def test_random_baseline_ignores_cap_and_validation() -> None:
    group = _many_groups(10, 100)
    info: dict[str, Any] = {}
    a = select_shared(None, 0.2, group, CLIP, np.random.default_rng(9), 0.5)
    b = select_shared(None, 0.2, group, CLIP, np.random.default_rng(9), 0.5, per_group_cap=1, info=info)
    assert np.array_equal(a, b)
    assert info["per_group_cap"] is None and info["n_cap_overflow"] == 0
    score = np.zeros(len(group))
    for bad in (0, -1, 1.5):
        with pytest.raises(ValueError):
            select_shared(score, 0.2, group, CLIP, np.random.default_rng(0), 0.5, per_group_cap=bad)  # type: ignore[arg-type]
