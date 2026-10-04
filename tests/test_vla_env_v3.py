"""A10 v3 벤치마크 수정 테스트: SimEnv 플래그(B1·B3), 반사실 사양(B2), 폐루프 검출 특징 관측.

기본 플래그 회귀(비트 동일)는 `tests/test_vla_env.py`의 fixture 테스트가 맡는다.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from vcp.sim.env import SimEnv
from vcp.sim.world import ROBOT_SCENARIO_TYPES, SCENARIO_TYPES
from vcp.vla.closed_loop import EpisodeSpec, is_counterfactual, make_test_specs, run_closed_loop
from vcp.vla.instructions import N_PARAPHRASES, STYLE_NAMES, styles_for_domain
from vcp.vla.obs import FEATURE_DIM, FeatureHistory, OnlineFeatureTracker, stack_feature_history
from vcp.vla.pool import PoolConfig, generate_pool, load_pool, style_for_seed


def _run(env: SimEnv, cmd: float | None = None) -> list:
    infos = [env.reset()]
    while not env.done:
        infos.append(env.step(cmd))
    return infos


@pytest.mark.parametrize("domain,scenario", [("driving", "follow"), ("robot", "robot_follow_agent")])
def test_decouple_initial_speed_same_across_styles(domain: str, scenario: str) -> None:
    """decouple=True면 같은 시드의 세 스타일 초기 속도가 같고, 목표 속도는 스타일 목표 속도다."""

    styles = styles_for_domain(domain)
    # 생성 직후(reset 전) ego_v가 초기 속도다(_reset_state가 생성자에서 실행됨)
    envs = {n: SimEnv(scenario, seed=11, duration_s=2.0, style=styles[n], decouple_initial_speed=True) for n in STYLE_NAMES}
    assert len({env.ego_v for env in envs.values()}) == 1
    for env in envs.values():
        assert env.v_target == env.v0
        assert 0.85 * env.v0_scenario - 1e-9 <= env.ego_v <= env.v0_scenario + 1e-9
    # reset 후 프레임 0의 속도도 같다(프레임 0 구간은 반응 지연 큐가 0이라 명령이 사실상 0)
    assert len({env.reset().ego_v for env in envs.values()}) == 1
    # 기존 동작(False)은 스타일 목표 속도에 비례하므로 cautious(v_factor 0.85)와 normal이 다르다
    old = {n: SimEnv(scenario, seed=11, duration_s=2.0, style=styles[n]).ego_v for n in STYLE_NAMES}
    assert old["cautious"] != old["normal"]


def test_decouple_keeps_rng_sequence() -> None:
    """decouple 여부와 무관하게 난수 소비 순서가 같아 주변 객체·검출 노이즈 흐름이 같다."""

    style = styles_for_domain("driving")["cautious"]
    a = SimEnv("vru_crossing", seed=7, duration_s=3.0, style=style)
    b = SimEnv("vru_crossing", seed=7, duration_s=3.0, style=style, decouple_initial_speed=True)
    assert a.rng.getstate() == b.rng.getstate()
    assert [x.x for x in a.actors] == [x.x for x in b.actors]
    # 스타일이 없으면 decouple은 효과가 없다
    c = SimEnv("vru_crossing", seed=7, duration_s=3.0)
    d = SimEnv("vru_crossing", seed=7, duration_s=3.0, decouple_initial_speed=True)
    assert c.ego_v == d.ego_v
    assert d.summary()["decouple_initial_speed"] is True and "decouple_initial_speed" not in c.summary()


def test_collision_pushback_false_monotone_ego_x() -> None:
    """밀어내기를 끄면 ego_x가 단조 비감소다. 충돌 집계는 유지된다."""

    hit_any = False
    for scenario, seed in (("robot_crowded", 500066), ("robot_human_headon", 3), ("lead_brake", 21)):
        env = SimEnv(scenario, seed=seed, duration_s=12.0, collision_pushback=False)
        infos = _run(env, 1.0)  # 계속 가속해 충돌을 유도
        xs = np.asarray([i.ego_x for i in infos])
        assert np.all(np.diff(xs) >= 0.0), scenario
        hit_any = hit_any or env.collisions > 0
        assert env.collisions == 0 or any(i.collided for i in infos)
        assert env.summary()["collision_pushback"] is False
    assert hit_any  # 적어도 한 경우는 실제 충돌이 있어야 검사가 의미 있다
    # 대조: 기존 동작(True)에서는 같은 조건에서 ego_x가 뒤로 밀린다
    old = _run(SimEnv("robot_crowded", seed=500066, duration_s=12.0), 1.0)
    assert np.any(np.diff([i.ego_x for i in old]) < 0.0)


def test_collision_pushback_true_is_default() -> None:
    a = _run(SimEnv("lead_brake", seed=21, duration_s=10.0), 1.0)
    b = _run(SimEnv("lead_brake", seed=21, duration_s=10.0, collision_pushback=True), 1.0)
    assert [i.ego_x for i in a] == [i.ego_x for i in b]


@pytest.mark.parametrize("domain,scenarios", [("driving", SCENARIO_TYPES), ("robot", ROBOT_SCENARIO_TYPES)])
def test_counterfactual_specs(domain: str, scenarios: tuple[str, ...]) -> None:
    n = 3
    specs = make_test_specs(domain, n, 850000, counterfactual=True)
    assert len(specs) == len(scenarios) * n * 3
    assert specs == make_test_specs(domain, n, 850000, counterfactual=True)
    assert Counter(s.style for s in specs) == {name: len(scenarios) * n for name in STYLE_NAMES}
    assert Counter(s.scenario for s in specs) == {sc: n * 3 for sc in scenarios}
    groups: dict[tuple[str, int], list[EpisodeSpec]] = {}
    for s in specs:
        groups.setdefault((s.scenario, s.seed), []).append(s)
    assert len(groups) == len(scenarios) * n
    assert len({seed for _, seed in groups}) == len(groups)  # 시드는 (시나리오, 시드) 묶음마다 고유
    for (_, seed), items in groups.items():
        assert sorted(s.style for s in items) == sorted(STYLE_NAMES)
        assert {s.paraphrase for s in items} == {seed % N_PARAPHRASES}
    assert is_counterfactual(specs)
    assert not is_counterfactual(make_test_specs(domain, n, 850000))
    with pytest.raises(ValueError):
        make_test_specs(domain, 0, 850000, counterfactual=True)


def test_online_tracker_matches_pool_features(tmp_path: Path) -> None:
    """풀 X_v1v2(step(None))와 폐루프 특징 경로(FeatureHistory)를 같은 프레임 시퀀스에 적용한 결과가 같다."""

    cfg = PoolConfig(n_episodes=2, seed_base=100000, duration_s=3.0, workers=1)
    generate_pool(tmp_path / "pool", cfg)
    data, meta = load_pool(tmp_path / "pool")
    frames_per_ep: list[list] = []
    for ep_meta in meta["episodes"]:
        style, _ = style_for_seed(ep_meta["seed"], "driving")
        env = SimEnv(ep_meta["scenario"], ep_meta["seed"], duration_s=cfg.duration_s, style=style)
        frames_per_ep.append([i.frame for i in _run(env, None)])
    # 1) 단일 추적기
    for k, frames in enumerate(frames_per_ep):
        tracker = OnlineFeatureTracker()
        got = np.stack([tracker.update_vector(f) for f in frames])
        expected = data["X_v1v2"][data["ep"] == k]
        assert got.dtype == np.float32 and got.shape == expected.shape == (len(frames), FEATURE_DIM)
        assert np.array_equal(got, expected)
    # 2) 폐루프가 쓰는 lockstep 버퍼: t, max(t−2,0) 규칙까지 포함해 비교
    hist = FeatureHistory(len(frames_per_ep))
    pool_feats = [data["X_v1v2"][data["ep"] == k] for k in range(len(frames_per_ep))]
    for t in range(len(frames_per_ep[0])):
        obs = hist.update([frames[t] for frames in frames_per_ep])
        assert obs.shape == (2, 2, FEATURE_DIM) and obs.dtype == np.float32
        expected = np.stack([stack_feature_history(f, t) for f in pool_feats])
        assert np.array_equal(obs, expected), t
    with pytest.raises(ValueError):
        hist.update([frames_per_ep[0][0]])


def test_closed_loop_features_obs_and_flags() -> None:
    """폐루프 obs["features"] 형태 [B,2,32]와, 같은 명령으로 재현한 SimEnv 프레임의 특징과의 동일성."""

    specs = make_test_specs("driving", 1, 850000, counterfactual=True)[:3]
    seen: list[np.ndarray] = []

    def zero_policy(obs: dict[str, np.ndarray]) -> np.ndarray:
        seen.append(obs["features"].copy())
        return np.zeros(len(obs["proprio"]), dtype=np.float32)

    res = run_closed_loop(zero_policy, specs, duration_s=2.0, collision_pushback=False, decouple_initial_speed=True)
    assert res["counterfactual"] is True
    assert res["config"]["collision_pushback"] is False and res["config"]["decouple_initial_speed"] is True
    assert "features_s" in res["timing"]
    assert len(seen) == 29 and seen[0].shape == (3, 2, FEATURE_DIM) and seen[0].dtype == np.float32
    assert np.array_equal(seen[0][:, 0], seen[0][:, 1])  # t=0이면 과거 프레임도 프레임 0
    styles = styles_for_domain("driving")
    for i, spec in enumerate(specs):
        env = SimEnv(
            spec.scenario,
            spec.seed,
            duration_s=2.0,
            style=styles[spec.style],
            collision_pushback=False,
            decouple_initial_speed=True,
        )
        tracker = OnlineFeatureTracker()
        feats = np.stack([tracker.update_vector(info.frame) for info in _run(env, 0.0)])
        for t, obs in enumerate(seen):
            assert np.array_equal(obs[i], stack_feature_history(feats, t))
    assert {e["style"] for e in res["episodes"]} == set(STYLE_NAMES)
    # 전문가 참조에도 플래그가 적용되고 특징은 계산하지 않는다
    expert = run_closed_loop(None, specs, duration_s=2.0, collision_pushback=False, decouple_initial_speed=True)
    assert expert["timing"]["features_s"] == 0.0 and expert["counterfactual"] is True
    plain = run_closed_loop(None, specs[:2], duration_s=2.0)
    assert plain["counterfactual"] is False and plain["config"]["collision_pushback"] is True
