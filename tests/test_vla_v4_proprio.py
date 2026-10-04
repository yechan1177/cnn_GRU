"""v4 P5 자차 운동 이력(docs/36 2.0b) 테스트.

- 관측 "proprio" [B,Hp] 생성 규칙(특징 이력과 같은 잘라 냄, feature_group 기준)
- 모델 전처리(속도 + 차분)와 형태 검사, Hp=1 동일성
- 폐루프가 정책 함수 속성대로 [B,Hp] 이력을 넘기는지
- 합성 문제: 행동이 속도 차분에 비례하면 Hp=8이 Hp=1보다 확실히 낫다
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from vcp.vla.closed_loop import make_test_specs, run_closed_loop
from vcp.vla.obs import proprio as obs_proprio
from vcp.vla.policy import ArrayImageSource, PolicyConfig, PolicyData, VLALitePolicy, train_policy, predict_open_loop


def _data(n: int = 40, group: np.ndarray | None = None, feature_group: np.ndarray | None = None, ego_v: np.ndarray | None = None) -> PolicyData:
    return PolicyData(
        images=ArrayImageSource(np.zeros((n, 64, 64, 3), np.uint8)),
        group=np.zeros(n, int) if group is None else group,
        ego_v=np.arange(n, dtype=np.float32) if ego_v is None else ego_v,
        action=np.zeros(n, np.float32),
        tokens=np.ones((n, 4), np.int64),
        domain="driving",
        feature_group=feature_group,
    )


def test_observation_proprio_history_shape_and_truncation() -> None:
    n = 40
    group = np.repeat(np.arange(4), 10)  # 클립 4개(10프레임)
    d = _data(n, group=group)
    obs1 = d.observation(np.array([12]))
    assert obs1["proprio"].shape == (1, 1)
    obs = d.observation(np.array([12, 25]), proprio_history=4, proprio_stride=2)
    assert obs["proprio"].shape == (2, 4)
    # 클립 기준: 12 → [12, 10, 10, 10], 25 → [25, 23, 21, 20]
    expect = obs_proprio(np.array([[12, 10, 10, 10], [25, 23, 21, 20]], np.float32), "driving")[..., 0]
    np.testing.assert_allclose(obs["proprio"], expect)
    # feature_group(에피소드 하나)이 있으면 클립 경계를 넘어 이력을 쓴다(프리롤)
    d2 = _data(n, group=group, feature_group=np.zeros(n, int))
    o2 = d2.observation(np.array([12]), proprio_history=4, proprio_stride=2)
    np.testing.assert_allclose(o2["proprio"], obs_proprio(np.array([[12, 10, 8, 6]], np.float32), "driving")[..., 0])


def test_prep_proprio_identity_and_differences() -> None:
    m1 = VLALitePolicy(vocab_size=10)
    x = torch.randn(3, 1)
    assert torch.equal(m1.prep_proprio(x), x)
    m8 = VLALitePolicy(vocab_size=10, proprio_dim=8)
    v = torch.arange(8, dtype=torch.float32).flip(0)[None]  # 현재 7, 이전 6, … → 차분 1
    out = m8.prep_proprio(v)
    assert out.shape == (1, 8)
    assert out[0, 0] == 7 and torch.allclose(out[0, 1:], torch.full((7,), 10.0))
    with pytest.raises(ValueError):
        m8.prep_proprio(torch.zeros(2, 1))


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        PolicyConfig(proprio_history=0)
    cfg = PolicyConfig(proprio_history=8, proprio_stride=2)
    m = VLALitePolicy.from_config(cfg, vocab_size=10)
    assert m.proprio_history == 8 and m.proprio_stride == 2


def test_closed_loop_passes_proprio_history() -> None:
    seen: list[np.ndarray] = []
    speeds: list[np.ndarray] = []

    def fn(obs: dict[str, Any]) -> np.ndarray:
        seen.append(np.array(obs["proprio"]))
        return np.zeros(len(obs["tokens"]), np.float32)

    fn.proprio_history = 4  # type: ignore[attr-defined]
    fn.proprio_stride = 2  # type: ignore[attr-defined]
    specs = make_test_specs("driving", 1, 123000)[:2]
    res = run_closed_loop(fn, specs, domain="driving", duration_s=2.0)
    assert res["config"]["proprio_history"] == 4
    assert all(s.shape == (2, 4) for s in seen)
    cur = np.stack([s[:, 0] for s in seen])  # [T,B] 현재 속도
    for t in range(len(seen)):
        for k in range(4):
            np.testing.assert_allclose(seen[t][:, k], cur[max(t - 2 * k, 0)])


def test_synthetic_speed_difference_needs_history() -> None:
    """행동 = 속도 변화율(가속)에 비례. 현재 속도만으로는 알 수 없다."""

    rng = np.random.default_rng(0)
    n_ep, L = 60, 40
    v, a = [], []
    for _ in range(n_ep):
        acc = rng.uniform(-1.0, 1.0)
        vv = 10.0 + rng.uniform(-3, 3) + acc * np.arange(L) * 0.1
        v.append(vv)
        a.append(np.full(L, acc))
    ego_v, act = np.concatenate(v).astype(np.float32), np.concatenate(a).astype(np.float32)
    grp = np.repeat(np.arange(n_ep), L)
    data = PolicyData(images=ArrayImageSource(np.zeros((len(grp), 64, 64, 3), np.uint8)), group=grp, ego_v=ego_v, action=act, tokens=np.ones((len(grp), 3), np.int64), domain="driving")
    tr = np.where((grp < 45) & (np.arange(len(grp)) % L >= 14))[0]
    te = np.where((grp >= 45) & (np.arange(len(grp)) % L >= 14))[0]
    maes = {}
    for hp in (1, 8):
        cfg = PolicyConfig(steps=300, batch=64, seed=0, augment=False, threads=1, chunk=1, proprio_history=hp, proprio_stride=2)
        model, _ = train_policy(data, tr, cfg)
        pred = predict_open_loop(model, data, te)[:, 0]
        maes[hp] = float(np.abs(pred - act[te]).mean())
    base = float(np.abs(act[te] - act[tr].mean()).mean())
    assert maes[8] < 0.5 * base
    assert maes[8] < 0.6 * maes[1]
