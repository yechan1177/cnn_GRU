"""VLA-lite v3 검출 특징 토큰(`PolicyConfig.use_features`, docs/33 P1) 테스트."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vcp.vla.policy import (
    ArrayImageSource,
    PolicyConfig,
    PolicyData,
    VLALitePolicy,
    count_parameters,
    make_policy_fn,
    predict_open_loop,
    train_policy,
)

H = W = 64
D = 32


def _data(
    group: np.ndarray,
    action: np.ndarray | None = None,
    features: np.ndarray | None = None,
    frames: np.ndarray | None = None,
) -> PolicyData:
    n = len(group)
    rng = np.random.default_rng(0)
    if frames is None:
        frames = np.broadcast_to((np.arange(n) % 256).astype(np.uint8)[:, None, None, None], (n, H, W, 3)).copy()
    return PolicyData(
        images=ArrayImageSource(frames),
        group=group,
        ego_v=rng.uniform(0, 30, n).astype(np.float32),
        action=action if action is not None else np.arange(n, dtype=np.float32) * 0.01,
        tokens=rng.integers(2, 20, size=(n, 12)).astype(np.int64),
        domain="driving",
        features=features,
    )


def _small_cfg(**kw: object) -> PolicyConfig:
    base = dict(steps=5, batch=8, augment=False, threads=1, seed=0)
    base.update(kw)
    return PolicyConfig(**base)  # type: ignore[arg-type]


def _index_features(n: int) -> np.ndarray:
    """특징 행 i의 모든 값이 i인 가짜 특징(경계 규칙 확인용)."""

    return np.repeat(np.arange(n, dtype=np.float32)[:, None], D, axis=1)


def test_feature_input_shapes() -> None:
    model = VLALitePolicy(vocab_size=30, chunk=8, use_features=True, feature_dim=D)
    out = model(torch.rand(5, 6, H, W), torch.randint(0, 30, (5, 12)), torch.rand(5, 1), torch.randn(5, 2, D))
    assert out.shape == (5, 8)
    with pytest.raises(ValueError):
        model(torch.rand(2, 6, H, W), torch.randint(0, 30, (2, 12)), torch.rand(2, 1))  # features 누락
    with pytest.raises(ValueError):
        model(torch.rand(2, 6, H, W), torch.randint(0, 30, (2, 12)), torch.rand(2, 1), torch.randn(2, 2, D + 1))

    group = np.array([7, 7, 7, 7, 3, 3, 3, 3])
    data = _data(group, features=_index_features(8))
    obs = data.observation(np.arange(8))
    assert obs["features"].shape == (8, 2, D) and obs["features"].dtype == np.float32
    assert obs["features"][:, 0, 0].tolist() == list(range(8))                 # 프레임 t
    assert obs["features"][:, 1, 0].tolist() == [0, 0, 0, 1, 4, 4, 4, 5]       # t−2(영상과 같은 경계 규칙)
    assert obs["features"][:, 1, 0].tolist() == obs["image"][:, 0, 0, 3].tolist()
    # features가 없으면 키도 없다(기존 관측 형식 유지).
    assert "features" not in _data(group).observation(np.arange(8))
    with pytest.raises(ValueError):
        _data(group, features=np.zeros((7, D), np.float32))


def test_outlier_features_are_clamped() -> None:
    torch.manual_seed(0)
    model = VLALitePolicy(vocab_size=30, use_features=True, feature_dim=D).eval()
    img, tok, prop = torch.rand(3, 6, H, W), torch.randint(0, 30, (3, 12)), torch.rand(3, 1)
    big = torch.full((3, 2, D), 100.0)
    nan = torch.full((3, 2, D), float("nan"))
    with torch.no_grad():
        assert torch.equal(model(img, tok, prop, big), model(img, tok, prop, torch.full((3, 2, D), 3.0)))
        assert torch.equal(model(img, tok, prop, nan), model(img, tok, prop, torch.zeros(3, 2, D)))


def test_use_features_false_is_identical_to_v2() -> None:
    torch.manual_seed(0)
    ref = VLALitePolicy(vocab_size=30)
    torch.manual_seed(0)
    off = VLALitePolicy(vocab_size=30, use_features=False, feature_dim=D, feature_hidden=64)
    torch.manual_seed(0)
    on = VLALitePolicy(vocab_size=30, use_features=True, feature_dim=D, feature_hidden=64)
    assert count_parameters(ref) == count_parameters(off)
    s_ref, s_off = ref.state_dict(), off.state_dict()
    assert s_ref.keys() == s_off.keys() and all(torch.equal(s_ref[k], s_off[k]) for k in s_ref)
    assert not hasattr(off, "feat_mlp")
    # 특징 모듈 증가분: 헤드 첫 층 64×256 + MLP(64→64→64).
    assert count_parameters(on) - count_parameters(ref) == 64 * 256 + (2 * D * 64 + 64) + (64 * 64 + 64)
    # 특징 모듈을 마지막에 만들므로 공통 모듈(헤드 첫 층 제외)의 초기 가중치가 같다.
    s_on = on.state_dict()
    for k in s_ref:
        if not k.startswith("head.0.") and not k.startswith("head.2.") and not k.startswith("head.4."):
            assert torch.equal(s_ref[k], s_on[k]), k
    # False 모델은 features 입력을 무시한다.
    ref.eval()
    img, tok, prop = torch.rand(4, 6, H, W), torch.randint(0, 30, (4, 12)), torch.rand(4, 1)
    with torch.no_grad():
        assert torch.equal(ref(img, tok, prop), ref(img, tok, prop, torch.randn(4, 2, D)))

    # 학습: use_features=False이면 data.features 유무와 무관하게 v2와 같은 결과(증강 포함).
    group = np.repeat(np.arange(4), 12)
    n = len(group)
    feats = np.random.default_rng(3).normal(size=(n, D)).astype(np.float32)
    m_plain, i_plain = train_policy(_data(group), np.arange(n), _small_cfg(augment=True))
    m_feat, i_feat = train_policy(_data(group, features=feats), np.arange(n), _small_cfg(augment=True))
    sp, sf = m_plain.state_dict(), m_feat.state_dict()
    assert all(torch.equal(sp[k], sf[k]) for k in sp)
    assert i_plain["n_params"] == i_feat["n_params"]


def test_default_param_count_unchanged() -> None:
    torch.manual_seed(0)
    assert count_parameters(VLALitePolicy(vocab_size=64)) == 517_736  # docs/31 7절 값
    assert count_parameters(VLALitePolicy(vocab_size=89)) == 519_336
    assert count_parameters(VLALitePolicy(vocab_size=89, use_features=True)) == 519_336 + 24_704


def test_learns_feature_only_problem() -> None:
    """영상은 상수(정보 없음), 특징 한 차원 → 가속도인 문제를 특징 모델만 학습한다."""

    n_groups, glen = 60, 10
    rng = np.random.default_rng(2)
    u = rng.uniform(-1, 1, n_groups)
    group = np.repeat(np.arange(n_groups), glen)
    n = len(group)
    feats = rng.normal(0, 0.3, size=(n, D)).astype(np.float32)  # 방해 차원
    feats[:, 5] = u[group]
    action = (u[group] * 4.0).astype(np.float32)  # 정규화 단위 [−1, 1]
    frames = np.full((n, H, W, 3), 90, dtype=np.uint8)
    data = _data(group, action=action, features=feats, frames=frames)
    idx = np.arange(n)
    baseline = float(np.abs(action - action.mean()).mean())

    cfg_on = PolicyConfig(steps=200, batch=16, augment=True, threads=1, seed=0, use_features=True)
    model_on, info = train_policy(data, idx, cfg_on)
    mae_on = float(np.abs(predict_open_loop(model_on, data, idx)[:, 0] - action).mean())
    assert mae_on < 0.25 * baseline, (mae_on, baseline)
    assert info["loss_curve"][-1] < info["loss_curve"][0]
    assert info["config"]["use_features"] is True

    cfg_off = PolicyConfig(steps=200, batch=16, augment=True, threads=1, seed=0, use_features=False)
    model_off, _ = train_policy(data, idx, cfg_off)
    mae_off = float(np.abs(predict_open_loop(model_off, data, idx)[:, 0] - action).mean())
    assert mae_off > 0.7 * baseline, (mae_off, baseline)


def test_feature_training_is_deterministic() -> None:
    group = np.repeat(np.arange(4), 12)
    n = len(group)
    data = _data(group, features=np.random.default_rng(4).normal(size=(n, D)).astype(np.float32))
    idx = np.arange(n)
    cfg = _small_cfg(augment=True, use_features=True)
    m1, _ = train_policy(data, idx, cfg)
    m2, _ = train_policy(data, idx, cfg)
    m3, _ = train_policy(data, idx, _small_cfg(augment=True, use_features=True, seed=1))
    m4, _ = train_policy(data, idx, _small_cfg(augment=True, use_features=True, feature_noise=0.0))
    s1, s2, s3, s4 = m1.state_dict(), m2.state_dict(), m3.state_dict(), m4.state_dict()
    assert all(torch.equal(s1[k], s2[k]) for k in s1)
    assert not all(torch.equal(s1[k], s3[k]) for k in s1)
    assert not all(torch.equal(s1[k], s4[k]) for k in s1)  # 특징 노이즈가 실제로 적용된다


def test_make_policy_fn_and_errors() -> None:
    group = np.repeat(np.arange(3), 10)
    n = len(group)
    feats = np.random.default_rng(5).normal(size=(n, D)).astype(np.float32)
    data = _data(group, features=feats)
    model, _ = train_policy(data, np.arange(n), _small_cfg(use_features=True))
    fn = make_policy_fn(model, "driving")
    idx = np.array([0, 4, 15, 29])
    obs = data.observation(idx)
    out = fn(obs)
    assert out.shape == (4,) and out.dtype == np.float32
    np.testing.assert_allclose(out, predict_open_loop(model, data, idx)[:, 0], rtol=1e-5, atol=1e-6)

    no_feat = {k: v for k, v in obs.items() if k != "features"}
    with pytest.raises(ValueError, match="features"):
        fn(no_feat)
    with pytest.raises(ValueError, match="features"):
        predict_open_loop(model, _data(group), idx)
    with pytest.raises(ValueError, match="features"):
        train_policy(_data(group), np.arange(n), _small_cfg(use_features=True))
    with pytest.raises(ValueError, match="feature_dim"):
        train_policy(_data(group, features=feats[:, :16]), np.arange(n), _small_cfg(use_features=True))

    # 특징을 쓰지 않는 모델은 "features" 키가 있어도 무시한다.
    m_off, _ = train_policy(_data(group), np.arange(n), _small_cfg())
    fn_off = make_policy_fn(m_off, "driving")
    np.testing.assert_array_equal(fn_off(obs), fn_off(no_feat))
