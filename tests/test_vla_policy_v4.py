"""VLA-lite v4 정책 블록 테스트(docs/36: P2 시간 특징 인코더, P3 위험 맥락 보조 헤드, T1 지시문 드롭아웃).

기본값(H=2, s=2, "mlp", aux_weight=0, lang_dropout=0)에서 v3와 비트 단위로 같은지는 두 가지로 확인한다.
1. 저장소의 v3 커밋(`V3_COMMIT`)에 있는 policy.py를 git에서 꺼내 별도 모듈로 불러와 초기 가중치·학습 결과를 비교한다
   (git이나 해당 커밋이 없으면 건너뛴다).
2. v4 필드를 명시한 기본값 설정과 생략한 설정의 결과가 같은지, 새 모듈이 생기지 않는지 확인한다.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from vcp.vla.policy import (
    ArrayImageSource,
    PolicyConfig,
    PolicyData,
    VLALitePolicy,
    apply_lang_dropout,
    count_parameters,
    make_policy_fn,
    predict_open_loop,
    soft_cross_entropy,
    train_policy,
)

H = W = 64
D = 32
V3_COMMIT = "857a4fe7319b3f8781aa50addb3502a7e6b2b945"  # v4 작업 직전 커밋(docs/36 계획 커밋)
REPO = Path(__file__).resolve().parents[1]


def _data(
    group: np.ndarray,
    action: np.ndarray | None = None,
    features: np.ndarray | None = None,
    frames: np.ndarray | None = None,
    aux_targets: np.ndarray | None = None,
    tokens: np.ndarray | None = None,
    feature_group: np.ndarray | None = None,
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
        tokens=tokens if tokens is not None else rng.integers(2, 20, size=(n, 12)).astype(np.int64),
        domain="driving",
        features=features,
        aux_targets=aux_targets,
        feature_group=feature_group,
    )


def _small_cfg(**kw: object) -> PolicyConfig:
    base = dict(steps=5, batch=8, augment=False, threads=1, seed=0)
    base.update(kw)
    return PolicyConfig(**base)  # type: ignore[arg-type]


def _index_features(n: int) -> np.ndarray:
    """특징 행 i의 모든 값이 i인 가짜 특징(경계 규칙 확인용)."""

    return np.repeat(np.arange(n, dtype=np.float32)[:, None], D, axis=1)


def _uniform_aux(n: int, c: int = 6) -> np.ndarray:
    return np.full((n, c), 1.0 / c, dtype=np.float32)


def _same_state(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def _load_v3_policy(tmp_path: Path) -> ModuleType:
    """v3 커밋의 policy.py를 git에서 꺼내 독립 모듈로 불러온다. 불가능하면 테스트를 건너뛴다."""

    try:
        src = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{V3_COMMIT}:src/vcp/vla/policy.py"],
            capture_output=True, check=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"v3 기준 policy.py를 git에서 읽지 못했습니다: {exc}")
    path = tmp_path / "policy_v3_ref.py"
    path.write_bytes(src)
    spec = importlib.util.spec_from_file_location("policy_v3_ref", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["policy_v3_ref"] = mod  # dataclass가 모듈을 찾을 수 있게 등록
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop("policy_v3_ref", None)
    return mod


# ---------------------------------------------------------------------------
# 기본값 비트 동일성
# ---------------------------------------------------------------------------
def test_defaults_bit_identical_to_v3_reference(tmp_path: Path) -> None:
    """기본값의 초기 가중치·고정 시드 학습 결과(final_loss, 가중치, 예측)가 v3 커밋 구현과 비트 단위로 같다."""

    v3 = _load_v3_policy(tmp_path)
    for kw in ({}, {"use_features": True}, {"use_language": False}, {"use_features": True, "use_language": False}):
        torch.manual_seed(0)
        new = VLALitePolicy(vocab_size=89, **kw)
        torch.manual_seed(0)
        ref = v3.VLALitePolicy(vocab_size=89, **kw)
        assert count_parameters(new) == v3.count_parameters(ref), kw
        assert _same_state(new, ref), kw

    group = np.repeat(np.arange(4), 15)
    n = len(group)
    rng = np.random.default_rng(1)
    feats = rng.normal(size=(n, D)).astype(np.float32)
    frames = rng.integers(0, 256, size=(n, H, W, 3), dtype=np.uint8)
    action = rng.normal(0, 2, n).astype(np.float32)
    tokens = rng.integers(0, 20, size=(n, 12)).astype(np.int64)
    common = dict(group=group, ego_v=np.linspace(0, 25, n).astype(np.float32), action=action, tokens=tokens,
                  domain="driving", features=feats)
    d_new = PolicyData(images=ArrayImageSource(frames), **common)
    d_ref = v3.PolicyData(images=v3.ArrayImageSource(frames), **common)
    idx = np.arange(n)
    for kw in ({}, {"use_features": True}):
        m_new, i_new = train_policy(d_new, idx, PolicyConfig(steps=12, batch=8, threads=1, seed=2, **kw))
        m_ref, i_ref = v3.train_policy(d_ref, idx, v3.PolicyConfig(steps=12, batch=8, threads=1, seed=2, **kw))
        assert i_new["final_loss"] == i_ref["final_loss"], kw
        assert i_new["loss_curve"] == i_ref["loss_curve"], kw
        assert _same_state(m_new, m_ref), kw
        np.testing.assert_array_equal(predict_open_loop(m_new, d_new, idx), v3.predict_open_loop(m_ref, d_ref, idx))
        obs = d_new.observation(idx)
        np.testing.assert_array_equal(make_policy_fn(m_new, "driving")(obs), v3.make_policy_fn(m_ref, "driving")(obs))


def test_explicit_defaults_equal_implicit() -> None:
    """v4 필드를 기본값으로 명시해도 생략한 것과 같고, 새 모듈(GRU·보조 헤드)이 생기지 않는다."""

    v4_defaults = dict(feature_history=2, feature_stride=2, feature_encoder="mlp", aux_weight=0.0, aux_classes=6,
                       lang_dropout=0.0)
    group = np.repeat(np.arange(3), 12)
    n = len(group)
    data = _data(group, features=np.random.default_rng(3).normal(size=(n, D)).astype(np.float32))
    m1, i1 = train_policy(data, np.arange(n), _small_cfg(augment=True, use_features=True))
    m2, i2 = train_policy(data, np.arange(n), _small_cfg(augment=True, use_features=True, **v4_defaults))
    assert _same_state(m1, m2) and i1["final_loss"] == i2["final_loss"]
    assert i1["final_aux_loss"] is None and i1["aux_loss_curve"] == [] and i1["lang_dropped_frac"] == 0.0
    for name in ("aux_head", "feat_gru", "feat_in"):
        assert not hasattr(m1, name)
    assert m1.feature_history == 2 and m1.feature_stride == 2 and m1.aux_classes == 0
    torch.manual_seed(0)
    assert count_parameters(VLALitePolicy(vocab_size=89, use_features=True)) == 544_040  # docs/31 10.3


# ---------------------------------------------------------------------------
# P2 특징 이력·시간 인코더
# ---------------------------------------------------------------------------
def test_observation_feature_history_and_episode_clipping() -> None:
    group = np.array([7] * 20 + [3] * 10)
    data = _data(group, features=_index_features(30))
    idx = np.array([0, 5, 19, 20, 25, 29])
    obs = data.observation(idx, history=8, stride=2)
    assert obs["features"].shape == (6, 8, D) and obs["features"].dtype == np.float32
    got = obs["features"][:, :, 0].astype(int).tolist()
    assert got[0] == [0] * 8                              # 에피소드 첫 프레임
    assert got[1] == [5, 3, 1, 0, 0, 0, 0, 0]             # 시작 프레임(0)으로 잘라 냄
    assert got[2] == [19, 17, 15, 13, 11, 9, 7, 5]        # k=0이 현재, k는 t−2k
    assert got[3] == [20] * 8                             # 다음 에피소드 시작: 앞 에피소드로 넘어가지 않음
    assert got[4] == [25, 23, 21, 20, 20, 20, 20, 20]
    assert got[5] == [29, 27, 25, 23, 21, 20, 20, 20]
    # 영상은 history·stride와 무관하게 (t, t−2) 그대로다.
    np.testing.assert_array_equal(obs["image"], data.observation(idx)["image"])
    assert data.observation(idx, history=3, stride=5)["features"][:, :, 0].astype(int).tolist()[2] == [19, 14, 9]

    # history=2, stride=2(기본)는 v3 규칙 [features[t], features[prev_index(t)]]와 같은 배열이다.
    full = np.arange(30)
    feats = np.random.default_rng(0).normal(size=(30, D)).astype(np.float32)
    d2 = _data(group, features=feats)
    v3_style = np.stack([feats[full], feats[d2.prev_index(full)]], axis=1)
    out = d2.observation(full)["features"]
    np.testing.assert_array_equal(out, v3_style)
    assert out.dtype == v3_style.dtype and out.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(d2.observation(full, history=2, stride=2)["features"], v3_style)
    with pytest.raises(ValueError):
        d2.observation(full, history=0)
    with pytest.raises(ValueError):
        d2.observation(full, stride=0)


def test_feature_group_preroll_crosses_clip_start() -> None:
    """특징 프리롤: group=클립 id, feature_group=에피소드 id이면 특징 이력이 클립 시작 이전(같은 에피소드)까지 이어진다.

    영상(t−2)·행동 청크는 계속 클립(group) 기준이다. 에피소드 경계는 넘지 않는다.
    """

    n_ep, ep_len, clip_len = 2, 40, 10
    n = n_ep * ep_len
    ep = np.repeat(np.arange(n_ep), ep_len)
    clip = np.arange(n) // clip_len  # 클립 id(에피소드 안 연속 구간)
    feats = _index_features(n)
    d_clip = _data(clip, features=feats)
    d_pre = _data(clip, features=feats, feature_group=ep)
    idx = np.array([10, 12, 19, 41, 52])
    f_pre = d_pre.observation(idx, history=8, stride=2)["features"][:, :, 0].astype(int).tolist()
    f_clip = d_clip.observation(idx, history=8, stride=2)["features"][:, :, 0].astype(int).tolist()
    assert f_pre[0] == [10, 8, 6, 4, 2, 0, 0, 0]          # 클립 시작 프레임: 앞 클립(같은 에피소드) 프레임 포함
    assert f_clip[0] == [10] * 8                          # feature_group 없음: 클립 시작에서 잘라 냄(기존 규칙)
    assert f_pre[1] == [12, 10, 8, 6, 4, 2, 0, 0]
    assert f_clip[1] == [12, 10, 10, 10, 10, 10, 10, 10]
    assert f_pre[2] == [19, 17, 15, 13, 11, 9, 7, 5]
    assert f_pre[3] == [41, 40, 40, 40, 40, 40, 40, 40]   # 에피소드 1 시작(40)을 넘지 않는다
    assert f_pre[4] == [52, 50, 48, 46, 44, 42, 40, 40]
    # 영상(t, t−2)과 행동 청크는 group(클립) 기준 그대로다.
    o_pre, o_clip = d_pre.observation(idx, history=8, stride=2), d_clip.observation(idx, history=8, stride=2)
    np.testing.assert_array_equal(o_pre["image"], o_clip["image"])
    np.testing.assert_array_equal(d_pre.action_chunk(idx, 8), d_clip.action_chunk(idx, 8))
    # H=2 기본 관측도 feature_group 기준(클립 시작 프레임의 t−2가 앞 클립 프레임).
    assert d_pre.observation(np.array([10]))["features"][0, :, 0].tolist() == [10, 8]
    assert d_clip.observation(np.array([10]))["features"][0, :, 0].tolist() == [10, 10]
    with pytest.raises(ValueError, match="feature_group"):
        _data(clip, features=feats, feature_group=ep[:-1])
    # 학습: 클립 프레임만 학습 인덱스로 주고, 특징은 전체 길이 N(프리롤 포함) 배열을 쓴다.
    rng = np.random.default_rng(2)
    d_train = _data(clip, features=rng.normal(size=(n, D)).astype(np.float32), feature_group=ep)
    train_idx = np.flatnonzero(np.isin(clip, [1, 5]))
    model, _ = train_policy(d_train, train_idx, _small_cfg(use_features=True, feature_history=8, feature_encoder="gru"))
    assert predict_open_loop(model, d_train, train_idx).shape == (len(train_idx), 8)


def test_gru_encoder_shapes_order_and_errors() -> None:
    torch.manual_seed(0)
    model = VLALitePolicy(vocab_size=30, use_features=True, feature_history=8, feature_encoder="gru").eval()
    img, tok, prop = torch.rand(4, 6, H, W), torch.randint(0, 30, (4, 12)), torch.rand(4, 1)
    f = torch.randn(4, 8, D)
    assert model(img, tok, prop, f).shape == (4, 8)
    assert not hasattr(model, "feat_mlp") and isinstance(model.feat_gru, torch.nn.GRU)
    with torch.no_grad():
        enc = model.encode_features(f)
        assert enc.shape == (4, 64)  # feature_hidden: 헤드 입력 차원이 MLP 인코더와 같다
        # 시간 순서: 오래된 것(k=H−1) → 현재(k=0)로 뒤집어 GRU에 넣고 마지막 은닉을 쓴다.
        x = torch.relu(model.feat_in(f.clamp(-3.0, 3.0).flip(1)))  # clamp(−3,3) 포함(randn에는 |x|>3 값이 있다)
        _, h_n = model.feat_gru(x)
        assert torch.equal(enc, h_n[-1])
        assert not torch.equal(enc, model.encode_features(f.flip(1)))  # 순서에 민감
        # 이상치 처리: 100 → 3, NaN → 0.
        assert torch.equal(model.encode_features(torch.full((2, 8, D), 100.0)), model.encode_features(torch.full((2, 8, D), 3.0)))
        assert torch.equal(
            model.encode_features(torch.full((2, 8, D), float("nan"))), model.encode_features(torch.zeros(2, 8, D))
        )
    for bad in (torch.randn(4, 2, D), torch.randn(4, 8, D + 1), torch.randn(4, 8 * D)):
        with pytest.raises(ValueError):
            model(img, tok, prop, bad)
    with pytest.raises(ValueError):
        model(img, tok, prop)  # features 누락
    # mlp 인코더도 [B,H,D]로 일반화(H=8 → 입력 8·D).
    mlp8 = VLALitePolicy(vocab_size=30, use_features=True, feature_history=8)
    assert mlp8.feat_mlp[0].in_features == 8 * D
    assert mlp8(img, tok, prop, f).shape == (4, 8)
    with pytest.raises(ValueError):
        mlp8(img, tok, prop, torch.randn(4, 2, D))
    with pytest.raises(ValueError):
        VLALitePolicy(vocab_size=30, use_features=True, feature_encoder="lstm")
    with pytest.raises(ValueError):
        PolicyConfig(feature_encoder="lstm")
    with pytest.raises(ValueError):
        PolicyConfig(feature_history=0)


def test_v4_param_count() -> None:
    torch.manual_seed(0)
    v3 = VLALitePolicy(vocab_size=89, use_features=True)
    cfg = PolicyConfig(use_features=True, feature_history=8, feature_stride=2, feature_encoder="gru", aux_weight=0.2,
                       lang_dropout=0.15)
    v4 = VLALitePolicy.from_config(cfg, 89)
    hid = 64
    mlp2 = (2 * D * hid + hid) + (hid * hid + hid)
    gru = (D * hid + hid) + 3 * (hid * hid + hid * hid + 2 * hid)
    aux = (128 + 64 + 32 + hid) * 6 + 6
    assert count_parameters(v4) == count_parameters(v3) - mlp2 + gru + aux == 564_526
    assert v4.feature_history == 8 and v4.feature_stride == 2 and v4.aux_classes == 6


def test_make_policy_fn_history_attributes() -> None:
    group = np.repeat(np.arange(3), 20)
    n = len(group)
    feats = np.random.default_rng(5).normal(size=(n, D)).astype(np.float32)
    data = _data(group, features=feats)
    cfg = _small_cfg(use_features=True, feature_history=8, feature_stride=2, feature_encoder="gru")
    model, _ = train_policy(data, np.arange(n), cfg)
    fn = make_policy_fn(model, "driving")
    assert fn.feature_history == 8 and fn.feature_stride == 2 and fn.use_features is True  # type: ignore[attr-defined]
    idx = np.array([0, 3, 21, 59])
    obs = data.observation(idx, history=8, stride=2)
    np.testing.assert_allclose(fn(obs), predict_open_loop(model, data, idx)[:, 0], rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError, match="features"):
        fn(data.observation(idx))  # [B,2,D]는 H=8 모델에 맞지 않는다
    # 특징을 쓰지 않는 모델도 속성을 가진다(기본 2, 2).
    m_off, _ = train_policy(_data(group), np.arange(n), _small_cfg())
    fn_off = make_policy_fn(m_off, "driving")
    assert (fn_off.feature_history, fn_off.feature_stride, fn_off.use_features) == (2, 2, False)  # type: ignore[attr-defined]


def test_gru_history_learns_trend_better_than_two_frame_mlp() -> None:
    """특징 한 차원의 t와 t−14 차이(약 1초 추세)만 행동을 정하는 문제: gru(H=8, s=2)가 mlp(H=2)보다 MAE가 낮다.

    특징 5번 차원은 에피소드별 무작위 보행(증분 σ=0.12)이고 행동 = 4·(f[t] − f[t−14])다. 2프레임(t, t−2)으로는
    마지막 2개 증분만 보이므로 추세를 거의 알 수 없다. 영상은 상수, 언어 끔, chunk=1. 학습 에피소드 30개,
    평가는 처음 보는 에피소드 10개의 t ≥ 14 프레임. 측정(시드 0·1): gru 약 0.43, mlp 약 0.89~0.91(MAE/기준선).
    """

    n_groups, glen = 40, 40
    rng = np.random.default_rng(7)
    group = np.repeat(np.arange(n_groups), glen)
    n = len(group)
    feats = rng.normal(0, 0.3, size=(n, D)).astype(np.float32)  # 방해 차원
    inc = rng.normal(0, 0.12, size=n)
    walk = np.zeros(n)
    for g in range(n_groups):
        s = g * glen
        walk[s] = rng.normal(0, 0.5)
        for t in range(1, glen):
            walk[s + t] = walk[s + t - 1] + inc[s + t]
    feats[:, 5] = walk
    t_in = np.tile(np.arange(glen), n_groups)
    action = (4.0 * (walk - walk[np.arange(n) - np.minimum(t_in, 14)])).astype(np.float32)
    data = _data(group, action=action, features=feats, frames=np.full((n, H, W, 3), 90, dtype=np.uint8))
    usable = np.flatnonzero(t_in >= 14)
    tr, te = usable[group[usable] < 30], usable[group[usable] >= 30]
    baseline = float(np.abs(action[te] - action[tr].mean()).mean())

    def mae(**kw: object) -> float:
        cfg = PolicyConfig(steps=300, batch=32, chunk=1, augment=False, threads=1, seed=0, use_features=True,
                           use_language=False, **kw)  # type: ignore[arg-type]
        model, _ = train_policy(data, tr, cfg)
        return float(np.abs(predict_open_loop(model, data, te)[:, 0] - action[te]).mean()) / baseline

    r_gru = mae(feature_history=8, feature_stride=2, feature_encoder="gru")
    r_mlp = mae()
    assert r_gru < 0.6, (r_gru, r_mlp)
    assert r_mlp > 0.75, (r_gru, r_mlp)
    assert r_gru < 0.7 * r_mlp, (r_gru, r_mlp)


# ---------------------------------------------------------------------------
# P3 위험 맥락 보조 헤드
# ---------------------------------------------------------------------------
def test_aux_head_module_and_errors() -> None:
    group = np.repeat(np.arange(3), 10)
    n = len(group)
    data = _data(group)
    m0, _ = train_policy(data, np.arange(n), _small_cfg(aux_weight=0.0))
    assert not hasattr(m0, "aux_head") and m0.aux_classes == 0
    img, tok, prop = torch.rand(2, 6, H, W), torch.randint(0, 30, (2, 12)), torch.rand(2, 1)
    with pytest.raises(ValueError):
        m0(img, tok, prop, return_aux=True)

    with pytest.raises(ValueError, match="aux_targets"):
        train_policy(data, np.arange(n), _small_cfg(aux_weight=0.2))
    with pytest.raises(ValueError, match="aux_classes"):
        train_policy(_data(group, aux_targets=_uniform_aux(n, 4)), np.arange(n), _small_cfg(aux_weight=0.2))
    with pytest.raises(ValueError):
        _data(group, aux_targets=np.ones((n, 6), np.float32))  # 행 합이 1이 아님
    with pytest.raises(ValueError):
        _data(group, aux_targets=_uniform_aux(n - 1))
    with pytest.raises(ValueError):
        PolicyConfig(aux_weight=-0.1)

    # 보조 헤드: 마지막에 만들고, 기본 forward는 행동만 반환한다.
    torch.manual_seed(0)
    plain = VLALitePolicy(vocab_size=30)
    torch.manual_seed(0)
    aux = VLALitePolicy(vocab_size=30, aux_classes=6).eval()
    sp, sa = plain.state_dict(), aux.state_dict()
    assert all(torch.equal(sp[k], sa[k]) for k in sp)  # 공통 모듈 초기 가중치 동일(생성 순서 보존)
    assert set(sa) - set(sp) == {"aux_head.weight", "aux_head.bias"}
    assert aux.aux_head.in_features == aux.fused_dim == 128 + 64 + 32
    with torch.no_grad():
        act = aux(img, tok, prop)
        act2, logits = aux(img, tok, prop, return_aux=True)
    assert isinstance(act, torch.Tensor) and act.shape == (2, 8) and logits.shape == (2, 6)
    assert torch.equal(act, act2)

    # 보조 헤드가 있어도 추론 경로(make_policy_fn, predict_open_loop)는 그대로 동작한다.
    m_aux, info = train_policy(_data(group, aux_targets=_uniform_aux(n)), np.arange(n), _small_cfg(aux_weight=0.2))
    assert hasattr(m_aux, "aux_head") and info["final_aux_loss"] is not None
    obs = data.observation(np.arange(4))
    np.testing.assert_allclose(make_policy_fn(m_aux, "driving")(obs), predict_open_loop(m_aux, data, np.arange(4))[:, 0],
                               rtol=1e-5, atol=1e-6)


def test_soft_cross_entropy_value() -> None:
    logits = torch.tensor([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    q = torch.tensor([[0.7, 0.2, 0.1], [1.0, 0.0, 0.0]])
    expected = -(q * torch.log_softmax(logits, -1)).sum(-1).mean()
    assert torch.allclose(soft_cross_entropy(logits, q), expected)
    assert abs(float(soft_cross_entropy(torch.zeros(1, 6), torch.full((1, 6), 1 / 6))) - float(np.log(6))) < 1e-6


def test_aux_loss_decreases_on_synthetic_problem() -> None:
    """맥락 클래스가 특징 한 차원의 구간으로 정해지는 합성 문제에서 보조 손실이 줄어든다."""

    n_groups, glen = 40, 10
    rng = np.random.default_rng(11)
    group = np.repeat(np.arange(n_groups), glen)
    n = len(group)
    feats = rng.normal(0, 0.3, size=(n, D)).astype(np.float32)
    u = rng.uniform(-1, 1, n_groups)
    feats[:, 5] = u[group]
    cls = np.digitize(u[group], np.linspace(-1, 1, 7)[1:-1])
    q = np.full((n, 6), 0.02, dtype=np.float32)
    q[np.arange(n), cls] = 0.9
    data = _data(group, action=(u[group] * 2.0).astype(np.float32), features=feats, aux_targets=q,
                 frames=np.full((n, H, W, 3), 90, dtype=np.uint8))
    cfg = PolicyConfig(steps=200, batch=16, augment=True, threads=1, seed=0, use_features=True, aux_weight=0.5)
    _, info = train_policy(data, np.arange(n), cfg)
    curve = info["aux_loss_curve"]
    assert len(curve) == len(info["loss_curve"])
    assert curve[0] > 1.5                       # 시작: 약 ln 6 = 1.79
    assert curve[-1] < 0.75 * curve[0], curve   # 감소
    assert info["final_aux_loss"] < 0.75 * curve[0]
    # final_loss·loss_curve는 행동 손실만(보조 손실과 별도 기록, v3와 비교 가능).
    assert np.isfinite(info["final_loss"]) and info["loss_curve"] != curve


# ---------------------------------------------------------------------------
# T1 지시문 드롭아웃
# ---------------------------------------------------------------------------
def test_apply_lang_dropout() -> None:
    tokens = np.random.default_rng(0).integers(2, 20, size=(1000, 12)).astype(np.int64)
    rng = np.random.default_rng(4)
    same, none = apply_lang_dropout(tokens, rng, 0.0)
    assert same is tokens and not none.any()
    assert rng.random() == np.random.default_rng(4).random()  # p=0이면 난수를 소비하지 않는다
    allp, mask = apply_lang_dropout(tokens, np.random.default_rng(4), 1.0)
    assert mask.all() and (allp == 0).all() and (tokens != 0).all()  # 모든 표본이 빈 지시, 원본은 그대로
    part, m = apply_lang_dropout(tokens, np.random.default_rng(4), 0.15)
    assert 0.1 < m.mean() < 0.2
    assert (part[m] == 0).all() and np.array_equal(part[~m], tokens[~m])
    p2, m2 = apply_lang_dropout(tokens, np.random.default_rng(4), 0.15)
    assert np.array_equal(part, p2) and np.array_equal(m, m2)
    # 빈 지시(PAD만)의 언어 특징은 유한하다(마스크 평균 분모 clamp, 0으로 나누기 없음).
    torch.manual_seed(0)
    model = VLALitePolicy(vocab_size=30)
    lang = model.encode_language(torch.zeros(3, 12, dtype=torch.int64))
    assert torch.isfinite(lang).all()
    assert torch.equal(lang, torch.relu(model.lang_fc.bias).expand(3, -1))


def test_lang_dropout_training() -> None:
    group = np.repeat(np.arange(4), 12)
    n = len(group)
    idx = np.arange(n)
    data = _data(group)
    # p=0: 기본과 동일.
    m_def, _ = train_policy(data, idx, _small_cfg(augment=True))
    m_p0, i_p0 = train_policy(data, idx, _small_cfg(augment=True, lang_dropout=0.0))
    assert _same_state(m_def, m_p0) and i_p0["lang_dropped_frac"] == 0.0
    # p=1: 모든 표본이 빈 지시 → 토큰을 모두 PAD로 바꾼 데이터의 p=0 학습과 비트 단위로 같다
    # (드롭아웃 난수는 전용 Generator라 다른 난수 흐름을 바꾸지 않는다).
    m_p1, i_p1 = train_policy(data, idx, _small_cfg(augment=True, lang_dropout=1.0))
    pad_data = _data(group, tokens=np.zeros((n, 12), np.int64))
    m_pad, _ = train_policy(pad_data, idx, _small_cfg(augment=True))
    assert i_p1["lang_dropped_frac"] == 1.0
    assert _same_state(m_p1, m_pad)
    assert not _same_state(m_p1, m_def)
    # 결정성: 같은 시드 → 같은 결과, 드롭아웃이 실제로 학습을 바꾼다.
    cfg = _small_cfg(augment=True, lang_dropout=0.5, steps=8)
    a, ia = train_policy(data, idx, cfg)
    b, ib = train_policy(data, idx, cfg)
    assert _same_state(a, b) and ia["final_loss"] == ib["final_loss"] and ia["lang_dropped_frac"] == ib["lang_dropped_frac"]
    assert 0.0 < ia["lang_dropped_frac"] < 1.0
    c, _ = train_policy(data, idx, _small_cfg(augment=True, lang_dropout=0.5, steps=8, seed=1))
    assert not _same_state(a, c)
    # use_language=False면 드롭아웃은 효과가 없다(기록 0).
    _, i_nolang = train_policy(data, idx, _small_cfg(use_language=False, lang_dropout=0.5))
    assert i_nolang["lang_dropped_frac"] == 0.0
    with pytest.raises(ValueError):
        PolicyConfig(lang_dropout=1.5)


def test_v4_full_combination_trains_and_is_deterministic() -> None:
    """v4 기본 조합(H=8, s=2, gru, aux 0.2, lang_dropout 0.15)이 학습·추론되고 결정적이다."""

    group = np.repeat(np.arange(4), 20)
    n = len(group)
    rng = np.random.default_rng(9)
    feats = rng.normal(size=(n, D)).astype(np.float32)
    q = rng.dirichlet(np.ones(6), size=n).astype(np.float32)
    data = _data(group, features=feats, aux_targets=q)
    cfg = _small_cfg(augment=True, steps=8, use_features=True, feature_history=8, feature_stride=2,
                     feature_encoder="gru", aux_weight=0.2, lang_dropout=0.15)
    m1, i1 = train_policy(data, np.arange(n), cfg)
    m2, i2 = train_policy(data, np.arange(n), cfg)
    assert _same_state(m1, m2) and i1["final_loss"] == i2["final_loss"] and i1["final_aux_loss"] == i2["final_aux_loss"]
    assert i1["config"]["feature_encoder"] == "gru" and i1["n_params"] == count_parameters(m1)
    pred = predict_open_loop(m1, data, np.arange(n))
    assert pred.shape == (n, 8) and np.isfinite(pred).all()
