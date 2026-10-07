from __future__ import annotations

"""VLA-lite 정책(`vcp.vla.policy`) 테스트.

A2 모듈(obs/render/instructions)과 독립적으로 검증하도록 가짜 영상 소스와 가짜 토큰을 쓴다.
A2 모듈이 있으면 연결 테스트를 추가로 수행한다.
"""

import numpy as np
import pytest
import torch

from vcp.vla import policy as P
from vcp.vla.policy import (
    ArrayImageSource,
    PolicyConfig,
    PolicyData,
    PoolImageSource,
    VLALitePolicy,
    chunk_weights,
    make_policy_fn,
    predict_open_loop,
    train_policy,
)

H = W = 64


def _index_frames(n: int) -> np.ndarray:
    """프레임 i의 모든 픽셀 값이 i % 256인 가짜 프레임."""

    return np.broadcast_to((np.arange(n) % 256).astype(np.uint8)[:, None, None, None], (n, H, W, 3)).copy()


def _data(group: np.ndarray, action: np.ndarray | None = None, frames: np.ndarray | None = None) -> PolicyData:
    n = len(group)
    rng = np.random.default_rng(0)
    return PolicyData(
        images=ArrayImageSource(frames if frames is not None else _index_frames(n)),
        group=group,
        ego_v=rng.uniform(0, 30, n).astype(np.float32),
        action=action if action is not None else np.arange(n, dtype=np.float32),
        tokens=rng.integers(2, 20, size=(n, 12)).astype(np.int64),
        domain="driving",
    )


def _small_cfg(**kw: object) -> PolicyConfig:
    base = dict(steps=5, batch=8, augment=False, threads=1, seed=0)
    base.update(kw)
    return PolicyConfig(**base)  # type: ignore[arg-type]


def test_model_output_shape() -> None:
    model = VLALitePolicy(vocab_size=30, chunk=8)
    out = model(torch.rand(5, 6, H, W), torch.randint(0, 30, (5, 12)), torch.rand(5, 1))
    assert out.shape == (5, 8)
    # 어휘 밖 토큰도 UNK로 처리되어야 한다.
    out2 = model(torch.rand(2, 6, H, W), torch.full((2, 12), 999), torch.rand(2, 1))
    assert torch.isfinite(out2).all()


def test_language_ablation_ignores_tokens() -> None:
    torch.manual_seed(0)
    ablated = VLALitePolicy(vocab_size=30, use_language=False)
    torch.manual_seed(0)
    full = VLALitePolicy(vocab_size=30, use_language=True)
    for mod in (ablated, full):
        mod.eval()
    img, prop = torch.rand(4, 6, H, W), torch.rand(4, 1)
    tok_a = torch.randint(2, 30, (4, 12))
    tok_b = torch.randint(2, 30, (4, 12))
    with torch.no_grad():
        assert torch.equal(ablated(img, tok_a, prop), ablated(img, tok_b, prop))
        assert not torch.allclose(full(img, tok_a, prop), full(img, tok_b, prop))
    assert not hasattr(ablated, "embed")


def test_chunk_targets_do_not_cross_group() -> None:
    group = np.array([0, 0, 0, 1, 1, 1, 1])
    data = _data(group)
    chunks = data.action_chunk(np.array([0, 1, 2, 3, 5, 6]), chunk=4)
    assert chunks.tolist() == [
        [0, 1, 2, 2],
        [1, 2, 2, 2],
        [2, 2, 2, 2],
        [3, 4, 5, 6],
        [5, 6, 6, 6],
        [6, 6, 6, 6],
    ]


def test_history_rule_respects_group_start() -> None:
    group = np.array([7, 7, 7, 7, 3, 3, 3, 3])  # id 값이 아니라 연속 구간으로 경계를 정한다
    data = _data(group)
    idx = np.arange(8)
    assert data.prev_index(idx).tolist() == [0, 0, 0, 1, 4, 4, 4, 5]
    obs = data.observation(idx)
    assert obs["image"].shape == (8, H, W, 6) and obs["image"].dtype == np.uint8
    assert obs["image"][:, 0, 0, 0].tolist() == idx.tolist()                    # 현재 프레임
    assert obs["image"][:, 0, 0, 3].tolist() == [0, 0, 0, 1, 4, 4, 4, 5]       # t−2 프레임
    assert obs["tokens"].shape == (8, 12) and obs["proprio"].shape == (8, 1)
    np.testing.assert_allclose(obs["proprio"][:, 0], data.ego_v / 30.0, rtol=1e-6)


def test_chunk_weights_front_loaded() -> None:
    w = chunk_weights(8, front_weight=0.5)
    assert w.shape == (8,) and abs(float(w.mean()) - 1.0) < 1e-6
    assert np.all(np.diff(w) < 0)
    assert chunk_weights(1).tolist() == [1.0]


def test_learns_object_position_to_accel() -> None:
    """영상 속 밝은 사각형(선행 객체 대용)의 세로 위치 → 가속도의 작은 합성 문제를 수백 단계 안에 학습한다.

    참고: 전역 밝기만 다른 균일 영상은 GroupNorm이 밝기 크기를 정규화로 지우므로(부호만 남는다) 쓰지 않는다.
    실제 과제(객체 위치·크기 → 제동)와 같은 위치 의존 문제로 검증한다.
    """

    n_groups, glen = 40, 10
    rng = np.random.default_rng(1)
    pos = rng.uniform(0, 1, n_groups)
    group = np.repeat(np.arange(n_groups), glen)
    frames = np.full((len(group), H, W, 3), 40, dtype=np.uint8)
    for i, g in enumerate(group):
        top = int(8 + pos[g] * 40)
        frames[i, top : top + 10, 26:38] = 230
    action = ((pos[group] - 0.5) * 2.0 * 4.0).astype(np.float32)  # 정규화 단위 [-1, 1]
    data = _data(group, action=action, frames=frames)
    cfg = PolicyConfig(steps=200, batch=16, augment=False, threads=1, seed=0)
    model, info = train_policy(data, np.arange(len(group)), cfg)
    pred = predict_open_loop(model, data, np.arange(len(group)))
    assert pred.shape == (len(group), cfg.chunk)
    mae = float(np.abs(pred[:, 0] - action).mean())
    baseline = float(np.abs(action - action.mean()).mean())
    assert mae < 0.2 * baseline, (mae, baseline)
    assert info["loss_curve"][-1] < info["loss_curve"][0]
    for key in ("loss_curve", "loss_curve_steps", "train_time_s", "samples_per_s", "n_params", "final_loss"):
        assert key in info


def test_make_policy_fn_output() -> None:
    group = np.repeat(np.arange(3), 10)
    data = _data(group)
    model, _ = train_policy(data, np.arange(len(group)), _small_cfg(augment=True))
    fn = make_policy_fn(model, "driving")
    idx = np.array([0, 4, 15, 29])
    out = fn(data.observation(idx))
    assert out.shape == (4,) and out.dtype == np.float32
    np.testing.assert_allclose(out, predict_open_loop(model, data, idx)[:, 0], rtol=1e-5, atol=1e-6)
    assert not model.training


def test_training_is_deterministic() -> None:
    group = np.repeat(np.arange(4), 12)
    data = _data(group)
    idx = np.arange(len(group))
    m1, _ = train_policy(data, idx, _small_cfg(augment=True))
    m2, _ = train_policy(data, idx, _small_cfg(augment=True))
    m3, _ = train_policy(data, idx, _small_cfg(augment=True, seed=1))
    s1, s2, s3 = m1.state_dict(), m2.state_dict(), m3.state_dict()
    assert all(torch.equal(s1[k], s2[k]) for k in s1)
    assert not all(torch.equal(s1[k], s3[k]) for k in s1)


def test_threads_restored_after_training() -> None:
    before = torch.get_num_threads()
    group = np.repeat(np.arange(2), 8)
    train_policy(_data(group), np.arange(16), _small_cfg(steps=2, threads=1))
    assert torch.get_num_threads() == before


def test_shift_augmentation_shape_and_range() -> None:
    img = np.random.default_rng(0).integers(0, 255, (4, H, W, 6), dtype=np.uint8)
    out = P.shift_images(img, np.random.default_rng(0), max_shift=2)
    assert out.shape == img.shape and out.dtype == np.uint8
    jit = P.photometric_jitter(torch.rand(4, 6, H, W), torch.Generator().manual_seed(0))
    assert float(jit.min()) >= 0.0 and float(jit.max()) <= 1.0


def test_pool_image_source_lru_cap() -> None:
    n = 6
    box_ptr = np.array([0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    boxes = np.zeros((3, 6), dtype=np.float32)
    calls: list[float] = []

    def fake_render(b: np.ndarray, horizon_y: float, domain: str = "driving", size: tuple[int, int] = (H, W)) -> np.ndarray:
        calls.append(horizon_y)
        return np.full((*size, 3), int(horizon_y), dtype=np.uint8)

    src = PoolImageSource(box_ptr, boxes, np.arange(n, dtype=np.float32) * 10, domain="driving", size=(H, W), max_cache=3)
    src._render = fake_render  # 렌더러 주입(A2 render 없이 캐시 동작만 검증)
    out = src.get(np.array([0, 1, 2, 0]))
    assert out.shape == (4, H, W, 3) and out[:, 0, 0, 0].tolist() == [0, 10, 20, 0]
    assert len(calls) == 3  # 마지막 0은 캐시 적중
    src.get(np.array([3, 4]))  # 상한 3 → 1, 2 순으로 밀려난다(0은 최근에 썼다)
    info = src.cache_info()
    assert info["size"] == 3 and info["hits"] == 1 and info["misses"] == 5


def test_fallback_obs_matches_a2_contract() -> None:
    obs = pytest.importorskip("vcp.vla.obs")
    fb = P._FALLBACK_OBS
    assert tuple(obs.IMG_SIZE) == fb.IMG_SIZE and obs.HISTORY_OFFSET == fb.HISTORY_OFFSET
    for d in ("driving", "robot"):
        assert obs.accel_scale(d) == fb.accel_scale(d) and obs.speed_scale(d) == fb.speed_scale(d)
    a = np.random.default_rng(0).integers(0, 255, (2, H, W, 3), dtype=np.uint8)
    b = np.random.default_rng(1).integers(0, 255, (2, H, W, 3), dtype=np.uint8)
    np.testing.assert_array_equal(obs.stack_frames(a, b), fb.stack_frames(a, b))


def test_pool_image_source_with_real_renderer() -> None:
    pytest.importorskip("vcp.sim.render")
    box_ptr = np.array([0, 1, 1], dtype=np.int64)
    boxes = np.array([[280, 220, 360, 300, 1, 15.0]], dtype=np.float32)
    src = PoolImageSource(box_ptr, boxes, np.array([200.0, 200.0], dtype=np.float32), domain="driving")
    frames = src.get(np.array([0, 1]))
    assert frames.shape == (2, H, W, 3) and frames.dtype == np.uint8
    assert not np.array_equal(frames[0], frames[1])  # 객체가 있는 프레임과 없는 프레임은 달라야 한다
