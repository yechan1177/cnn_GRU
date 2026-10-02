"""A2 시뮬레이터 모듈 테스트: SimEnv 회귀·외부 제어, 렌더러, 지시문, 풀, 폐루프."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from vcp.sim.camera import KIND_TO_CLS, DetectorNoiseConfig
from vcp.sim.env import SimEnv
from vcp.sim.render import render_frame
from vcp.sim.world import ROBOT_SCENARIO_TYPES, SCENARIO_TYPES, simulate_episode
from vcp.vla.closed_loop import EpisodeSpec, make_test_specs, run_closed_loop
from vcp.vla.instructions import (
    N_PARAPHRASES,
    STYLE_NAMES,
    UNK_ID,
    VOCAB,
    encode_instruction,
    instruction_text,
    styles_for_domain,
    target_speed,
    tokenize,
)
from vcp.vla.obs import IMG_SIZE, proprio, stack_frames
from vcp.vla.pool import PoolConfig, frame_boxes, generate_pool, load_pool, style_for_seed

FIXTURE = Path(__file__).parent / "fixtures" / "sim_regression.json.gz"


def _digest(ep) -> dict:
    return {
        "boxes": [[[b.x1, b.y1, b.x2, b.y2, b.cls_id, b.conf] for b in f.boxes] for f in ep.frames],
        "frame_t": [f.t for f in ep.frames],
        "labels": list(ep.labels),
        "ego_v": list(ep.ego_v),
        "ego_a": list(ep.ego_a),
        "ttc": list(ep.ttc),
        "meta": json.loads(json.dumps(ep.meta)),
    }


def test_simulate_episode_matches_pre_refactor_fixture() -> None:
    """리팩터링 전 simulate_episode 출력과 비트 단위로 같아야 한다(주행 7 + 로봇 6 시나리오 × 2 시드)."""

    with gzip.open(FIXTURE, "rt", encoding="utf-8") as file:
        fixture = json.load(file)
    scenarios = {e["scenario"] for e in fixture["episodes"]}
    assert scenarios == set(SCENARIO_TYPES) | set(ROBOT_SCENARIO_TYPES)
    for expected in fixture["episodes"]:
        ep = simulate_episode(expected["scenario"], seed=expected["seed"], fps=fixture["fps"], duration_s=fixture["duration_s"])
        got = _digest(ep)
        for key, value in got.items():
            assert value == expected[key], f"{expected['scenario']} seed={expected['seed']} {key} 불일치"


def test_simenv_external_control_and_lifecycle() -> None:
    env = SimEnv("lead_brake", seed=21, duration_s=6.0)
    with pytest.raises(RuntimeError):
        env.step(0.0)
    first = env.reset()
    assert first.t == 0.0 and env.frame_idx == 0 and not env.done
    assert first.gt_boxes.dtype == np.float32 and first.gt_boxes.ndim == 2 and first.gt_boxes.shape[1] == 6
    infos = [first]
    while not env.done:
        infos.append(env.step(0.0))
    assert len(infos) == env.n_frames == 90
    with pytest.raises(RuntimeError):
        env.step(0.0)
    # 외부 명령 0을 유지하면 가속도는 0으로 수렴(액추에이터 1차 지연만 적용)
    assert abs(infos[-1].ego_a) < 1e-3

    # reset은 같은 초기 상태를 재현하고, 큰 제동 명령은 [-brake_max, accel_max]로 잘린다
    again = env.reset()
    assert again.ego_v == first.ego_v and np.array_equal(again.gt_boxes, first.gt_boxes)
    for _ in range(30):
        info = env.step(-100.0)
    assert info.ego_a >= -env.prof.ego_brake_max - 1e-6
    assert info.ego_v < first.ego_v

    # expert_command는 지연 없는 IDM: 스텝 정보의 expert_cmd와 같고 [-brake_max, a_max] 범위
    info = env.step(env.expert_command())
    assert info.expert_cmd == pytest.approx(env.expert_command())
    assert -env.prof.ego_brake_max <= info.expert_cmd <= env.a_max


def test_simenv_style_overrides_idm_and_keeps_scenario() -> None:
    style = styles_for_domain("driving")["cautious"]
    base = SimEnv("follow", seed=5, duration_s=4.0)
    styled = SimEnv("follow", seed=5, duration_s=4.0, style=style)
    b0, s0 = base.reset(), styled.reset()
    assert styled.t_head == style.t_head and styled.a_max == style.a_max and styled.b_comf == style.b_comf
    assert styled.v_target == pytest.approx(target_speed(base.v0, style, "driving"))
    assert styled.delay_steps == base.delay_steps and styled.s0 == base.s0
    assert len(styled.actors) == len(base.actors)
    assert [a.x for a in styled.actors] == [a.x for a in base.actors]
    assert styled.instruction_style == "cautious"
    meta = styled.summary()
    assert meta["style"] == "cautious" and "v_target" in meta
    assert b0.t == s0.t


def test_gt_class_ids_match_detections() -> None:
    """노이즈 0이면 검출 박스는 GT 박스 좌표·클래스와 같다(클래스 id 규약 일치 확인)."""

    assert KIND_TO_CLS == {"person": 0, "vehicle": 1, "bike": 2}
    env = SimEnv("vru_crossing", seed=3, duration_s=10.0, noise=DetectorNoiseConfig(level=0.0, pitch_jitter_px=0.0))
    info = env.reset()
    checked = 0
    while True:
        for det in info.frame.boxes:
            coords = np.asarray([det.x1, det.y1, det.x2, det.y2], dtype=np.float32)
            match = np.all(np.abs(info.gt_boxes[:, :4] - coords) < 1e-2, axis=1)
            assert match.sum() == 1 and int(info.gt_boxes[match][0, 4]) == det.cls_id
            checked += 1
        if env.done:
            break
        info = env.step(None)
    assert checked > 0


def test_render_shape_determinism_and_domains() -> None:
    env = SimEnv("dense_traffic", seed=2, duration_s=2.0)
    info = env.reset()
    img = render_frame(info.gt_boxes, info.horizon_y, "driving")
    assert img.shape == (*IMG_SIZE, 3) and img.dtype == np.uint8
    assert np.array_equal(img, render_frame(info.gt_boxes.copy(), info.horizon_y, "driving"))
    empty = render_frame(np.zeros((0, 6), np.float32), info.horizon_y, "driving")
    assert not np.array_equal(img, empty)  # 객체가 그려졌다
    assert not np.array_equal(empty, render_frame(np.zeros((0, 6), np.float32), info.horizon_y - 40.0, "driving"))
    assert not np.array_equal(empty, render_frame(np.zeros((0, 6), np.float32), info.horizon_y, "robot"))
    # 가까운 객체가 먼 객체를 덮는다
    near_far = np.asarray([[300, 200, 340, 260, 1, 10.0], [310, 210, 330, 250, 0, 50.0]], dtype=np.float32)
    out = render_frame(near_far, 216.0)
    assert np.array_equal(out, render_frame(near_far[::-1].copy(), 216.0))
    big = render_frame(near_far[:1], 216.0)
    assert np.array_equal(out, big)
    small = render_frame(np.asarray([[0, 0, 4, 4, 2, 5.0]], np.float32), 216.0, size=(32, 48))
    assert small.shape == (32, 48, 3)


def test_obs_helpers() -> None:
    cur = np.zeros((*IMG_SIZE, 3), np.uint8)
    prev = np.full((*IMG_SIZE, 3), 7, np.uint8)
    stacked = stack_frames(cur, prev)
    assert stacked.shape == (*IMG_SIZE, 6) and stacked[..., 3:].max() == 7
    p = proprio(np.asarray([15.0, 30.0]), "driving")
    assert p.shape == (2, 1) and p.dtype == np.float32 and p[1, 0] == pytest.approx(1.0)
    assert proprio(1.5, "robot").shape == (1,)


def test_encode_instruction_and_vocab() -> None:
    assert VOCAB["<pad>"] == 0 and VOCAB["<unk>"] == 1
    assert list(VOCAB.values()) == list(range(len(VOCAB)))
    for domain in ("driving", "robot"):
        for style in STYLE_NAMES:
            v = target_speed(17.0 if domain == "driving" else 1.1, styles_for_domain(domain)[style], domain)
            for k in range(N_PARAPHRASES):
                text = instruction_text(style, v, domain, k)
                ids = encode_instruction(text)
                assert ids.dtype == np.int64 and ids.shape == (24,)
                assert UNK_ID not in ids.tolist(), text
                assert int((ids > 0).sum()) == len(tokenize(text))
                assert instruction_text(style, v, domain, k, "ko")
    text = instruction_text("brisk", 60 / 3.6, "driving", 0)
    assert "60" in tokenize(text) and "km/h" in tokenize(text) and "short" in tokenize(text)
    assert encode_instruction(text, max_len=4).shape == (4,)
    assert encode_instruction("zzz qqq")[:2].tolist() == [UNK_ID, UNK_ID]
    assert target_speed(1.23, styles_for_domain("robot")["normal"], "robot") == pytest.approx(1.2)


def test_pool_roundtrip(tmp_path: Path) -> None:
    cfg = PoolConfig(n_episodes=3, seed_base=100000, duration_s=3.0, workers=1)
    npz = generate_pool(tmp_path / "pool", cfg)
    data, meta = load_pool(tmp_path / "pool")
    n = 3 * 45
    assert npz.exists() and meta["n_frames"] == n and len(meta["episodes"]) == 3
    for key in ("X_v1", "X_v2"):
        assert data[key].shape == (n, 16) and data[key].dtype == np.float32
    assert data["X_v1v2"].shape == (n, 32)
    assert data["y"].dtype == np.int16 and data["ep"].dtype == np.int32
    for key in ("t", "ego_v", "ego_a", "ttc", "expert_cmd", "horizon_y", "v_target"):
        assert data[key].shape == (n,) and data[key].dtype == np.float32
    assert data["box_ptr"].shape == (n + 1,) and data["box_ptr"][-1] == len(data["boxes"])
    assert data["boxes"].dtype == np.float32 and data["boxes"].shape[1] == 6
    assert data["style_id"].dtype == np.int8 and data["paraphrase"].dtype == np.int8
    ep0 = meta["episodes"][0]
    style, para = style_for_seed(ep0["seed"], "driving")
    assert ep0["style"] == style.name and ep0["paraphrase"] == para
    assert ep0["instruction_en"] == instruction_text(style.name, ep0["v_target"], "driving", para)
    # 저장한 박스는 같은 시드의 SimEnv GT 박스와 같다
    env = SimEnv(ep0["scenario"], ep0["seed"], duration_s=3.0, style=style)
    info = env.reset()
    for _ in range(10):
        info = env.step(None)
    assert np.array_equal(frame_boxes(data, 10), info.gt_boxes)
    assert data["horizon_y"][10] == np.float32(info.horizon_y)
    # 로드 경로는 .npz를 붙여도 된다
    data2, _ = load_pool(npz)
    assert np.array_equal(data2["y"], data["y"])


def test_make_test_specs_balanced() -> None:
    specs = make_test_specs("driving", 2, 200000)
    assert len(specs) == len(SCENARIO_TYPES) * len(STYLE_NAMES) * 2
    assert len({s.seed for s in specs}) == len(specs)
    assert specs == make_test_specs("driving", 2, 200000)
    robot = make_test_specs("robot", 1, 400000)
    assert {s.scenario for s in robot} == set(ROBOT_SCENARIO_TYPES)


def test_closed_loop_small_run() -> None:
    specs = [
        EpisodeSpec("lead_brake", 200001, "normal", 0),
        EpisodeSpec("free_drive", 200002, "brisk", 1),
        EpisodeSpec("vru_crossing", 200003, "cautious", 2),
    ]
    seen: list[tuple] = []

    def zero_policy(obs: dict[str, np.ndarray]) -> np.ndarray:
        seen.append((obs["image"].shape, obs["image"].dtype, obs["tokens"].shape, obs["tokens"].dtype, obs["proprio"].shape))
        return np.zeros(len(obs["proprio"]), dtype=np.float32)

    zero = run_closed_loop(zero_policy, specs, duration_s=4.0)
    expert = run_closed_loop(None, specs, duration_s=4.0)
    assert seen[0] == ((3, *IMG_SIZE, 6), np.uint8, (3, 24), np.int64, (3, 1))
    assert len(seen) == 59  # 60프레임 중 마지막을 제외한 프레임마다 1회
    for result in (zero, expert):
        assert len(result["episodes"]) == 3
        assert set(result["by_style"]) == {"normal", "brisk", "cautious"}
        overall = result["overall"]
        for key in ("collision_rate", "min_ttc_mean", "hazard_success_rate", "speed_error", "rms_jerk", "distance"):
            assert key in overall
        assert result["episodes"][1]["hazard_success"] is None
    assert zero["overall"]["rms_jerk"] <= expert["overall"]["rms_jerk"]
    # 같은 입력이면 결과가 같다(결정성)
    again = run_closed_loop(None, specs, duration_s=4.0)
    assert again["episodes"] == expert["episodes"]
    with pytest.raises(ValueError):
        run_closed_loop(None, [EpisodeSpec("robot_crowded", 1, "normal", 0)], domain="driving")
