from __future__ import annotations

"""VLA-lite 정책 폐루프 평가.

모든 테스트 에피소드(`EpisodeSpec`)를 lockstep으로 동시에 진행하고, 매 프레임 정책을 배치로 1회 호출한다.
정책 관측은 `vla.obs` 규칙을 따른다.
- image: uint8 [B,64,64,6] = (render_frame(프레임 t), render_frame(프레임 max(t−2, 0)))
- tokens: int64 [B,L] = encode_instruction(영어 지시문)
- proprio: float32 [B,1] = ego_v / speed_scale(domain)
정책 출력(물리 단위 가속도 명령, m/s^2)은 `SimEnv.step(cmd)`로 다음 프레임 동안 유지된다
(반응 지연 큐 없이 액추에이터 1차 지연만 적용).

`policy_fn=None`이면 전문가 참조로, 같은 시드에서 `SimEnv.expert_command()`(지연 없는 스타일 IDM 명령)를
같은 `step(cmd)` 경로로 적용한다. 정책과 액추에이터 경로가 같으므로 공정한 기준이 된다.

프레임 0은 `SimEnv.reset()`이 만든다(내부 전문가, 반응 지연 큐가 0으로 차 있어 사실상 무명령 구간).
정책은 프레임 0..n−2의 관측으로 명령을 내고, 지표는 프레임 0..n−1 전체에서 계산한다.

에피소드 지표 정의
- collision: 에피소드 중 한 번이라도 충돌(경로 객체까지 거리 < 0.3 m)이 있었는지
- collisions_steps: 충돌한 물리 서브스텝 수(`SimEnv.collisions`)
- collision_moving: 자차 속도가 MOVING_SPEED_EPS(주행 0.5, 로봇 0.1 m/s)를 넘는 상태에서 충돌했는지.
  정지에 가까운 자차를 상대(예: 마주 오는 작업자)가 들이받는 경우를 제외한 "자차 기인" 충돌 근사
- min_ttc: 프레임별 경로 TTC(접근 속도가 라벨 임계 이하이면 무한대)의 최솟값, 최대 TTC_CAP_S(10 s)로 자름
- hazard_success: 위험 시나리오(HAZARD_SCENARIOS)에서 충돌이 없으면 True, 그 외 시나리오는 None
- speed_error: 자유주행 프레임의 |ego_v − v_target| 평균(m/s). 자유주행 프레임 =
  경로 객체가 없거나, gap > max(FREE_GAP_FLOOR_M, 4·T_style·v_target)이면서 TTC가 없거나 FREE_TTC_S(8 s) 초과.
  해당 프레임이 없으면 None
- headway_error: 추종 프레임의 |gap/ego_v − T_style| 평균(초). 추종 프레임 =
  경로 객체가 차량이고, ego_v ≥ FOLLOW_MIN_SPEED, gap < 2.5·T_style·ego_v, |closing_speed| ≤ FOLLOW_DV_TOL.
  gap은 자차 앞범퍼–선행 객체 후미 거리다. 해당 프레임이 없으면 None.
  IDM 평형 간격은 (s0 + v·T)/sqrt(1 − (v/v0)^4)라서 전문가도 T보다 길게 따라가므로 0이 되지 않는다.
  스타일 구분 여부를 보려면 headway_mean(추종 프레임의 gap/ego_v 평균)을 함께 본다
- rms_jerk: 실제 가속도(ego_a, 액추에이터 지연 반영)의 프레임 차분 jerk(m/s^3)의 RMS
- distance: 에피소드 동안 진행 거리(m), progress: distance / (v_target·duration_s)
- hard_brake_frames: ego_a < −hard_decel(라벨 임계: 주행 4.0, 로봇 1.2 m/s^2)인 프레임 수
- mean_speed: 평균 속도(m/s)

집계(시나리오별·스타일별·전체): 에피소드 값의 평균. collision_rate는 충돌 에피소드 비율,
hazard_success_rate는 위험 시나리오 에피소드만으로 계산하고, speed_error/headway_error는 값이 있는
에피소드만 평균한다(None 제외). min_ttc는 평균(min_ttc_mean)과 최솟값(min_ttc_min)을 함께 낸다.
"""

import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..sim.camera import DetectorNoiseConfig
from ..sim.env import SimEnv, StepInfo
from ..sim.render import render_frame
from ..sim.world import scenario_types_for_domain
from .instructions import N_PARAPHRASES, STYLE_NAMES, encode_instruction, instruction_text, styles_for_domain
from .obs import HISTORY_OFFSET, IMG_SIZE, proprio

logger = logging.getLogger(__name__)

PolicyFn = Callable[[dict[str, np.ndarray]], np.ndarray]

TTC_CAP_S = 10.0
FREE_TTC_S = 8.0
HAZARD_SCENARIOS: dict[str, tuple[str, ...]] = {
    "driving": ("lead_brake", "cut_in", "vru_crossing", "stop_and_go"),
    "robot": ("robot_agent_stop", "robot_human_crossing", "robot_human_headon"),
}
FREE_GAP_FLOOR_M: dict[str, float] = {"driving": 40.0, "robot": 6.0}
FOLLOW_MIN_SPEED: dict[str, float] = {"driving": 3.0, "robot": 0.3}
FOLLOW_DV_TOL: dict[str, float] = {"driving": 1.5, "robot": 0.3}


@dataclass(frozen=True)
class EpisodeSpec:
    """폐루프 테스트 에피소드 1개(시나리오·시드·스타일·패러프레이즈)."""

    scenario: str
    seed: int
    style: str
    paraphrase: int


def make_test_specs(domain: str, n_per_cell: int, seed_base: int) -> list[EpisodeSpec]:
    """시나리오 × 스타일 칸마다 n_per_cell개씩 결정적으로 만든다.

    시드는 seed_base부터 연속으로 붙이고, 패러프레이즈는 전체 순번 % N_PARAPHRASES로 고르게 돌린다.
    """

    if n_per_cell <= 0:
        raise ValueError(f"n_per_cell은 양수여야 한다: {n_per_cell}")
    specs: list[EpisodeSpec] = []
    idx = 0
    for scenario in scenario_types_for_domain(domain):
        for style in STYLE_NAMES:
            for _ in range(n_per_cell):
                specs.append(EpisodeSpec(scenario, seed_base + idx, style, idx % N_PARAPHRASES))
                idx += 1
    return specs


def _nanmean(values: list[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _episode_metrics(
    spec: EpisodeSpec, env: SimEnv, infos: list[StepInfo], domain: str, t_style: float
) -> dict[str, Any]:
    fps = env.fps
    v = np.asarray([i.ego_v for i in infos], dtype=np.float64)
    a = np.asarray([i.ego_a for i in infos], dtype=np.float64)
    gap = np.asarray([i.gap for i in infos], dtype=np.float64)
    closing = np.asarray([i.closing_speed for i in infos], dtype=np.float64)
    ttc = np.asarray([i.ttc for i in infos], dtype=np.float64)
    is_vehicle = np.asarray([i.path_is_vehicle for i in infos], dtype=bool)
    v_t = float(env.v_target)

    finite_ttc = ttc[ttc >= 0.0]
    min_ttc = float(min(TTC_CAP_S, finite_ttc.min())) if finite_ttc.size else TTC_CAP_S
    collision = any(i.collided for i in infos)
    collision_moving = any(i.collided_moving for i in infos)

    free_gap = max(FREE_GAP_FLOOR_M[domain], 4.0 * t_style * v_t)
    no_obstacle = ~np.isfinite(gap)
    far = np.isfinite(gap) & (gap > free_gap) & ((ttc < 0.0) | (ttc > FREE_TTC_S))
    free = no_obstacle | far
    speed_error = float(np.mean(np.abs(v[free] - v_t))) if free.any() else None

    follow = (
        is_vehicle
        & np.isfinite(gap)
        & (v >= FOLLOW_MIN_SPEED[domain])
        & (gap < 2.5 * t_style * np.maximum(v, 1e-6))
        & (np.abs(closing) <= FOLLOW_DV_TOL[domain])
    )
    headway = gap[follow] / v[follow] if follow.any() else None
    headway_error = float(np.mean(np.abs(headway - t_style))) if headway is not None else None
    headway_mean = float(np.mean(headway)) if headway is not None else None

    jerk = np.diff(a) * fps
    rms_jerk = float(np.sqrt(np.mean(jerk**2))) if jerk.size else 0.0
    distance = float(infos[-1].ego_x)
    hard_decel = env.prof.thresholds.hard_decel
    hazard = spec.scenario in HAZARD_SCENARIOS[domain]
    return {
        "scenario": spec.scenario,
        "seed": spec.seed,
        "style": spec.style,
        "paraphrase": spec.paraphrase,
        "v_target": round(v_t, 4),
        "collision": bool(collision),
        "collisions_steps": int(env.collisions),
        "collision_moving": bool(collision_moving),
        "min_ttc": round(min_ttc, 4),
        "hazard": hazard,
        "hazard_success": (not collision) if hazard else None,
        "speed_error": speed_error,
        "headway_error": headway_error,
        "headway_mean": headway_mean,
        "n_free_frames": int(free.sum()),
        "n_follow_frames": int(follow.sum()),
        "rms_jerk": rms_jerk,
        "distance": distance,
        "progress": distance / max(1e-6, v_t * env.duration_s),
        "hard_brake_frames": int((a < -hard_decel).sum()),
        "mean_speed": float(v.mean()),
    }


def aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """에피소드 지표 목록을 평균 집계한다(정의는 모듈 docstring)."""

    if not episodes:
        return {"n": 0}
    hazard = [e for e in episodes if e["hazard"]]
    min_ttcs = [e["min_ttc"] for e in episodes]
    return {
        "n": len(episodes),
        "collision_rate": float(np.mean([e["collision"] for e in episodes])),
        "collisions_steps_mean": float(np.mean([e["collisions_steps"] for e in episodes])),
        "collision_moving_rate": float(np.mean([e["collision_moving"] for e in episodes])),
        "min_ttc_mean": float(np.mean(min_ttcs)),
        "min_ttc_min": float(np.min(min_ttcs)),
        "n_hazard": len(hazard),
        "hazard_success_rate": float(np.mean([e["hazard_success"] for e in hazard])) if hazard else None,
        "speed_error": _nanmean([e["speed_error"] for e in episodes]),
        "headway_error": _nanmean([e["headway_error"] for e in episodes]),
        "headway_mean": _nanmean([e["headway_mean"] for e in episodes]),
        "rms_jerk": float(np.mean([e["rms_jerk"] for e in episodes])),
        "distance": float(np.mean([e["distance"] for e in episodes])),
        "progress": float(np.mean([e["progress"] for e in episodes])),
        "hard_brake_frames": float(np.mean([e["hard_brake_frames"] for e in episodes])),
        "mean_speed": float(np.mean([e["mean_speed"] for e in episodes])),
    }


def _group(episodes: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for e in episodes:
        groups.setdefault(e[key], []).append(e)
    return {name: aggregate(items) for name, items in groups.items()}


def run_closed_loop(
    policy_fn: PolicyFn | None,
    specs: list[EpisodeSpec],
    domain: str = "driving",
    fps: float = 15.0,
    duration_s: float = 30.0,
    noise_level: float = 1.0,
) -> dict[str, Any]:
    """폐루프 평가를 실행해 에피소드별·시나리오별·스타일별·전체 지표를 반환한다.

    Args:
        policy_fn: 배치 관측 → float32 [B] 가속도 명령(m/s^2). None이면 지연 없는 IDM 전문가 참조.
        specs: 테스트 에피소드 목록(`make_test_specs`).
        domain: "driving" | "robot". specs의 시나리오 도메인과 같아야 한다.
    """

    if not specs:
        raise ValueError("specs가 비어 있다")
    styles = styles_for_domain(domain)
    valid_scenarios = set(scenario_types_for_domain(domain))
    bad = sorted({s.scenario for s in specs if s.scenario not in valid_scenarios})
    if bad:
        raise ValueError(f"도메인 {domain}에 없는 시나리오: {bad}")

    t0 = time.perf_counter()
    envs = [
        SimEnv(s.scenario, s.seed, fps=fps, duration_s=duration_s, noise=DetectorNoiseConfig(level=noise_level), style=styles[s.style])
        for s in specs
    ]
    b = len(envs)
    tokens = np.stack(
        [encode_instruction(instruction_text(s.style, env.v_target, domain, s.paraphrase, "en")) for s, env in zip(specs, envs)]
    )
    infos = [env.reset() for env in envs]
    history: list[list[StepInfo]] = [[info] for info in infos]
    n_frames = envs[0].n_frames
    h, w = IMG_SIZE
    slots = HISTORY_OFFSET + 1
    buf = np.zeros((slots, b, h, w, 3), dtype=np.uint8) if policy_fn is not None else None
    t_render = t_policy = 0.0

    for t in range(n_frames - 1):
        if policy_fn is None:
            cmds = np.asarray([env.expert_command() for env in envs], dtype=np.float32)
        else:
            assert buf is not None
            tr = time.perf_counter()
            slot = t % slots
            for i, info in enumerate(infos):
                buf[slot, i] = render_frame(info.gt_boxes, info.horizon_y, domain)
            prev_slot = max(t - HISTORY_OFFSET, 0) % slots
            obs = {
                "image": np.concatenate([buf[slot], buf[prev_slot]], axis=-1),
                "tokens": tokens,
                "proprio": proprio(np.asarray([info.ego_v for info in infos], dtype=np.float32), domain),
            }
            tp = time.perf_counter()
            t_render += tp - tr
            out = policy_fn(obs)
            t_policy += time.perf_counter() - tp
            cmds = np.asarray(out, dtype=np.float32).reshape(-1)
            if cmds.shape[0] != b:
                raise ValueError(f"정책 출력 크기 {cmds.shape[0]} != 배치 {b}")
        infos = [env.step(float(c)) for env, c in zip(envs, cmds)]
        for i, info in enumerate(infos):
            history[i].append(info)

    episodes = [
        _episode_metrics(spec, env, hist, domain, styles[spec.style].t_head)
        for spec, env, hist in zip(specs, envs, history)
    ]
    wall = time.perf_counter() - t0
    result = {
        "mode": "expert" if policy_fn is None else "policy",
        "config": {"domain": domain, "fps": fps, "duration_s": duration_s, "noise_level": noise_level, "n_episodes": b},
        "episodes": episodes,
        "by_scenario": _group(episodes, "scenario"),
        "by_style": _group(episodes, "style"),
        "overall": aggregate(episodes),
        "timing": {"wall_s": round(wall, 3), "render_s": round(t_render, 3), "policy_s": round(t_policy, 3)},
        "specs": [asdict(s) for s in specs],
    }
    logger.info(
        "폐루프(%s, %s): %d 에피소드, 충돌률 %.3f, %.1fs",
        result["mode"],
        domain,
        b,
        result["overall"]["collision_rate"],
        wall,
    )
    return result
