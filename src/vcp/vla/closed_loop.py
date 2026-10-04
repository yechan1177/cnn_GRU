from __future__ import annotations

"""VLA-lite 정책 폐루프 평가.

모든 테스트 에피소드(`EpisodeSpec`)를 lockstep으로 동시에 진행하고, 매 프레임 정책을 배치로 1회 호출한다.
정책 관측은 `vla.obs` 규칙을 따른다.
- image: uint8 [B,64,64,6] = (render_frame(프레임 t), render_frame(프레임 max(t−2, 0)))
- tokens: int64 [B,L] = encode_instruction(영어 지시문)
- proprio: float32 [B,1] = ego_v / speed_scale(domain)
- features(v3): float32 [B,2,32] = (프레임 t, 프레임 max(t−2, 0))의 v1+v2 검출 특징.
  에피소드마다 `obs.OnlineFeatureTracker`(풀과 같은 추출기·순서·conf 0.45·max_det 30)를 두고
  매 프레임 노이즈 검출(`info.frame`)로 갱신한다. 정책이 쓰지 않으면 무시해도 된다
- features(v4 일반화, docs/29 10절): float32 [B,H,32]. 인덱스 k는 프레임 max(t − k·s, 0)의 특징(k=0 현재).
  H·s는 `run_closed_loop(feature_history=, feature_stride=)`로 주고, None이면 policy_fn의 속성
  `feature_history`/`feature_stride`(A14 `make_policy_fn`이 붙임), 그것도 없으면 2/2(v3와 같음)를 쓴다.
  결과 `config`에 실제로 쓴 두 값을 기록한다
정책 출력(물리 단위 가속도 명령, m/s^2)은 `SimEnv.step(cmd)`로 다음 프레임 동안 유지된다
(반응 지연 큐 없이 액추에이터 1차 지연만 적용).

`policy_fn=None`이면 전문가 참조로, 같은 시드에서 `SimEnv.expert_command()`(지연 없는 스타일 IDM 명령)를
같은 `step(cmd)` 경로로 적용한다. 정책과 액추에이터 경로가 같으므로 공정한 기준이 된다.

프레임 0은 `SimEnv.reset()`이 만든다(내부 전문가, 반응 지연 큐가 0으로 차 있어 사실상 무명령 구간).
정책은 프레임 0..n−2의 관측으로 명령을 내고, 지표는 프레임 0..n−1 전체에서 계산한다.

v3 벤치마크 옵션(docs/33)
- `collision_pushback`, `decouple_initial_speed`: `SimEnv`에 그대로 전달한다(전문가 참조에도 같게 적용).
- 반사실 언어 평가(B2): `make_test_specs(..., counterfactual=True)`는 (시나리오, 시드)마다 세 스타일 사양을
  모두 만든다. 같은 시드라 주변 객체 시나리오가 같고 지시문만 다르다(decouple_initial_speed=True면 초기 속도도 같다).
  결과 dict의 `"counterfactual"`은 사양이 이 구조(모든 (시나리오, 시드) 묶음이 세 스타일을 1개씩 가짐)인지 기록한다.
  스타일별 speed_error·headway_mean은 `by_style`에 있다.

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
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..sim.camera import DetectorNoiseConfig
from ..sim.env import SimEnv, StepInfo
from ..sim.render import render_frame
from ..sim.world import scenario_types_for_domain
from .instructions import N_PARAPHRASES, STYLE_NAMES, encode_instruction, instruction_text, styles_for_domain
from .obs import DEFAULT_FEATURE_HISTORY, DEFAULT_FEATURE_STRIDE, HISTORY_OFFSET, IMG_SIZE, FeatureHistory, proprio

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


def make_test_specs(domain: str, n_per_cell: int, seed_base: int, counterfactual: bool = False) -> list[EpisodeSpec]:
    """시나리오 × 스타일 칸마다 n_per_cell개씩 결정적으로 만든다.

    - counterfactual=False(기존): 시드는 seed_base부터 사양마다 연속으로 붙이고,
      패러프레이즈는 전체 순번 % N_PARAPHRASES로 고르게 돌린다.
    - counterfactual=True(반사실 언어 평가, B2): 시나리오마다 시드 n_per_cell개를 seed_base부터 연속으로 붙이고
      (시나리오, 시드)마다 세 스타일 사양을 STYLE_NAMES 순서로 모두 만든다.
      패러프레이즈는 시드 % N_PARAPHRASES로 정해 세 스타일이 같은 번호를 쓴다.
      사양 수는 시나리오 수 × n_per_cell × 3이고, 고유 시드 수는 시나리오 수 × n_per_cell이다.
    """

    if n_per_cell <= 0:
        raise ValueError(f"n_per_cell은 양수여야 한다: {n_per_cell}")
    specs: list[EpisodeSpec] = []
    idx = 0
    if counterfactual:
        for scenario in scenario_types_for_domain(domain):
            for _ in range(n_per_cell):
                seed = seed_base + idx
                paraphrase = seed % N_PARAPHRASES
                specs.extend(EpisodeSpec(scenario, seed, style, paraphrase) for style in STYLE_NAMES)
                idx += 1
        return specs
    for scenario in scenario_types_for_domain(domain):
        for style in STYLE_NAMES:
            for _ in range(n_per_cell):
                specs.append(EpisodeSpec(scenario, seed_base + idx, style, idx % N_PARAPHRASES))
                idx += 1
    return specs


def is_counterfactual(specs: list[EpisodeSpec]) -> bool:
    """모든 (시나리오, 시드) 묶음이 세 스타일을 정확히 1개씩, 같은 패러프레이즈로 가지면 True."""

    groups: dict[tuple[str, int], list[EpisodeSpec]] = defaultdict(list)
    for s in specs:
        groups[(s.scenario, s.seed)].append(s)
    if not groups:
        return False
    full = set(STYLE_NAMES)
    for items in groups.values():
        if len(items) != len(full) or {s.style for s in items} != full or len({s.paraphrase for s in items}) != 1:
            return False
    return True


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


def resolve_feature_history(
    policy_fn: PolicyFn | None, feature_history: int | None = None, feature_stride: int | None = None
) -> tuple[int, int]:
    """폐루프 특징 이력 (H, s)를 정한다.

    우선순위: 명시 인자 → policy_fn 속성(`feature_history`, `feature_stride`) → 기본값 2/2(v3).
    두 값은 따로 정한다(예: 인자로 H만 주면 s는 속성 또는 기본값). 1 이상의 정수가 아니면 ValueError.
    """

    def pick(value: int | None, attr: str, default: int) -> int:
        if value is None:
            value = getattr(policy_fn, attr, None) if policy_fn is not None else None
        if value is None:
            value = default
        iv = int(value)
        if iv != value or iv < 1:
            raise ValueError(f"{attr}는 1 이상의 정수여야 한다: {value!r}")
        return iv

    return (
        pick(feature_history, "feature_history", DEFAULT_FEATURE_HISTORY),
        pick(feature_stride, "feature_stride", DEFAULT_FEATURE_STRIDE),
    )


def run_closed_loop(
    policy_fn: PolicyFn | None,
    specs: list[EpisodeSpec],
    domain: str = "driving",
    fps: float = 15.0,
    duration_s: float = 30.0,
    noise_level: float = 1.0,
    collision_pushback: bool = True,
    decouple_initial_speed: bool = False,
    feature_history: int | None = None,
    feature_stride: int | None = None,
) -> dict[str, Any]:
    """폐루프 평가를 실행해 에피소드별·시나리오별·스타일별·전체 지표를 반환한다.

    Args:
        policy_fn: 배치 관측 → float32 [B] 가속도 명령(m/s^2). None이면 지연 없는 IDM 전문가 참조.
        specs: 테스트 에피소드 목록(`make_test_specs`).
        domain: "driving" | "robot". specs의 시나리오 도메인과 같아야 한다.
        collision_pushback: False면 충돌 시 자차를 뒤로 밀지 않는다(v3 B3). 전문가 참조에도 같게 적용한다.
        decouple_initial_speed: True면 스타일 초기 속도를 시나리오 기본 속도 기준으로 정한다(v3 B1).
        feature_history: 관측 `"features"`의 이력 길이 H(v4). None이면 policy_fn.feature_history, 없으면 2.
        feature_stride: 이력 간격 s(프레임). None이면 policy_fn.feature_stride, 없으면 2.
            기본값(2/2)이면 v3와 같은 관측·결과다(`config`에 두 값이 추가로 기록되는 것만 다르다).
    """

    if not specs:
        raise ValueError("specs가 비어 있다")
    styles = styles_for_domain(domain)
    valid_scenarios = set(scenario_types_for_domain(domain))
    bad = sorted({s.scenario for s in specs if s.scenario not in valid_scenarios})
    if bad:
        raise ValueError(f"도메인 {domain}에 없는 시나리오: {bad}")
    feat_h, feat_s = resolve_feature_history(policy_fn, feature_history, feature_stride)

    t0 = time.perf_counter()
    envs = [
        SimEnv(
            s.scenario,
            s.seed,
            fps=fps,
            duration_s=duration_s,
            noise=DetectorNoiseConfig(level=noise_level),
            style=styles[s.style],
            collision_pushback=collision_pushback,
            decouple_initial_speed=decouple_initial_speed,
        )
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
    # 검출 특징 버퍼(인덱스 k = 프레임 max(t − k·s, 0), 기본 2/2는 영상과 같은 t, t−2 규칙).
    # 전문가 참조는 관측을 쓰지 않으므로 계산하지 않는다
    feat_hist = FeatureHistory(b, history=feat_h, stride=feat_s) if policy_fn is not None else None
    # P5 자차 운동 이력(v4): policy_fn.proprio_history > 1이면 "proprio" [B,Hp] = 프레임 max(t − k·s, 0)의 정규화 속도
    prop_h = int(getattr(policy_fn, "proprio_history", 1) or 1) if policy_fn is not None else 1
    prop_s = int(getattr(policy_fn, "proprio_stride", 2) or 2) if policy_fn is not None else 2
    v_hist: list[np.ndarray] = []
    t_render = t_policy = t_features = 0.0

    for t in range(n_frames - 1):
        if policy_fn is None:
            cmds = np.asarray([env.expert_command() for env in envs], dtype=np.float32)
        else:
            assert buf is not None and feat_hist is not None
            tr = time.perf_counter()
            slot = t % slots
            for i, info in enumerate(infos):
                buf[slot, i] = render_frame(info.gt_boxes, info.horizon_y, domain)
            tf = time.perf_counter()
            features = feat_hist.update([info.frame for info in infos])
            dt_feat = time.perf_counter() - tf
            prev_slot = max(t - HISTORY_OFFSET, 0) % slots
            obs = {
                "image": np.concatenate([buf[slot], buf[prev_slot]], axis=-1),
                "tokens": tokens,
                "proprio": proprio(np.asarray([info.ego_v for info in infos], dtype=np.float32), domain),
                "features": features,
            }
            if prop_h > 1:
                v_hist.append(obs["proprio"][:, 0])
                ks = [max(t - k * prop_s, 0) for k in range(prop_h)]
                obs["proprio"] = np.stack([v_hist[j] for j in ks], axis=1).astype(np.float32)
            tp = time.perf_counter()
            # render_s는 기존처럼 렌더링 + 관측 배치 구성 시간, features_s는 특징 갱신 시간
            t_features += dt_feat
            t_render += (tp - tr) - dt_feat
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
    counterfactual = is_counterfactual(specs)
    result = {
        "mode": "expert" if policy_fn is None else "policy",
        "config": {
            "domain": domain,
            "fps": fps,
            "duration_s": duration_s,
            "noise_level": noise_level,
            "n_episodes": b,
            "collision_pushback": bool(collision_pushback),
            "decouple_initial_speed": bool(decouple_initial_speed),
            "feature_history": feat_h,
            "feature_stride": feat_s,
            "proprio_history": prop_h,
            "proprio_stride": prop_s,
        },
        "counterfactual": counterfactual,
        "episodes": episodes,
        "by_scenario": _group(episodes, "scenario"),
        "by_style": _group(episodes, "style"),
        "overall": aggregate(episodes),
        "timing": {
            "wall_s": round(wall, 3),
            "render_s": round(t_render, 3),
            "features_s": round(t_features, 3),
            "policy_s": round(t_policy, 3),
        },
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
