from __future__ import annotations

"""단계형(reset/step) 시뮬레이션 환경.

`world.simulate_episode`의 프레임 루프를 그대로 옮긴 환경이다. 폐루프 평가에서 외부 정책이
매 프레임 가속도 명령을 넣을 수 있도록 상태를 객체에 보관한다.

난수 소비 순서(회귀 fixture와 비트 단위 동일성의 근거)
1. `random.Random(seed)` 생성
2. `_build_scenario`(시나리오 객체 배치)
3. 자차 IDM 파라미터 t_head, a_max, b_comf, s0, 반응 지연(5회)
4. 초기 속도 배율(1회)
5. 프레임마다: 피치 노이즈(gauss 1회) → `simulate_detections`(가변)

스타일(`style`)을 주어도 위 순서는 그대로 두고, 뽑은 값 중 t_head/a_max/b_comf와 v0만 덮어쓴다.
따라서 같은 시드면 스타일과 무관하게 주변 객체 시나리오가 같다.

프레임 규약
- `reset()`은 프레임 0을 만든다. 프레임 0의 물리 구간은 내부 전문가(반응 지연 큐)로 진행한다.
  반응 지연 큐는 0으로 채워져 시작하므로(지연 ≥ 서브스텝 2개), 이 구간의 실제 명령은 사실상 0이다.
- `step(cmd)`는 다음 프레임 1개(물리 서브스텝 `substeps`개)를 진행하고 그 프레임의 관측을 돌려준다.
  `step(None)`은 기존 내부 전문가(IDM + 반응 지연 큐), `step(cmd)`는 외부 명령을 프레임 동안 유지한다.
- 에피소드는 `n_frames`개 프레임(0..n_frames-1)으로 끝나며, 마지막 프레임 이후 `done=True`다.
"""

import logging
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..features.base import Detection, FrameDetections
from .camera import KIND_TO_CLS, CameraConfig, DetectorNoiseConfig, project_actor, simulate_detections
from .labels import ContextLabeler, FrameState, LabelThresholds
from .world import Actor, DomainProfile, _build_scenario, _idm, profile_for_scenario

if TYPE_CHECKING:  # 순환 import 방지(vla → sim 방향만 실제 import)
    from ..vla.instructions import DrivingStyle

logger = logging.getLogger(__name__)

# 액추에이터 1차 지연 시정수(초). world.simulate_episode와 같은 값.
ACTUATOR_TAU_S = 0.3
# 충돌 판정 거리(m). 경로 객체까지 거리가 이보다 작으면 충돌로 센다.
COLLISION_GAP_M = 0.3
# "자차 주행 중 충돌" 판정 속도(m/s). 정지에 가까운 자차를 상대가 들이받는 경우와 구분한다.
MOVING_SPEED_EPS: dict[str, float] = {"driving": 0.5, "robot": 0.1}


@dataclass(slots=True)
class StepInfo:
    """프레임 1개의 관측·GT 상태.

    - frame: 노이즈 검출 결과(특징 계산용)
    - gt_boxes: float32 [M,6] = (x1, y1, x2, y2, cls_id, depth_m), 원본 카메라 픽셀 좌표(GT 투영)
    - horizon_y: 피치(가감속+노이즈)가 반영된 지평선 y(원본 픽셀)
    - ego_v, ego_a: 소수 4자리 반올림 값(기존 Episode와 동일), ego_x: 자차 위치(m)
    - label: 6맥락 라벨, ttc: 경로 TTC(무한대는 -1, 소수 4자리)
    - collided: 이번 프레임 물리 구간에서 충돌이 있었는지
    - t: 프레임 시각(초), gap: 경로 객체까지 거리(없으면 inf), closing_speed: ego_v - 객체 속도(없으면 0)
    - expert_cmd: 프레임 시점 상태의 지연 없는 IDM 명령(스타일 파라미터 사용)
    - path_is_vehicle: 경로 객체가 차량인지(계약 외 추가 필드, 추종 구간 판정용)
    - collided_moving: 이번 프레임에 자차 속도 > MOVING_SPEED_EPS 상태에서 충돌했는지(계약 외 추가 필드)
    """

    frame: FrameDetections
    gt_boxes: np.ndarray
    horizon_y: float
    ego_v: float
    ego_a: float
    ego_x: float
    label: int
    ttc: float
    collided: bool
    t: float
    gap: float
    closing_speed: float
    expert_cmd: float
    path_is_vehicle: bool = False
    collided_moving: bool = False


class SimEnv:
    """시나리오 1개를 프레임 단위로 진행하는 환경."""

    def __init__(
        self,
        scenario: str,
        seed: int,
        fps: float = 15.0,
        duration_s: float = 30.0,
        noise: DetectorNoiseConfig | None = None,
        style: DrivingStyle | None = None,
        physics_hz: float = 30.0,
        *,
        cam: CameraConfig | None = None,
        thresholds: LabelThresholds | None = None,
    ) -> None:
        if fps <= 0 or duration_s <= 0:
            raise ValueError(f"fps와 duration_s는 양수여야 한다: fps={fps}, duration_s={duration_s}")
        self.scenario = scenario
        self.seed = int(seed)
        self.fps = float(fps)
        self.duration_s = float(duration_s)
        self.style = style
        self.instruction_style: str | None = style.name if style is not None else None
        self.prof: DomainProfile = profile_for_scenario(scenario)
        self.cam = cam or CameraConfig(cam_height_m=self.prof.cam_height_m, max_range_m=self.prof.max_range_m)
        self.noise = noise or DetectorNoiseConfig()
        self._thresholds = thresholds or self.prof.thresholds
        self.substeps = max(1, int(math.ceil(physics_hz / fps)))
        self.dt = 1.0 / (fps * self.substeps)
        self.n_frames = int(round(duration_s * fps))
        self.frame_idx = -1
        self.done = False
        self.v_target = 0.0
        self._reset_state()

    # ------------------------------------------------------------------ 초기화
    def _reset_state(self) -> None:
        """난수 순서를 지키며 시나리오와 자차 파라미터를 만든다."""

        prof = self.prof
        self.rng = random.Random(self.seed)
        rng = self.rng
        self.labeler = ContextLabeler(self._thresholds)
        v0, self.actors, self.meta = _build_scenario(self.scenario, rng, self.duration_s)
        self.v0_scenario = v0
        # 자차 IDM 파라미터(운전자 성향 무작위화) — 스타일이 있어도 난수는 그대로 소비한다
        t_head = rng.uniform(*prof.t_head)
        a_max = rng.uniform(*prof.a_max)
        b_comf = rng.uniform(*prof.b_comf)
        self.s0 = rng.uniform(*prof.s0)
        self.delay_steps = max(0, int(round(rng.uniform(*prof.delay_s) / self.dt)))
        if self.style is not None:
            from ..vla.instructions import target_speed

            t_head, a_max, b_comf = self.style.t_head, self.style.a_max, self.style.b_comf
            v0 = target_speed(v0, self.style, prof.name)
        self.t_head, self.a_max, self.b_comf, self.v0 = t_head, a_max, b_comf, v0
        self.v_target = v0
        self.cmd_queue: list[float] = [0.0] * self.delay_steps
        self.ego_x, self.ego_v, self.ego_a = 0.0, v0 * rng.uniform(0.85, 1.0), 0.0
        self.pitch_noise = 0.0
        self.collisions = 0
        self.collisions_moving = 0
        self.step_count = 0
        self.frame_idx = -1
        self.done = False

    def reset(self) -> StepInfo:
        """환경을 처음 상태로 되돌리고 프레임 0의 관측을 반환한다."""

        self._reset_state()
        return self._advance_frame(None)

    # ------------------------------------------------------------------ 경로/전문가
    def _in_path(self, actor: Actor, horizon_s: float = 2.5) -> bool:
        prof = self.prof
        half = prof.path_half_vehicle if actor.kind == "vehicle" else prof.path_half_vru
        if abs(actor.y) < half:
            return True
        if actor.kind != "vehicle" and actor.y_target is not None and actor.lat_speed > 0:
            direction = math.copysign(1.0, actor.y_target - actor.y)
            future_y = actor.y + direction * actor.lat_speed * horizon_s
            return (actor.y > 0) != (future_y > 0) or abs(future_y) < half
        return False

    def _path_obstacle(self) -> tuple[float, float, bool]:
        best_gap, best_v, best_vehicle = math.inf, 0.0, False
        ego_x = self.ego_x
        for actor in self.actors:
            gap = actor.x - ego_x
            if gap <= -0.5 or not self._in_path(actor):
                continue
            if gap < best_gap:
                best_gap, best_v, best_vehicle = gap, actor.v, actor.kind == "vehicle"
        return best_gap, best_v, best_vehicle

    def _idm_command(self) -> float:
        gap, v_obs, _ = self._path_obstacle()
        cmd = _idm(self.ego_v, self.v0, gap, self.ego_v - v_obs, self.a_max, self.b_comf, self.t_head, self.s0)
        return max(-self.prof.ego_brake_max, min(self.a_max, cmd))

    def expert_command(self) -> float:
        """현재 상태의 지연 없는 IDM 명령(m/s^2). [-brake_max, a_max]로 자른다."""

        return self._idm_command()

    # ------------------------------------------------------------------ 물리
    def _move_actors(self, t: float) -> None:
        prof, dt, actors, ego_x = self.prof, self.dt, self.actors, self.ego_x
        for actor in actors:
            actor.apply_events(t, ego_x)
            acc = max(-actor.brake_max, min(actor.acc_max, 0.8 * (actor.v_des - actor.v)))
            if actor.follows and actor.kind == "vehicle":
                lane_tol = 0.43 * prof.lane_w
                ahead = [
                    o
                    for o in actors
                    if o is not actor and o.kind == "vehicle" and 0.0 < o.x - actor.x and abs(o.y - actor.y) < lane_tol
                ]
                if ahead:
                    lead = min(ahead, key=lambda o: o.x)
                    gap = lead.x - actor.x - prof.vehicle_length
                    ia, ib, it, is0 = prof.actor_idm
                    acc = min(acc, _idm(actor.v, max(actor.v_des, 0.3 * prof.lane_w), gap, actor.v - lead.v, ia, ib, it, is0))
                    acc = max(acc, -prof.ego_brake_max)
            actor.v = actor.v + acc * dt
            if actor.kind == "vehicle":
                actor.v = max(0.0, actor.v)
            actor.x += actor.v * dt
            if actor.y_target is not None and actor.lat_speed > 0:
                dy = actor.y_target - actor.y
                move = math.copysign(min(abs(dy), actor.lat_speed * dt), dy)
                actor.y += move
                if abs(dy) < 1e-3:
                    actor.lat_speed = 0.0

    def _physics_substep(self, ext_cmd: float | None) -> tuple[bool, bool]:
        """물리 서브스텝 1회. (충돌 여부, 자차 주행 중 충돌 여부)를 반환한다."""

        dt = self.dt
        self._move_actors(self.step_count * dt)
        if ext_cmd is None:
            # 내부 전문가: IDM + 반응 지연 큐
            self.cmd_queue.append(self._idm_command())
            applied = self.cmd_queue.pop(0)
        else:
            applied = ext_cmd
        # 실제 가속도는 1차 지연(브레이크/구동 응답) 적용
        self.ego_a += (applied - self.ego_a) * min(1.0, dt / ACTUATOR_TAU_S)
        self.ego_v = max(0.0, self.ego_v + self.ego_a * dt)
        if self.ego_v == 0.0 and self.ego_a < 0:
            self.ego_a = 0.0
        self.ego_x += self.ego_v * dt
        gap_now, v_obs_now, _ = self._path_obstacle()
        collided = moving = False
        if gap_now < COLLISION_GAP_M:
            self.collisions += 1
            moving = self.ego_v > MOVING_SPEED_EPS[self.prof.name]
            self.collisions_moving += int(moving)
            self.ego_x = self.ego_x - (COLLISION_GAP_M - gap_now)
            self.ego_v = min(self.ego_v, v_obs_now)
            collided = True
        self.step_count += 1
        return collided, moving

    # ------------------------------------------------------------------ 프레임
    def _advance_frame(self, ext_cmd: float | None) -> StepInfo:
        prof, cam, noise, rng = self.prof, self.cam, self.noise, self.rng
        self.frame_idx += 1
        frame_idx = self.frame_idx
        collided = collided_moving = False
        for _ in range(self.substeps):
            hit, hit_moving = self._physics_substep(ext_cmd)
            collided = collided or hit
            collided_moving = collided_moving or hit_moving

        # --- 프레임 샘플링: 투영, 검출, 라벨
        t_frame = self.step_count * self.dt
        self.pitch_noise = 0.8 * self.pitch_noise + rng.gauss(0.0, noise.pitch_jitter_px * noise.level)
        horizon_y = cam.horizon_ratio * cam.height - cam.pitch_px_per_mps2 * (-self.ego_a) + self.pitch_noise
        projected = []
        cam_x = self.ego_x - prof.cam_behind_front_m
        for idx, actor in enumerate(self.actors):
            depth = actor.x - cam_x
            width_m, height_m = prof.sizes[actor.kind]
            box = project_actor(cam, idx, actor.kind, depth, actor.y, width_m, height_m, horizon_y)
            if box is not None:
                projected.append(box)
        dets: list[Detection] = simulate_detections(projected, cam, noise, rng)
        frame = FrameDetections(frame_idx, round(frame_idx / self.fps, 6), cam.width, cam.height, dets)
        gt_boxes = np.asarray(
            [(b.x1, b.y1, b.x2, b.y2, float(KIND_TO_CLS[b.kind]), b.depth) for b in projected], dtype=np.float32
        ).reshape(-1, 6)

        gap, v_obs, is_vehicle = self._path_obstacle()
        nearby = sum(
            1
            for a in self.actors
            if (a.kind == "vehicle" or prof.name == "robot")
            and 0.0 < a.x - self.ego_x < prof.nearby_range_m
            and abs(a.y) < prof.nearby_lateral_m
        )
        state = FrameState(t_frame, self.ego_v, self.ego_a, gap, self.ego_v - v_obs, is_vehicle, nearby)
        label = self.labeler(state)
        ttc = self.labeler.ttc(state)
        self.done = frame_idx >= self.n_frames - 1
        return StepInfo(
            frame=frame,
            gt_boxes=gt_boxes,
            horizon_y=float(horizon_y),
            ego_v=round(self.ego_v, 4),
            ego_a=round(self.ego_a, 4),
            ego_x=float(self.ego_x),
            label=int(label),
            ttc=round(ttc, 4) if math.isfinite(ttc) else -1.0,
            collided=collided,
            t=frame.t,
            gap=float(gap),
            closing_speed=float(self.ego_v - v_obs) if math.isfinite(gap) else 0.0,
            expert_cmd=float(self._idm_command()),
            path_is_vehicle=bool(is_vehicle) and math.isfinite(gap),
            collided_moving=collided_moving,
        )

    def step(self, cmd: float | None = None) -> StepInfo:
        """다음 프레임으로 진행한다.

        - cmd=None: 내부 전문가(IDM + 반응 지연 큐)를 서브스텝마다 계산한다(기존 simulate_episode와 동일).
        - cmd=float: 외부 가속도 명령(m/s^2)을 프레임 동안 유지한다. 반응 지연 큐는 쓰지 않고
          액추에이터 1차 지연만 적용한다. 명령은 [-ego_brake_max, ego_accel_max]로 자른다.
        """

        if self.frame_idx < 0:
            raise RuntimeError("reset()을 먼저 호출해야 한다")
        if self.done:
            raise RuntimeError(f"에피소드가 끝났다(n_frames={self.n_frames})")
        ext: float | None = None
        if cmd is not None:
            value = float(cmd)
            if not math.isfinite(value):
                logger.warning("비유한 명령 %r을 0으로 대체(scenario=%s, seed=%d)", cmd, self.scenario, self.seed)
                value = 0.0
            ext = max(-self.prof.ego_brake_max, min(self.prof.ego_accel_max, value))
        return self._advance_frame(ext)

    def summary(self) -> dict[str, Any]:
        """에피소드 메타(기존 Episode.meta와 같은 키 + 스타일 정보)."""

        meta = dict(self.meta)
        meta.update(
            {
                "seed": self.seed,
                "v0": round(self.v0, 3),
                "t_head": round(self.t_head, 3),
                "reaction_delay_s": round(self.delay_steps * self.dt, 3),
                "collisions_steps": self.collisions,
                "noise_level": self.noise.level,
                "domain": self.prof.name,
            }
        )
        if self.style is not None:
            meta.update(
                {
                    "style": self.style.name,
                    "v_target": round(self.v_target, 4),
                    "v0_scenario": round(self.v0_scenario, 3),
                    "a_max": round(self.a_max, 3),
                    "b_comf": round(self.b_comf, 3),
                }
            )
        return meta
