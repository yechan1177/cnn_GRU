from __future__ import annotations

"""1차원 도로(3차선) 위 차량/보행자 운동 시뮬레이션과 시나리오 정의."""

import math
import random
from dataclasses import dataclass, field
from typing import Any

from ..features.base import Detection, FrameDetections
from .camera import CameraConfig, DetectorNoiseConfig, project_actor, simulate_detections
from .labels import ContextLabeler, FrameState, LabelThresholds

LANE_W = 3.5

SCENARIO_TYPES: dict[str, float] = {
    "free_drive": 0.14,
    "follow": 0.14,
    "lead_brake": 0.24,
    "cut_in": 0.14,
    "vru_crossing": 0.16,
    "dense_traffic": 0.10,
    "stop_and_go": 0.08,
}


@dataclass(frozen=True, slots=True)
class DomainProfile:
    """도메인(주행 / 실내 이동로봇)별 물리·카메라·라벨 파라미터."""

    name: str
    lane_w: float
    sizes: dict[str, tuple[float, float]]
    vehicle_length: float
    cam_height_m: float
    cam_behind_front_m: float
    max_range_m: float
    t_head: tuple[float, float]
    a_max: tuple[float, float]
    b_comf: tuple[float, float]
    s0: tuple[float, float]
    delay_s: tuple[float, float]
    ego_brake_max: float
    path_half_vehicle: float
    path_half_vru: float
    nearby_range_m: float
    nearby_lateral_m: float
    actor_idm: tuple[float, float, float, float]  # a, b, T, s0
    thresholds: LabelThresholds
    label_names: tuple[str, ...]


DRIVING = DomainProfile(
    name="driving",
    lane_w=3.5,
    sizes={"vehicle": (1.8, 1.5), "person": (0.5, 1.7), "bike": (0.6, 1.7)},
    vehicle_length=4.5,
    cam_height_m=1.3,
    cam_behind_front_m=1.5,
    max_range_m=120.0,
    t_head=(1.0, 1.8),
    a_max=(1.5, 2.5),
    b_comf=(2.0, 3.0),
    s0=(2.0, 3.0),
    delay_s=(0.3, 1.0),
    ego_brake_max=8.0,
    path_half_vehicle=1.7,
    path_half_vru=1.5,
    nearby_range_m=35.0,
    nearby_lateral_m=5.5,
    actor_idm=(1.5, 3.0, 1.2, 2.0),
    thresholds=LabelThresholds(),
    label_names=(
        "normal_drive",
        "front_vehicle_follow",
        "brake_warning",
        "hard_brake_risk",
        "post_brake_recovery",
        "dense_traffic",
    ),
)

# 실내 물류 이동로봇(AMR): vehicle=지게차/AGV, person=작업자, bike=대차(cart)로 해석한다.
ROBOT = DomainProfile(
    name="robot",
    lane_w=2.0,
    sizes={"vehicle": (1.2, 2.0), "person": (0.5, 1.7), "bike": (0.7, 1.0)},
    vehicle_length=2.5,
    cam_height_m=0.8,
    cam_behind_front_m=0.3,
    max_range_m=30.0,
    t_head=(0.8, 1.5),
    a_max=(0.5, 1.0),
    b_comf=(0.8, 1.2),
    s0=(0.5, 1.0),
    delay_s=(0.1, 0.3),
    ego_brake_max=2.5,
    path_half_vehicle=0.9,
    path_half_vru=0.8,
    nearby_range_m=6.0,
    nearby_lateral_m=3.0,
    actor_idm=(0.6, 1.0, 1.0, 0.5),
    thresholds=LabelThresholds(
        hard_ttc_s=1.5,
        hard_decel=1.2,
        warn_ttc_s=3.0,
        warn_decel=0.6,
        follow_headway_s=3.0,
        follow_gap_m=4.0,
        dense_count=3,
        dense_range_m=6.0,
        dense_speed=0.8,
        min_closing=0.2,
    ),
    label_names=("normal_move", "follow_agent", "slow_down", "safety_stop", "resume", "crowded"),
)


def profile_for_scenario(scenario: str) -> DomainProfile:
    return ROBOT if scenario.startswith("robot_") else DRIVING


@dataclass(slots=True)
class Actor:
    """주변 객체. x는 차량 후미(보행자는 중심) 종방향 위치, y는 횡위치(좌측 +)."""

    kind: str
    x: float
    y: float
    v: float
    v_des: float
    acc_max: float = 1.5
    brake_max: float = 3.0
    vy: float = 0.0
    y_target: float | None = None
    lat_speed: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)
    follows: bool = True

    def apply_events(self, t: float, ego_x: float = 0.0) -> None:
        """시간 조건(t) 또는 자차와의 거리 조건(gap_lt)을 만족한 이벤트를 적용한다."""

        while self.events:
            ev = self.events[0]
            if ev.get("t", 0.0) > t:
                break
            if "gap_lt" in ev and (self.x - ego_x) >= ev["gap_lt"]:
                break
            self.events.pop(0)
            for key in ("v_des", "brake_max", "acc_max", "y_target", "lat_speed"):
                if key in ev:
                    setattr(self, key, ev[key])


@dataclass(slots=True)
class Episode:
    """한 에피소드의 프레임별 검출/라벨/GT 상태."""

    scenario: str
    fps: float
    frames: list[FrameDetections]
    labels: list[int]
    ego_v: list[float]
    ego_a: list[float]
    ttc: list[float]
    meta: dict[str, Any]


def _idm(v: float, v0: float, gap: float, dv: float, a: float, b: float, t_head: float, s0: float) -> float:
    s_star = s0 + max(0.0, v * t_head + v * dv / (2.0 * math.sqrt(a * b)))
    free = 1.0 - (v / max(0.1, v0)) ** 4
    inter = (s_star / max(0.1, gap)) ** 2 if math.isfinite(gap) else 0.0
    return a * (free - inter)


def _lane_y(lane: int) -> float:
    return lane * LANE_W


def _build_scenario(kind: str, rng: random.Random, duration: float) -> tuple[float, list[Actor], dict[str, Any]]:
    """시나리오별 자차 목표속도와 주변 객체 목록을 만든다."""

    if kind.startswith("robot_"):
        return _build_robot_scenario(kind, rng, duration)
    actors: list[Actor] = []
    v0 = rng.uniform(12.0, 25.0)
    meta: dict[str, Any] = {}

    def add_background(n_adj: int, n_side: int, speed_ref: float) -> None:
        for _ in range(n_adj):
            lane = rng.choice([-1, 1])
            v = max(0.0, speed_ref + rng.uniform(-5.0, 5.0))
            actors.append(Actor("vehicle", rng.uniform(-30.0, 90.0), _lane_y(lane) + rng.uniform(-0.3, 0.3), v, v))
        for _ in range(n_side):
            kind_side = "person" if rng.random() < 0.7 else "bike"
            side = rng.choice([-1, 1]) * rng.uniform(5.5, 8.0)
            v = rng.uniform(0.0, 1.6) if kind_side == "person" else rng.uniform(2.0, 5.0)
            actors.append(Actor(kind_side, rng.uniform(5.0, 70.0), side, v, v, follows=False))
        # 길가 주차 차량
        for _ in range(rng.randint(0, 3)):
            actors.append(Actor("vehicle", rng.uniform(10.0, 100.0), rng.choice([-1, 1]) * rng.uniform(5.2, 6.0), 0.0, 0.0, follows=False))

    if kind == "free_drive":
        add_background(rng.randint(1, 4), rng.randint(0, 3), v0)
    elif kind in {"follow", "lead_brake", "stop_and_go"}:
        v_lead = v0 * rng.uniform(0.75, 1.0)
        gap = v_lead * rng.uniform(1.2, 3.0) + 5.0
        lead = Actor("vehicle", gap, rng.uniform(-0.3, 0.3), v_lead, v_lead)
        if kind == "follow":
            for t_ev in range(5, int(duration), rng.randint(5, 9)):
                lead.events.append({"t": float(t_ev), "v_des": max(3.0, v_lead * rng.uniform(0.7, 1.15)), "brake_max": 1.5})
        elif kind == "lead_brake":
            t_b = rng.uniform(6.0, max(7.0, duration - 12.0))
            decel = rng.uniform(3.0, 8.0)
            v_target = v_lead * rng.uniform(0.0, 0.5)
            hold = rng.uniform(1.5, 5.0)
            lead.events.append({"t": t_b, "v_des": v_target, "brake_max": decel})
            lead.events.append({"t": t_b + v_lead / decel + hold, "v_des": v_lead, "acc_max": rng.uniform(1.0, 2.5)})
            meta.update({"lead_brake_t": t_b, "lead_decel": decel})
        else:  # stop_and_go
            t_s = rng.uniform(4.0, 10.0)
            while t_s < duration - 4.0:
                lead.events.append({"t": t_s, "v_des": 0.0, "brake_max": rng.uniform(1.5, 3.0)})
                t_go = t_s + rng.uniform(4.0, 8.0)
                lead.events.append({"t": t_go, "v_des": rng.uniform(6.0, 12.0), "acc_max": rng.uniform(1.0, 2.0)})
                t_s = t_go + rng.uniform(5.0, 9.0)
        actors.append(lead)
        add_background(rng.randint(0, 3), rng.randint(0, 2), v0)
    elif kind == "cut_in":
        lane = rng.choice([-1, 1])
        v_c = v0 * rng.uniform(0.65, 0.95)
        # 옆 차선 전방에서 출발해 자차와의 거리가 gap_c 미만이 되면 끼어든다.
        gap_c = rng.uniform(8.0, 25.0)
        cutter = Actor("vehicle", rng.uniform(25.0, 55.0), _lane_y(lane), v_c, v_c, follows=False)
        cutter.events.append({"t": 3.0, "gap_lt": gap_c, "y_target": rng.uniform(-0.3, 0.3), "lat_speed": rng.uniform(0.8, 1.8)})
        if rng.random() < 0.35:
            cutter.events.append({"t": 0.0, "gap_lt": gap_c * 0.8, "v_des": v_c * rng.uniform(0.3, 0.8), "brake_max": rng.uniform(2.0, 5.0)})
        actors.append(cutter)
        meta["cut_in_gap"] = gap_c
        add_background(rng.randint(0, 2), rng.randint(0, 2), v0)
    elif kind == "vru_crossing":
        v0 = rng.uniform(8.0, 16.0)
        kind_vru = "person" if rng.random() < 0.65 else "bike"
        speed = rng.uniform(1.0, 1.8) if kind_vru == "person" else rng.uniform(2.5, 5.0)
        side = rng.choice([-1, 1])
        start_y = side * rng.uniform(4.0, 6.0)
        # 자차와의 거리가 dist 미만이 되면 횡단 시작
        dist = rng.uniform(25.0, 60.0)
        vru = Actor(kind_vru, rng.uniform(70.0, 140.0), start_y, 0.0, 0.0, follows=False)
        vru.events.append({"t": 2.0, "gap_lt": dist, "y_target": -side * rng.uniform(4.0, 6.0), "lat_speed": speed})
        actors.append(vru)
        meta["cross_gap"] = dist
        add_background(rng.randint(0, 2), rng.randint(1, 4), v0)
    elif kind == "dense_traffic":
        v0 = rng.uniform(6.0, 12.0)
        for lane in (-1, 0, 1):
            x = rng.uniform(6.0, 14.0) if lane == 0 else rng.uniform(-10.0, 5.0)
            for _ in range(rng.randint(4, 7)):
                v = rng.uniform(1.0, 8.0)
                act = Actor("vehicle", x, _lane_y(lane) + rng.uniform(-0.3, 0.3), v, v)
                for t_ev in range(3, int(duration), rng.randint(3, 6)):
                    act.events.append({"t": float(t_ev), "v_des": rng.uniform(0.0, 9.0), "brake_max": rng.uniform(1.0, 3.0)})
                actors.append(act)
                x += rng.uniform(7.0, 16.0)
        add_background(0, rng.randint(0, 3), v0)
    else:
        raise KeyError(f"알 수 없는 시나리오: {kind}")
    return v0, actors, meta


def simulate_episode(
    scenario: str,
    seed: int,
    fps: float = 15.0,
    duration_s: float = 30.0,
    cam: CameraConfig | None = None,
    noise: DetectorNoiseConfig | None = None,
    thresholds: LabelThresholds | None = None,
    physics_hz: float = 30.0,
) -> Episode:
    """시나리오 1개를 시뮬레이션해 프레임별 검출/라벨을 만든다."""

    rng = random.Random(seed)
    prof = profile_for_scenario(scenario)
    cam = cam or CameraConfig(cam_height_m=prof.cam_height_m, max_range_m=prof.max_range_m)
    noise = noise or DetectorNoiseConfig()
    labeler = ContextLabeler(thresholds or prof.thresholds)
    v0, actors, meta = _build_scenario(scenario, rng, duration_s)

    substeps = max(1, int(math.ceil(physics_hz / fps)))
    dt = 1.0 / (fps * substeps)
    n_frames = int(round(duration_s * fps))

    # 자차 IDM 파라미터(운전자 성향 무작위화)
    t_head = rng.uniform(*prof.t_head)
    a_max = rng.uniform(*prof.a_max)
    b_comf = rng.uniform(*prof.b_comf)
    s0 = rng.uniform(*prof.s0)
    delay_steps = max(0, int(round(rng.uniform(*prof.delay_s) / dt)))
    cmd_queue: list[float] = [0.0] * delay_steps

    ego_x, ego_v, ego_a = 0.0, v0 * rng.uniform(0.85, 1.0), 0.0
    pitch_noise = 0.0
    collisions = 0
    frames: list[FrameDetections] = []
    labels: list[int] = []
    ego_vs: list[float] = []
    ego_as: list[float] = []
    ttcs: list[float] = []

    def in_path(actor: Actor, horizon_s: float = 2.5) -> bool:
        half = prof.path_half_vehicle if actor.kind == "vehicle" else prof.path_half_vru
        if abs(actor.y) < half:
            return True
        if actor.kind != "vehicle" and actor.y_target is not None and actor.lat_speed > 0:
            direction = math.copysign(1.0, actor.y_target - actor.y)
            future_y = actor.y + direction * actor.lat_speed * horizon_s
            return (actor.y > 0) != (future_y > 0) or abs(future_y) < half
        return False

    def path_obstacle() -> tuple[float, float, bool]:
        best_gap, best_v, best_vehicle = math.inf, 0.0, False
        for actor in actors:
            gap = actor.x - ego_x
            if gap <= -0.5 or not in_path(actor):
                continue
            if gap < best_gap:
                best_gap, best_v, best_vehicle = gap, actor.v, actor.kind == "vehicle"
        return best_gap, best_v, best_vehicle

    step = 0
    for frame_idx in range(n_frames):
        for _ in range(substeps):
            t = step * dt
            # --- 주변 객체 운동
            for actor in actors:
                actor.apply_events(t, ego_x)
                acc = max(-actor.brake_max, min(actor.acc_max, 0.8 * (actor.v_des - actor.v)))
                if actor.follows and actor.kind == "vehicle":
                    lane_tol = 0.43 * prof.lane_w
                    ahead = [o for o in actors if o is not actor and o.kind == "vehicle" and 0.0 < o.x - actor.x and abs(o.y - actor.y) < lane_tol]
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
            # --- 자차 IDM + 반응 지연
            gap, v_obs, _ = path_obstacle()
            cmd = _idm(ego_v, v0, gap, ego_v - v_obs, a_max, b_comf, t_head, s0)
            cmd = max(-prof.ego_brake_max, min(a_max, cmd))
            cmd_queue.append(cmd)
            applied = cmd_queue.pop(0)
            # 실제 가속도는 1차 지연(브레이크/구동 응답) 적용
            ego_a += (applied - ego_a) * min(1.0, dt / 0.3)
            ego_v = max(0.0, ego_v + ego_a * dt)
            if ego_v == 0.0 and ego_a < 0:
                ego_a = 0.0
            ego_x += ego_v * dt
            gap_now, v_obs_now, _ = path_obstacle()
            if gap_now < 0.3:
                collisions += 1
                ego_x = ego_x - (0.3 - gap_now)
                ego_v = min(ego_v, v_obs_now)
            step += 1

        # --- 프레임 샘플링: 투영, 검출, 라벨
        t_frame = step * dt
        pitch_noise = 0.8 * pitch_noise + rng.gauss(0.0, noise.pitch_jitter_px * noise.level)
        horizon_y = cam.horizon_ratio * cam.height - cam.pitch_px_per_mps2 * (-ego_a) + pitch_noise
        projected = []
        for idx, actor in enumerate(actors):
            depth = actor.x - (ego_x - prof.cam_behind_front_m)
            width_m, height_m = prof.sizes[actor.kind]
            box = project_actor(cam, idx, actor.kind, depth, actor.y, width_m, height_m, horizon_y)
            if box is not None:
                projected.append(box)
        dets: list[Detection] = simulate_detections(projected, cam, noise, rng)
        frames.append(FrameDetections(frame_idx, round(frame_idx / fps, 6), cam.width, cam.height, dets))

        gap, v_obs, is_vehicle = path_obstacle()
        nearby = sum(
            1
            for a in actors
            if (a.kind == "vehicle" or prof.name == "robot")
            and 0.0 < a.x - ego_x < prof.nearby_range_m
            and abs(a.y) < prof.nearby_lateral_m
        )
        state = FrameState(t_frame, ego_v, ego_a, gap, ego_v - v_obs, is_vehicle, nearby)
        labels.append(labeler(state))
        ego_vs.append(round(ego_v, 4))
        ego_as.append(round(ego_a, 4))
        ttc = labeler.ttc(state)
        ttcs.append(round(ttc, 4) if math.isfinite(ttc) else -1.0)

    meta.update(
        {
            "seed": seed,
            "v0": round(v0, 3),
            "t_head": round(t_head, 3),
            "reaction_delay_s": round(delay_steps * dt, 3),
            "collisions_steps": collisions,
            "noise_level": noise.level,
            "domain": prof.name,
        }
    )
    return Episode(scenario, fps, frames, labels, ego_vs, ego_as, ttcs, meta)


def scenario_types_for_domain(domain: str = "driving") -> dict[str, float]:
    """도메인별 시나리오 가중치 표를 반환한다."""

    key = str(domain).strip().lower()
    if key == "driving":
        return dict(SCENARIO_TYPES)
    if key == "robot":
        return dict(ROBOT_SCENARIO_TYPES)
    raise KeyError(f"지원하지 않는 도메인: {domain}")


ROBOT_SCENARIO_TYPES: dict[str, float] = {
    "robot_aisle_free": 0.15,
    "robot_follow_agent": 0.15,
    "robot_agent_stop": 0.2,
    "robot_human_crossing": 0.2,
    "robot_human_headon": 0.15,
    "robot_crowded": 0.15,
}


def _build_robot_scenario(kind: str, rng: random.Random, duration: float) -> tuple[float, list[Actor], dict[str, Any]]:
    """실내 물류 통로(폭 2m 레인 3개)에서의 AMR 시나리오."""

    lane = ROBOT.lane_w
    v0 = rng.uniform(0.8, 1.6)
    actors: list[Actor] = []
    meta: dict[str, Any] = {}

    def background(n_people: int, n_parked: int) -> None:
        for _ in range(n_people):
            v = rng.uniform(-1.2, 1.2)
            actors.append(Actor("person", rng.uniform(3.0, 25.0), rng.choice([-1, 1]) * rng.uniform(1.6, 3.0), v, v, follows=False))
        for _ in range(n_parked):
            kind_p = "bike" if rng.random() < 0.6 else "vehicle"
            actors.append(Actor(kind_p, rng.uniform(3.0, 28.0), rng.choice([-1, 1]) * rng.uniform(1.4, 2.6), 0.0, 0.0, follows=False))

    if kind == "robot_aisle_free":
        background(rng.randint(0, 3), rng.randint(1, 4))
    elif kind in {"robot_follow_agent", "robot_agent_stop"}:
        v_l = v0 * rng.uniform(0.6, 0.95)
        agent = Actor("vehicle", v_l * rng.uniform(1.2, 2.5) + 1.0, rng.uniform(-0.15, 0.15), v_l, v_l, acc_max=0.6, brake_max=0.8)
        if kind == "robot_agent_stop":
            t_s = rng.uniform(5.0, max(6.0, duration - 10.0))
            agent.events.append({"t": t_s, "v_des": 0.0, "brake_max": rng.uniform(1.5, 3.0)})
            agent.events.append({"t": t_s + rng.uniform(2.0, 6.0), "v_des": v_l, "acc_max": 0.6})
            meta["agent_stop_t"] = t_s
        else:
            for t_ev in range(5, int(duration), rng.randint(5, 9)):
                agent.events.append({"t": float(t_ev), "v_des": max(0.3, v_l * rng.uniform(0.7, 1.2)), "brake_max": 0.5})
        actors.append(agent)
        background(rng.randint(0, 2), rng.randint(0, 3))
    elif kind == "robot_human_crossing":
        side = rng.choice([-1, 1])
        kind_c = "person" if rng.random() < 0.8 else "bike"
        speed = rng.uniform(0.8, 1.5) if kind_c == "person" else rng.uniform(0.5, 1.0)
        dist = rng.uniform(3.0, 8.0)
        crosser = Actor(kind_c, rng.uniform(12.0, 30.0), side * rng.uniform(2.0, 3.0), 0.0, 0.0, follows=False)
        crosser.events.append({"t": 2.0, "gap_lt": dist, "y_target": -side * rng.uniform(2.0, 3.0), "lat_speed": speed})
        actors.append(crosser)
        meta["cross_gap"] = dist
        background(rng.randint(0, 2), rng.randint(0, 3))
    elif kind == "robot_human_headon":
        # 같은 통로에서 마주 오는 작업자: 일정 거리에서 옆으로 비켜선다.
        walker = Actor("person", rng.uniform(15.0, 28.0), rng.uniform(-0.3, 0.3), -rng.uniform(0.8, 1.4), -1.0, acc_max=0.8, brake_max=1.0, follows=False)
        walker.v_des = walker.v
        if rng.random() < 0.7:
            walker.events.append({"t": 0.0, "gap_lt": rng.uniform(2.0, 5.0), "y_target": rng.choice([-1, 1]) * rng.uniform(1.2, 1.8), "lat_speed": rng.uniform(0.6, 1.0)})
        else:
            walker.events.append({"t": 0.0, "gap_lt": rng.uniform(2.0, 4.0), "v_des": 0.0, "brake_max": 1.5})
        actors.append(walker)
        background(rng.randint(0, 2), rng.randint(0, 3))
    elif kind == "robot_crowded":
        v0 = rng.uniform(0.4, 0.8)
        for _ in range(rng.randint(4, 8)):
            v = rng.uniform(-1.0, 1.0)
            y = rng.uniform(-2.5, 2.5)
            p = Actor("person", rng.uniform(1.5, 10.0), y, v, v, follows=False)
            if rng.random() < 0.5:
                p.events.append({"t": rng.uniform(1.0, duration), "y_target": rng.uniform(-2.5, 2.5), "lat_speed": rng.uniform(0.3, 0.8)})
            actors.append(p)
        background(0, rng.randint(2, 5))
    else:
        raise KeyError(f"알 수 없는 로봇 시나리오: {kind}")
    _ = lane
    return v0, actors, meta
