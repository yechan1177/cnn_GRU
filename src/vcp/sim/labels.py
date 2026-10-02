from __future__ import annotations

"""GT 물리량 기반 6맥락 라벨 규칙.

라벨은 시뮬레이터의 실제 상태(거리, 상대속도, 자차 가감속)에서만 계산하며,
모델 입력 특징(검출 박스 기반)은 전혀 사용하지 않는다. → 라벨/특징 순환 제거.
"""

import math
from dataclasses import dataclass

CONTEXT_LABELS: list[str] = [
    "normal_drive",
    "front_vehicle_follow",
    "brake_warning",
    "hard_brake_risk",
    "post_brake_recovery",
    "dense_traffic",
]
LABEL_INDEX = {name: idx for idx, name in enumerate(CONTEXT_LABELS)}
BRAKE_LABELS = ("brake_warning", "hard_brake_risk")


@dataclass(slots=True)
class LabelThresholds:
    hard_ttc_s: float = 1.8
    hard_decel: float = 4.0
    warn_ttc_s: float = 3.5
    warn_decel: float = 2.0
    follow_headway_s: float = 3.0
    follow_gap_m: float = 25.0
    dense_count: int = 4
    dense_range_m: float = 35.0
    dense_speed: float = 10.0
    recovery_window_s: float = 3.0
    recovery_ratio: float = 0.85
    min_closing: float = 1.0  # TTC 계산 최소 접근 속도(m/s), 정체 중 미세 접근 제외


@dataclass(slots=True)
class FrameState:
    """라벨 계산에 필요한 프레임 단위 GT 상태."""

    t: float
    ego_v: float
    ego_a: float
    path_gap: float  # 진행 경로 위 가장 가까운 객체까지 거리(m), 없으면 inf
    path_closing: float  # 자차 속도 - 객체 종방향 속도
    path_is_vehicle: bool
    nearby_vehicles: int


class ContextLabeler:
    """프레임 순서대로 호출하는 상태 기반 라벨러(회복 구간 추적)."""

    def __init__(self, thr: LabelThresholds | None = None) -> None:
        self.thr = thr or LabelThresholds()
        self._in_event = False
        self._event_hard = False
        self._event_v_pre = 0.0
        self._recovery_until = -1.0
        self._recovery_v_pre = 0.0
        self._speed_hist: list[tuple[float, float]] = []

    def ttc(self, state: FrameState) -> float:
        if math.isinf(state.path_gap) or state.path_closing <= self.thr.min_closing:
            return math.inf
        return max(0.0, state.path_gap) / state.path_closing

    def __call__(self, state: FrameState) -> int:
        thr = self.thr
        ttc = self.ttc(state)
        decel = -state.ego_a
        # 이벤트 시작 직전 속도를 기억하기 위해 최근 2초 최대 속도를 유지
        self._speed_hist.append((state.t, state.ego_v))
        self._speed_hist = [(t, v) for t, v in self._speed_hist if state.t - t <= 2.0]

        hard = ttc < thr.hard_ttc_s or decel > thr.hard_decel
        warn = ttc < thr.warn_ttc_s or decel > thr.warn_decel
        if hard or warn:
            if not self._in_event:
                self._in_event = True
                self._event_hard = False
                self._event_v_pre = max(v for _, v in self._speed_hist)
            self._event_hard = self._event_hard or hard
            return 3 if hard else 2

        if self._in_event:
            self._in_event = False
            # 회복 구간은 강한 제동(hard) 이벤트 직후에만 부여한다(라벨 스키마 정의).
            if self._event_hard:
                self._recovery_until = state.t + thr.recovery_window_s
                self._recovery_v_pre = self._event_v_pre
        if (
            state.t <= self._recovery_until
            and state.ego_a >= -0.5
            and state.ego_v < thr.recovery_ratio * self._recovery_v_pre
        ):
            return 4
        self._recovery_until = -1.0 if state.t > self._recovery_until else self._recovery_until

        if state.nearby_vehicles >= thr.dense_count and state.ego_v < thr.dense_speed:
            return 5
        if state.path_is_vehicle and not math.isinf(state.path_gap):
            headway = state.path_gap / max(0.5, state.ego_v)
            if headway < thr.follow_headway_s or state.path_gap < thr.follow_gap_m:
                return 1
        return 0
