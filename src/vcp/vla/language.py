from __future__ import annotations

"""맥락 라벨 → 자연어 서술 템플릿(한국어/영어).

VLA 학습에서 '언어' 축은 보통 에피소드 단위 과업 지시문만 쓰지만, 최근에는 프레임 단위
추론 서술(embodied chain-of-thought)을 함께 쓰는 연구가 늘고 있다. 여기서는 경량 맥락
모델의 출력과 물리량으로 그런 서술을 자동 생성한다. 템플릿 기반이므로 표현 다양성은
제한적이며, 대규모 언어모델 기반 서술 확장은 향후 과제로 둔다.
"""

import math

_CONTEXT_TEXT: dict[str, tuple[str, str]] = {
    # 주행 도메인
    "normal_drive": ("진행 경로에 특이 객체가 없어 차선을 유지하며 주행한다.", "The path ahead is clear; keep the lane."),
    "front_vehicle_follow": ("전방 차량을 일정 거리로 추종하고 있다.", "Following the lead vehicle at a steady gap."),
    "brake_warning": ("전방 객체와의 거리가 빠르게 줄고 있어 감속이 필요하다.", "The gap to the object ahead is closing; slow down."),
    "hard_brake_risk": ("충돌 위험이 커서 강하게 제동해야 한다.", "Collision risk is high; brake hard."),
    "post_brake_recovery": ("강한 제동 이후 속도를 회복하고 있다.", "Recovering speed after hard braking."),
    "dense_traffic": ("주변 차량이 밀집해 저속으로 주행한다.", "Dense traffic around; proceed slowly."),
    # 실내 이동로봇 도메인
    "normal_move": ("통로가 비어 있어 계획 경로를 따라 이동한다.", "The aisle is clear; follow the planned path."),
    "follow_agent": ("앞선 운반 차량을 안전거리로 따라간다.", "Following the vehicle ahead at a safe distance."),
    "slow_down": ("작업자나 차량이 경로에 접근해 감속한다.", "A person or vehicle is entering the path; slow down."),
    "safety_stop": ("충돌 위험이 있어 안전 정지한다.", "Collision risk; perform a safety stop."),
    "resume": ("위험이 해소되어 이동을 재개한다.", "Risk cleared; resume moving."),
    "crowded": ("주변에 사람이 많아 저속으로 이동한다.", "Crowded area; move slowly."),
    # 실데이터(자차 운동 상태)
    "cruise": ("일정한 속도로 주행 중이다.", "Cruising at a steady speed."),
    "accelerating": ("가속 중이다.", "Accelerating."),
    "braking": ("감속(제동) 중이다.", "Braking."),
    "stopped": ("정지해 있다.", "Stopped."),
}

_TASKS: dict[str, tuple[str, str]] = {
    "driving": (
        "차선을 유지하며 전방 객체와 안전거리를 확보해 주행하라.",
        "Drive along the lane while keeping a safe distance from objects ahead.",
    ),
    "robot": (
        "물류 통로를 따라 이동하되 작업자와 운반 차량에 대해 안전거리를 유지하라.",
        "Navigate the warehouse aisle while keeping a safe distance from workers and vehicles.",
    ),
    "real_driving": (
        "도로를 따라 안전하게 주행하라.",
        "Drive safely along the road.",
    ),
}


def task_instruction(domain: str) -> tuple[str, str]:
    return _TASKS.get(domain, _TASKS["driving"])


def action_phrase(accel: float, domain: str = "driving") -> tuple[str, str]:
    """가감속 값을 이산 행동 서술로 바꾼다(도메인별 스케일)."""

    scale = 0.3 if domain == "robot" else 1.0
    a = accel / scale
    if a <= -3.0:
        return "급감속", "brake hard"
    if a <= -0.6:
        return "감속", "slow down"
    if a >= 0.6:
        return "가속", "speed up"
    return "속도 유지", "keep speed"


def narrate(
    context: str,
    speed: float | None = None,
    ttc: float | None = None,
    confidence: float | None = None,
) -> tuple[str, str]:
    """맥락 + 물리량 → (한국어, 영어) 서술."""

    ko, en = _CONTEXT_TEXT.get(context, (f"맥락: {context}", f"Context: {context}"))
    extra_ko, extra_en = [], []
    if speed is not None and math.isfinite(speed):
        extra_ko.append(f"현재 속도 {speed:.1f}")
        extra_en.append(f"speed {speed:.1f}")
    if ttc is not None and math.isfinite(ttc) and ttc > 0:
        extra_ko.append(f"예상 충돌시간 {ttc:.1f}초")
        extra_en.append(f"TTC {ttc:.1f}s")
    if confidence is not None:
        extra_ko.append(f"모델 확신도 {confidence:.2f}")
        extra_en.append(f"confidence {confidence:.2f}")
    if extra_ko:
        ko = f"{ko} ({', '.join(extra_ko)})"
        en = f"{en} ({', '.join(extra_en)})"
    return ko, en
