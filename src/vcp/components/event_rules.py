from __future__ import annotations

"""현재 프레임의 YOLO 박스만으로 즉시 상황을 판정하는 규칙 모듈.

이 파일은 3개 비교 모델 중 1번 모델(`YOLO + 규칙`)의 핵심 구현이다.
시계열 모델을 전혀 사용하지 않고 현재 프레임의 박스 위치, 크기, 종류만으로
`normal_drive`, `front_vehicle_follow`, `brake_warning`, `hard_brake_risk`,
`dense_traffic`를 빠르게 추정한다.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class InstantRuleParams:
    """현재 프레임 YOLO 박스만으로 이벤트를 판정하는 규칙 파라미터.

    좌표 값은 화면을 0~1로 정규화했을 때의 비율이다.
    예를 들어 `roi_x1=0.30`은 화면 너비의 30% 지점부터 관심 영역이 시작된다는 뜻이다.
    """

    roi_x1: float = 0.30
    roi_x2: float = 0.70
    roi_y1: float = 0.18
    roi_y2: float = 0.95
    inner_x1: float = 0.40
    inner_x2: float = 0.60
    inner_y1: float = 0.18
    inner_y2: float = 0.95
    follow_vehicle_area_thr: float = 0.010
    warn_vehicle_area_thr: float = 0.030
    hard_vehicle_area_thr: float = 0.070
    warn_person_area_thr: float = 0.010
    hard_person_area_thr: float = 0.020
    warn_bike_area_thr: float = 0.012
    hard_bike_area_thr: float = 0.025
    dense_det_thr: int = 5
    dense_vehicle_thr: int = 3


def _safe_ratio(value: float) -> float:
    """비율 값이 0~1 범위를 넘지 않도록 자른다."""
    return max(0.0, min(1.0, float(value)))


def _box_metrics(box: dict[str, Any], width: float, height: float) -> dict[str, float | str]:
    """YOLO 박스 하나를 규칙 계산에 바로 쓸 수 있는 값으로 정리한다.

    반환 값:
    - area_ratio: 화면 전체 대비 박스 면적 비율
    - center_x_norm, center_y_norm: 박스 중심의 정규화 좌표
    - center_score: 화면 중앙에 가까울수록 커지는 점수
    """
    xyxy = box.get("xyxy", [])
    if not isinstance(xyxy, list) or len(xyxy) != 4:
        return {
            "cls_name": str(box.get("cls_name", "unknown")),
            "area_ratio": 0.0,
            "center_x_norm": 0.0,
            "center_y_norm": 0.0,
            "center_score": 0.0,
        }

    x1, y1, x2, y2 = [float(item) for item in xyxy]
    box_w = max(0.0, x2 - x1)
    box_h = max(0.0, y2 - y1)
    area_ratio = (box_w * box_h) / max(1.0, width * height)
    center_x = ((x1 + x2) * 0.5) / max(1.0, width)
    center_y = ((y1 + y2) * 0.5) / max(1.0, height)
    center_score = max(0.0, 1.0 - (abs(center_x - 0.5) / 0.5))
    return {
        "cls_name": str(box.get("cls_name", "unknown")),
        "area_ratio": _safe_ratio(area_ratio),
        "center_x_norm": _safe_ratio(center_x),
        "center_y_norm": _safe_ratio(center_y),
        "center_score": _safe_ratio(center_score),
    }


def predict_instant_rule(
    detection: dict[str, Any],
    params: InstantRuleParams | None = None,
) -> dict[str, Any]:
    """현재 프레임의 검출 박스만으로 순간 규칙 판단 결과를 반환한다.

    동작 순서:
    1. 박스별 면적/중심 정보를 정규화한다.
    2. ROI 내부의 사람/이륜차/차량 점수를 따로 계산한다.
    3. warning/hard 점수를 만든다.
    4. 점수와 규칙 조건을 조합해 최종 라벨을 정한다.
    """

    # 별도 설정이 없으면 기본 규칙 세트를 사용한다.
    cfg = params or InstantRuleParams()

    # 원본 이미지 크기를 읽어 박스 좌표를 정규화할 준비를 한다.
    orig_shape = detection.get("orig_shape") or [720, 1280]
    try:
        height = float(orig_shape[0])
        width = float(orig_shape[1])
    except (TypeError, ValueError, IndexError):
        height = 720.0
        width = 1280.0

    # YOLO가 만든 박스 목록을 규칙 계산용 요약 지표로 바꾼다.
    boxes = detection.get("boxes", []) if isinstance(detection, dict) else []
    metrics = [_box_metrics(box, width, height) for box in boxes if isinstance(box, dict)]

    # 클래스별 위험 점수를 따로 누적하면 어떤 객체가 경고를 만들었는지 해석하기 쉽다.
    person_scores: list[float] = []
    bike_scores: list[float] = []
    vehicle_scores: list[float] = []
    reasons: list[str] = []

    person_count = 0
    bike_count = 0
    vehicle_count = 0

    for item in metrics:
        # 현재 박스의 핵심 값만 꺼내면 아래 규칙이 읽기 쉬워진다.
        cls_name = str(item["cls_name"])
        area_ratio = float(item["area_ratio"])
        cx = float(item["center_x_norm"])
        cy = float(item["center_y_norm"])
        center_score = float(item["center_score"])

        # 넓은 ROI는 관심 영역, inner ROI는 더 위험한 중앙 영역으로 사용한다.
        in_roi = cfg.roi_x1 <= cx <= cfg.roi_x2 and cfg.roi_y1 <= cy <= cfg.roi_y2
        in_inner = cfg.inner_x1 <= cx <= cfg.inner_x2 and cfg.inner_y1 <= cy <= cfg.inner_y2

        # 사람은 전방 중앙에 등장하면 바로 위험도가 커지도록 가중치를 높게 둔다.
        if cls_name == "person":
            person_count += 1
            if in_roi:
                person_scores.append((0.55 * center_score) + (8.0 * area_ratio))
            if in_inner and area_ratio >= cfg.warn_person_area_thr:
                reasons.append("person_roi")

        # 이륜차도 차량보다 빠른 반응이 필요하므로 비슷한 방식으로 처리한다.
        elif cls_name == "bike":
            bike_count += 1
            if in_roi:
                bike_scores.append((0.50 * center_score) + (7.5 * area_ratio))
            if in_inner and area_ratio >= cfg.warn_bike_area_thr:
                reasons.append("bike_roi")

        # 차량은 전방 추종과 브레이크 경고를 모두 만들 수 있다.
        elif cls_name == "vehicle":
            vehicle_count += 1
            if in_roi:
                vehicle_scores.append((0.45 * center_score) + (6.0 * area_ratio))
            if in_inner and area_ratio >= cfg.follow_vehicle_area_thr:
                reasons.append("vehicle_roi")

    # 각 클래스별 최고 점수만 남기면 현재 프레임에서 가장 위험한 객체를 알 수 있다.
    max_person = max(person_scores, default=0.0)
    max_bike = max(bike_scores, default=0.0)
    max_vehicle = max(vehicle_scores, default=0.0)

    # warning 점수는 사람/이륜차/차량 중 가장 위험한 쪽을 사용한다.
    warn_score = max(max_person, max_bike, max_vehicle)
    hard_score = 0.0

    # hard brake는 중앙에 크게 들어온 객체만 별도로 다시 검사한다.
    for item in metrics:
        cls_name = str(item["cls_name"])
        area_ratio = float(item["area_ratio"])
        cx = float(item["center_x_norm"])
        cy = float(item["center_y_norm"])
        center_score = float(item["center_score"])
        in_inner = cfg.inner_x1 <= cx <= cfg.inner_x2 and cfg.inner_y1 <= cy <= cfg.inner_y2
        if not in_inner:
            continue

        if cls_name == "person" and area_ratio >= cfg.hard_person_area_thr:
            hard_score = max(hard_score, (0.7 * center_score) + (12.0 * area_ratio))
            reasons.append("person_hard")
        elif cls_name == "bike" and area_ratio >= cfg.hard_bike_area_thr:
            hard_score = max(hard_score, (0.65 * center_score) + (11.0 * area_ratio))
            reasons.append("bike_hard")
        elif cls_name == "vehicle" and area_ratio >= cfg.hard_vehicle_area_thr:
            hard_score = max(hard_score, (0.55 * center_score) + (10.0 * area_ratio))
            reasons.append("vehicle_hard")

    # 기본값은 normal_drive로 두고, 강한 조건이 보이면 단계적으로 승격한다.
    label = "normal_drive"

    # 차량이 많고 전체 검출 수도 많으면 dense_traffic로 본다.
    if vehicle_count >= cfg.dense_vehicle_thr and int(detection.get("count", 0)) >= cfg.dense_det_thr:
        label = "dense_traffic"

    # 전방 차량이 중앙 ROI에서 일정 크기 이상이면 앞차 추종 상황으로 해석한다.
    if max_vehicle >= ((0.45 * 0.70) + (6.0 * cfg.follow_vehicle_area_thr)):
        label = "front_vehicle_follow"

    # hard brake는 가장 우선순위가 높다.
    if (
        hard_score > 0.0
        or any(
            str(item["cls_name"]) == "person"
            and float(item["area_ratio"]) >= cfg.hard_person_area_thr
            and cfg.inner_x1 <= float(item["center_x_norm"]) <= cfg.inner_x2
            for item in metrics
        )
    ):
        label = "hard_brake_risk"

    # hard가 아니어도 사람/이륜차/중앙 차량이 경고 조건을 넘으면 brake_warning으로 본다.
    elif (
        any(
            str(item["cls_name"]) in {"person", "bike"}
            and cfg.roi_x1 <= float(item["center_x_norm"]) <= cfg.roi_x2
            and cfg.roi_y1 <= float(item["center_y_norm"]) <= cfg.roi_y2
            and (
                (str(item["cls_name"]) == "person" and float(item["area_ratio"]) >= cfg.warn_person_area_thr)
                or (str(item["cls_name"]) == "bike" and float(item["area_ratio"]) >= cfg.warn_bike_area_thr)
            )
            for item in metrics
        )
        or any(
            str(item["cls_name"]) == "vehicle"
            and cfg.inner_x1 <= float(item["center_x_norm"]) <= cfg.inner_x2
            and float(item["area_ratio"]) >= cfg.warn_vehicle_area_thr
            for item in metrics
        )
    ):
        label = "brake_warning"

    # 반환 값에는 최종 라벨뿐 아니라, 나중에 해석용으로 쓸 보조 점수도 함께 넣는다.
    return {
        "label": label,
        "warn_score": round(_safe_ratio(min(1.0, warn_score)), 6),
        "hard_score": round(_safe_ratio(min(1.0, hard_score)), 6),
        "reason": "+".join(sorted(set(reasons))) if reasons else "none",
        "person_count": person_count,
        "bike_count": bike_count,
        "vehicle_count": vehicle_count,
        "max_person_score": round(_safe_ratio(min(1.0, max_person)), 6),
        "max_bike_score": round(_safe_ratio(min(1.0, max_bike)), 6),
        "max_vehicle_score": round(_safe_ratio(min(1.0, max_vehicle)), 6),
    }
