from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class InstantRuleParams:
    """현재 프레임 YOLO 박스만으로 이벤트를 판정하는 규칙 파라미터."""

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
    return max(0.0, min(1.0, float(value)))


def _box_metrics(box: dict[str, Any], width: float, height: float) -> dict[str, float | str]:
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
    """현재 프레임 검출 박스만으로 순간 규칙형 판단 결과를 반환한다."""

    cfg = params or InstantRuleParams()
    orig_shape = detection.get("orig_shape") or [720, 1280]
    try:
        height = float(orig_shape[0])
        width = float(orig_shape[1])
    except (TypeError, ValueError, IndexError):
        height = 720.0
        width = 1280.0

    boxes = detection.get("boxes", []) if isinstance(detection, dict) else []
    metrics = [_box_metrics(box, width, height) for box in boxes if isinstance(box, dict)]

    person_scores: list[float] = []
    bike_scores: list[float] = []
    vehicle_scores: list[float] = []
    reasons: list[str] = []

    person_count = 0
    bike_count = 0
    vehicle_count = 0

    for item in metrics:
        cls_name = str(item["cls_name"])
        area_ratio = float(item["area_ratio"])
        cx = float(item["center_x_norm"])
        cy = float(item["center_y_norm"])
        center_score = float(item["center_score"])

        in_roi = cfg.roi_x1 <= cx <= cfg.roi_x2 and cfg.roi_y1 <= cy <= cfg.roi_y2
        in_inner = cfg.inner_x1 <= cx <= cfg.inner_x2 and cfg.inner_y1 <= cy <= cfg.inner_y2
        if cls_name == "person":
            person_count += 1
            if in_roi:
                person_scores.append((0.55 * center_score) + (8.0 * area_ratio))
            if in_inner and area_ratio >= cfg.warn_person_area_thr:
                reasons.append("person_roi")
        elif cls_name == "bike":
            bike_count += 1
            if in_roi:
                bike_scores.append((0.50 * center_score) + (7.5 * area_ratio))
            if in_inner and area_ratio >= cfg.warn_bike_area_thr:
                reasons.append("bike_roi")
        elif cls_name == "vehicle":
            vehicle_count += 1
            if in_roi:
                vehicle_scores.append((0.45 * center_score) + (6.0 * area_ratio))
            if in_inner and area_ratio >= cfg.follow_vehicle_area_thr:
                reasons.append("vehicle_roi")

    max_person = max(person_scores, default=0.0)
    max_bike = max(bike_scores, default=0.0)
    max_vehicle = max(vehicle_scores, default=0.0)

    warn_score = max(max_person, max_bike, max_vehicle)
    hard_score = 0.0

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

    label = "normal_drive"
    if vehicle_count >= cfg.dense_vehicle_thr and int(detection.get("count", 0)) >= cfg.dense_det_thr:
        label = "dense_traffic"
    if max_vehicle >= ((0.45 * 0.70) + (6.0 * cfg.follow_vehicle_area_thr)):
        label = "front_vehicle_follow"
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
