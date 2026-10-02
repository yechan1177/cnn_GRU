from __future__ import annotations

"""핀홀 카메라 투영과 검출기 노이즈 모델."""

import math
import random
from dataclasses import dataclass

from ..features.base import CLS_BIKE, CLS_PERSON, CLS_VEHICLE, Detection

KIND_TO_CLS = {"person": CLS_PERSON, "vehicle": CLS_VEHICLE, "bike": CLS_BIKE}


@dataclass(slots=True)
class CameraConfig:
    """대시캠 근사 파라미터(640x480, 수평 화각 약 63도)."""

    width: int = 640
    height: int = 480
    focal_px: float = 520.0
    cam_height_m: float = 1.3
    horizon_ratio: float = 0.45
    # 자차 가감속에 따른 피치 변화(px / (m/s^2)). 제동 시 화면이 위로 이동한다.
    pitch_px_per_mps2: float = 1.5
    max_range_m: float = 120.0


@dataclass(slots=True)
class DetectorNoiseConfig:
    """검출기 오류 모델. level=1.0이 기본(mid)."""

    level: float = 1.0
    size50_px: float = 12.0  # 이 크기(sqrt 면적)에서 검출 확률 50%
    size_scale_px: float = 4.0
    base_miss: float = 0.05
    jitter_ratio: float = 0.04
    conf_noise: float = 0.07
    class_swap_vru: float = 0.05
    fp_rate: float = 0.15
    pitch_jitter_px: float = 1.5
    output_conf: float = 0.25


NOISE_LEVELS: dict[str, float] = {"low": 0.5, "mid": 1.0, "high": 2.0}


@dataclass(slots=True)
class ProjectedBox:
    actor_id: int
    kind: str
    depth: float
    x1: float
    y1: float
    x2: float
    y2: float


def project_actor(
    cam: CameraConfig,
    actor_id: int,
    kind: str,
    depth_m: float,
    lateral_m: float,
    width_m: float,
    height_m: float,
    horizon_y: float,
) -> ProjectedBox | None:
    """월드 좌표(전방 거리, 좌측 + 횡위치)를 이미지 박스로 투영한다."""

    if depth_m < 2.0 or depth_m > cam.max_range_m:
        return None
    f = cam.focal_px
    cx = 0.5 * cam.width - f * lateral_m / depth_m
    half_w = 0.5 * f * width_m / depth_m
    y_bottom = horizon_y + f * cam.cam_height_m / depth_m
    y_top = horizon_y + f * (cam.cam_height_m - height_m) / depth_m
    x1, x2 = cx - half_w, cx + half_w
    full_area = (x2 - x1) * (y_bottom - y_top)
    cx1, cy1 = max(0.0, x1), max(0.0, y_top)
    cx2, cy2 = min(float(cam.width), x2), min(float(cam.height), y_bottom)
    if cx2 <= cx1 or cy2 <= cy1 or full_area <= 0:
        return None
    if (cx2 - cx1) * (cy2 - cy1) < 0.3 * full_area:
        return None
    return ProjectedBox(actor_id, kind, depth_m, cx1, cy1, cx2, cy2)


def _overlap_fraction(inner: ProjectedBox, outer: ProjectedBox) -> float:
    ix = max(0.0, min(inner.x2, outer.x2) - max(inner.x1, outer.x1))
    iy = max(0.0, min(inner.y2, outer.y2) - max(inner.y1, outer.y1))
    area = max(1e-6, (inner.x2 - inner.x1) * (inner.y2 - inner.y1))
    return ix * iy / area


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, x))))


def simulate_detections(
    boxes: list[ProjectedBox],
    cam: CameraConfig,
    noise: DetectorNoiseConfig,
    rng: random.Random,
) -> list[Detection]:
    """GT 투영 박스에 가림/미검출/흔들림/클래스 혼동/오검출을 적용한다."""

    level = max(0.0, noise.level)
    ordered = sorted(boxes, key=lambda b: b.depth)
    out: list[Detection] = []
    for idx, box in enumerate(ordered):
        occluded = max((_overlap_fraction(box, near) for near in ordered[:idx]), default=0.0)
        if occluded > 0.7 and rng.random() < 0.8:
            continue
        w, h = box.x2 - box.x1, box.y2 - box.y1
        size = math.sqrt(max(0.0, w * h))
        p_det = _sigmoid((size - noise.size50_px) / noise.size_scale_px) * (1.0 - noise.base_miss * level)
        p_det *= 1.0 - 0.5 * occluded
        if rng.random() > p_det:
            continue
        sigma = noise.jitter_ratio * level
        x1 = box.x1 + rng.gauss(0.0, sigma * w)
        x2 = box.x2 + rng.gauss(0.0, sigma * w)
        y1 = box.y1 + rng.gauss(0.0, sigma * h)
        y2 = box.y2 + rng.gauss(0.0, sigma * h)
        x1, x2 = sorted((max(0.0, x1), min(float(cam.width), x2)))
        y1, y2 = sorted((max(0.0, y1), min(float(cam.height), y2)))
        if x2 - x1 < 2.0 or y2 - y1 < 2.0:
            continue
        conf = 0.35 + 0.6 * _sigmoid((size - 20.0) / 10.0) - 0.2 * occluded + rng.gauss(0.0, noise.conf_noise * level)
        conf = max(0.05, min(0.99, conf))
        if conf < noise.output_conf:
            continue
        cls_id = KIND_TO_CLS[box.kind]
        if cls_id in (CLS_PERSON, CLS_BIKE) and rng.random() < noise.class_swap_vru * level:
            cls_id = CLS_BIKE if cls_id == CLS_PERSON else CLS_PERSON
        out.append(Detection(x1, y1, x2, y2, cls_id, conf))

    # 오검출(false positive): 포아송 근사
    lam = noise.fp_rate * level
    n_fp = 0
    threshold = math.exp(-lam)
    prod = rng.random()
    while prod > threshold:
        n_fp += 1
        prod *= rng.random()
    for _ in range(n_fp):
        bw = rng.uniform(10.0, 80.0)
        bh = bw * rng.uniform(0.6, 1.8)
        x1 = rng.uniform(0.0, cam.width - bw)
        y1 = rng.uniform(0.3 * cam.height, cam.height - bh)
        out.append(Detection(x1, y1, x1 + bw, y1 + bh, rng.choice([CLS_PERSON, CLS_VEHICLE, CLS_BIKE]), rng.uniform(0.25, 0.55)))
    return out
