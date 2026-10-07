from __future__ import annotations

"""특징 v2: 시간 정규화(FPS 불변) + 경량 추적 기반 의미 특징.

v1 대비 변경점
- 모든 변화율을 프레임 차분이 아닌 초 단위(Δt)로 계산하고, 지수이동평균의
  계수도 Δt로부터 계산한다. → 10/15/30 fps에서 같은 물리 상황이면 비슷한 값.
- 선행 차량(자차 진행 경로 위의 가장 가까운 차량)과 가장 가까운 보행자/이륜차를
  IoU로 프레임 간 연결한다. 연결이 끊긴 신규 객체는 변화율 0에서 시작한다
  (v1의 신규 객체 진입 시 looming 급등 제거).
- 박스 크기 변화율 s' / s 로부터 역 TTC(1/TTC)를 추정한다.
  (단안 카메라에서 물체 폭 w ∝ 1/Z 이므로 d ln w / dt = -Ż / Z = 1/TTC)
"""

import math

from .base import CLS_BIKE, CLS_PERSON, CLS_VEHICLE, Detection, FrameDetections, iou

V2_KEYS: list[str] = [
    # global
    "det_norm",
    "mean_conf",
    "mean_area",
    "area_var",
    "det_rate",
    # vehicle / lead
    "vehicle_ratio",
    "lead_present",
    "lead_size",
    "lead_bottom",
    "lead_scale_rate",
    "lead_inv_ttc",
    # vulnerable road user (person / bike)
    "person_ratio",
    "bike_ratio",
    "vru_present",
    "vru_size",
    "vru_approach",
]


class _Track:
    """단일 객체 추적 상태(IoU 연결 + Δt 기반 EMA 변화율)."""

    __slots__ = ("box", "log_size", "rate", "lateral", "age")

    def __init__(self, box: Detection) -> None:
        self.box = box
        self.log_size = math.log(max(1.0, box.width))
        self.rate = 0.0
        self.lateral = 0.0
        self.age = 0


class SemanticFeatureV2:
    """v2 의미 특징 추출기(16차원)."""

    keys: list[str] = list(V2_KEYS)

    def __init__(
        self,
        conf_threshold: float = 0.0,
        max_det: int = 30,
        tau_s: float = 0.25,
        iou_match: float = 0.3,
        lane_half_width: float = 0.18,
    ) -> None:
        self._conf_threshold = float(conf_threshold)
        self._max_det = max(1, int(max_det))
        self._tau = max(1e-3, float(tau_s))
        self._iou_match = float(iou_match)
        self._lane_half_width = float(lane_half_width)
        self.reset()

    def reset(self) -> None:
        self._prev_t: float | None = None
        self._prev_det_norm = 0.0
        self._det_rate = 0.0
        self._lead: _Track | None = None
        self._vru: _Track | None = None

    # ------------------------------------------------------------------
    def update(self, frame: FrameDetections) -> list[float]:
        dt = 0.0 if self._prev_t is None else max(1e-3, frame.t - self._prev_t)
        alpha = 0.0 if dt == 0.0 else 1.0 - math.exp(-dt / self._tau)
        self._prev_t = frame.t

        w, h = float(max(1, frame.width)), float(max(1, frame.height))
        image_area = w * h
        boxes = [b for b in frame.boxes if b.conf >= self._conf_threshold]
        n = len(boxes)

        det_norm = min(1.0, n / float(self._max_det))
        if dt > 0.0:
            raw_rate = (det_norm - self._prev_det_norm) / dt
            self._det_rate += alpha * (raw_rate - self._det_rate)
        self._prev_det_norm = det_norm

        areas = [(b.width * b.height) / image_area for b in boxes]
        mean_area = sum(areas) / n if n else 0.0
        area_var = sum((a - mean_area) ** 2 for a in areas) / n if n else 0.0
        mean_conf = sum(b.conf for b in boxes) / n if n else 0.0
        n_vehicle = sum(1 for b in boxes if b.cls_id == CLS_VEHICLE)
        n_person = sum(1 for b in boxes if b.cls_id == CLS_PERSON)
        n_bike = sum(1 for b in boxes if b.cls_id == CLS_BIKE)

        # 선행 차량: 자차 진행 경로(화면 하단 중앙 통로) 위에서 하단 y가 가장 큰 차량
        lead_candidates = [
            b for b in boxes if b.cls_id == CLS_VEHICLE and abs(b.cx - 0.5 * w) <= self._lane_half_width * w + 0.25 * b.width
        ]
        lead_box = max(lead_candidates, key=lambda b: b.y2, default=None)
        self._lead = self._step_track(self._lead, lead_box, dt, alpha, w)

        # VRU: 보행자/이륜차 중 가장 큰(가까운) 객체, 화면 중앙 60% 범위
        vru_candidates = [
            b for b in boxes if b.cls_id in (CLS_PERSON, CLS_BIKE) and abs(b.cx - 0.5 * w) <= 0.30 * w
        ]
        vru_box = max(vru_candidates, key=lambda b: b.width * b.height, default=None)
        self._vru = self._step_track(self._vru, vru_box, dt, alpha, w)

        lead = self._lead
        if lead is not None:
            lead_size = min(1.0, math.sqrt((lead.box.width * lead.box.height) / image_area) * 2.0)
            lead_bottom = min(1.0, max(0.0, lead.box.y2 / h))
            lead_scale_rate = max(-1.0, min(1.0, lead.rate / 1.0))
            lead_inv_ttc = max(0.0, min(1.0, lead.rate / 1.0))
        else:
            lead_size = lead_bottom = lead_scale_rate = lead_inv_ttc = 0.0

        vru = self._vru
        if vru is not None:
            vru_size = min(1.0, math.sqrt((vru.box.width * vru.box.height) / image_area) * 2.0)
            # 화면 중앙 방향 횡이동 속도(화면폭/초) + 크기 증가율을 합친 접근 지표
            vru_approach = max(-1.0, min(1.0, vru.lateral + 0.5 * vru.rate))
        else:
            vru_size = vru_approach = 0.0

        vector = [
            det_norm,
            mean_conf,
            min(1.0, mean_area * 4.0),
            min(1.0, area_var * 10.0),
            max(-1.0, min(1.0, self._det_rate)),
            n_vehicle / n if n else 0.0,
            1.0 if lead is not None else 0.0,
            lead_size,
            lead_bottom,
            lead_scale_rate,
            lead_inv_ttc,
            n_person / n if n else 0.0,
            n_bike / n if n else 0.0,
            1.0 if vru is not None else 0.0,
            vru_size,
            vru_approach,
        ]
        return [round(float(v), 6) for v in vector]

    # ------------------------------------------------------------------
    def _step_track(
        self,
        track: _Track | None,
        box: Detection | None,
        dt: float,
        alpha: float,
        width: float,
    ) -> _Track | None:
        if box is None:
            return None
        if track is None or dt == 0.0 or iou(track.box, box) < self._iou_match:
            return _Track(box)
        log_size = math.log(max(1.0, box.width))
        raw_rate = (log_size - track.log_size) / dt
        prev_offset = abs(track.box.cx - 0.5 * width) / width
        offset = abs(box.cx - 0.5 * width) / width
        raw_lateral = (prev_offset - offset) / dt
        track.rate += alpha * (raw_rate - track.rate)
        track.lateral += alpha * (raw_lateral - track.lateral)
        track.box = box
        track.log_size = log_size
        track.age += 1
        return track
