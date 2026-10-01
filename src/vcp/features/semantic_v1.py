from __future__ import annotations

"""특징 v1: 2026-03 제출본과 동일한 16차원 의미 특징.

`YOLOSpatialEncoder`에 있던 계산을 그대로 옮겼다(배포 체크포인트 호환).
알려진 한계(문서화 목적으로 유지):
- looming/motion이 프레임 간 차분이라 FPS가 바뀌면 값의 스케일이 바뀐다.
- 직전 프레임에 ROI 객체가 없으면 이전 면적이 0으로 취급되어 looming이 급등한다.
"""

from .base import Detection, FrameDetections

V1_KEYS: list[str] = [
    "det_norm",
    "mean_conf",
    "max_conf",
    "mean_area",
    "area_var",
    "vehicle_ratio",
    "person_ratio",
    "bike_ratio",
    "vehicle_conf",
    "person_conf",
    "bike_conf",
    "roi_risk",
    "motion_delta",
    "center_closeness",
    "looming_score",
    "occlusion_score",
]


class SemanticFeatureV1:
    """v1 의미 특징 추출기.

    주의: 원본 구현은 클래스 순서를 검출기 클래스 ID 순서(person, vehicle, bike)로
    채우면서 key 이름은 vehicle_ratio, person_ratio 순서로 붙였다. 즉 실제 값은
    index 5 = person 비율, index 6 = vehicle 비율이다. 체크포인트 호환을 위해
    값 배치는 그대로 유지하고, 이 사실을 문서에 명시한다.
    """

    keys: list[str] = list(V1_KEYS)

    def __init__(
        self,
        conf_threshold: float = 0.0,
        max_det: int = 30,
        class_count: int = 3,
    ) -> None:
        self._conf_threshold = float(conf_threshold)
        self._max_det = max(1, int(max_det))
        self._class_count = max(1, int(class_count))
        self.reset()

    def reset(self) -> None:
        self._prev_roi_mean_area = 0.0
        self._prev_roi_center_y = 0.0
        self._prev_roi_count = 0
        self._prev_det_count = 0

    def update(self, frame: FrameDetections) -> list[float]:
        boxes: list[Detection] = [b for b in frame.boxes if b.conf >= self._conf_threshold]
        if not boxes:
            # 원본 구현: 검출이 없으면 0 벡터를 반환하고 직전 상태는 갱신하지 않는다.
            return [0.0] * len(V1_KEYS)

        counts = [0.0] * self._class_count
        conf_sums = [0.0] * self._class_count
        normalized_areas: list[float] = []
        roi_area_ratios: list[float] = []
        roi_center_scores: list[float] = []
        roi_vertical_scores: list[float] = []

        h, w = float(frame.height), float(frame.width)
        image_area = max(1.0, h * w)
        roi_x1, roi_x2 = 0.25 * w, 0.75 * w
        roi_y1, roi_y2 = 0.20 * h, 0.90 * h

        conf_values = [b.conf for b in boxes]
        for box in boxes:
            if box.cls_id < 0 or box.cls_id >= self._class_count:
                continue
            counts[box.cls_id] += 1.0
            conf_sums[box.cls_id] += box.conf
            area_ratio = (box.width * box.height) / image_area
            normalized_areas.append(area_ratio)
            center_x, center_y = box.cx, box.cy
            if roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2:
                roi_area_ratios.append(area_ratio)
                center_dx = abs(center_x - (w * 0.5)) / max(1.0, w * 0.5)
                roi_center_scores.append(max(0.0, 1.0 - center_dx))
                roi_vertical_scores.append(min(1.0, max(0.0, center_y / max(1.0, h))))

        det_count = float(sum(counts))
        mean_conf = float(sum(conf_values) / max(1, len(conf_values)))
        max_conf = float(max(conf_values)) if conf_values else 0.0
        mean_area = float(sum(normalized_areas) / max(1, len(normalized_areas)))
        area_var = 0.0
        if normalized_areas:
            area_var = sum((item - mean_area) ** 2 for item in normalized_areas) / len(normalized_areas)

        count_ratios = [count / max(1.0, det_count) for count in counts]
        class_mean_conf = [
            conf_sums[idx] / counts[idx] if counts[idx] > 0 else 0.0 for idx in range(len(counts))
        ]

        roi_count = len(roi_area_ratios)
        roi_mean_area = float(sum(roi_area_ratios) / max(1, roi_count))
        roi_center_closeness = float(sum(roi_center_scores) / max(1, roi_count))
        roi_vertical_bias = float(sum(roi_vertical_scores) / max(1, roi_count))
        roi_risk = min(1.0, (0.55 * min(1.0, roi_mean_area * 6.0)) + (0.45 * roi_center_closeness))

        looming_raw = max(0.0, roi_mean_area - self._prev_roi_mean_area)
        looming_score = min(1.0, looming_raw * 20.0)

        det_delta = abs(det_count - self._prev_det_count) / max(1.0, float(self._max_det))
        roi_center_y_shift = abs(roi_vertical_bias - self._prev_roi_center_y)
        motion_delta = min(1.0, (looming_raw * 14.0) + (0.5 * roi_center_y_shift) + (0.3 * det_delta))

        stop_motion_signal = 0.0
        if roi_count > 0 and self._prev_roi_count > 0:
            stop_motion_signal = max(0.0, min(1.0, (roi_risk * (1.0 - min(1.0, motion_delta * 1.8)))))
        occlusion_score = min(1.0, (0.6 * stop_motion_signal) + (0.4 * min(1.0, roi_count / 3.0)))

        vector = [
            min(1.0, det_count / max(1.0, float(self._max_det))),
            mean_conf,
            max_conf,
            mean_area,
            min(1.0, area_var * 10.0),
        ] + count_ratios + class_mean_conf + [
            roi_risk,
            motion_delta,
            roi_center_closeness,
            looming_score,
            occlusion_score,
        ]

        self._prev_roi_mean_area = roi_mean_area
        self._prev_roi_center_y = roi_vertical_bias
        self._prev_roi_count = roi_count
        self._prev_det_count = int(det_count)
        return _fit_dim([float(v) for v in vector], len(V1_KEYS))


def _fit_dim(values: list[float], dim: int) -> list[float]:
    """원본 `_fit_feature_dim`과 동일한 접기(fold) 규칙."""

    output = [0.0] * dim
    fold_count = [0] * dim
    for idx, value in enumerate(values):
        target = idx % dim
        output[target] += float(value)
        fold_count[target] += 1
    for idx in range(dim):
        if fold_count[idx] > 1:
            output[idx] /= float(fold_count[idx])
        output[idx] = round(output[idx], 6)
    return output
