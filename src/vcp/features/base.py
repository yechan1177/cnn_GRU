from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

# 3클래스 검출기 기준 클래스 ID (models/checkpoints/yolo3cls_best.pt)
CLS_PERSON = 0
CLS_VEHICLE = 1
CLS_BIKE = 2


@dataclass(slots=True, frozen=True)
class Detection:
    """단일 검출 박스(픽셀 좌표)."""

    x1: float
    y1: float
    x2: float
    y2: float
    cls_id: int
    conf: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return 0.5 * (self.x1 + self.x2)

    @property
    def cy(self) -> float:
        return 0.5 * (self.y1 + self.y2)


@dataclass(slots=True)
class FrameDetections:
    """한 프레임의 검출 결과와 시간 정보."""

    frame_id: int
    t: float
    width: int
    height: int
    boxes: list[Detection] = field(default_factory=list)


def frame_from_dict(row: dict[str, Any]) -> FrameDetections:
    """`extract_detections` JSONL 한 줄을 FrameDetections로 변환한다."""

    boxes = [
        Detection(float(b[0]), float(b[1]), float(b[2]), float(b[3]), int(b[4]), float(b[5]))
        for b in row.get("boxes", [])
    ]
    return FrameDetections(
        frame_id=int(row.get("frame_id", 0)),
        t=float(row.get("t", 0.0)),
        width=int(row.get("width", 640)),
        height=int(row.get("height", 480)),
        boxes=boxes,
    )


def iou(a: Detection, b: Detection) -> float:
    """두 박스의 IoU."""

    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


class FeatureExtractor(Protocol):
    """상태를 가지는(직전 프레임 참조) 특징 추출기 인터페이스."""

    keys: list[str]

    def reset(self) -> None: ...

    def update(self, frame: FrameDetections) -> list[float]: ...
