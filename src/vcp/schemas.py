from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class FramePacket:
    """카메라 입력 프레임의 최소 표현.

    raw_path는 이미지 파일 기반 입력 소스를 사용할 때 채워진다.
    image는 비디오/카메라 실시간 프레임(np.ndarray 등)을 직접 전달할 때 사용한다.
    """

    frame_id: int
    sensor_timestamp: float
    system_timestamp: float
    pixels: list[float]
    raw_path: str | None = None
    image: Any | None = None


@dataclass(slots=True)
class PackedFeature:
    """Spatial feature를 temporal 입력 형태로 저장한 구조."""

    frame_id: int
    sensor_timestamp: float
    system_timestamp: float
    spatial_vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TemporalOutput:
    """Temporal encoder 출력."""

    frame_id: int
    context_probs: dict[str, float]
    boundary_signal: float


@dataclass(slots=True)
class ScoreResult:
    """이벤트 점수 계산 결과."""

    frame_id: int
    context_score: float
    boundary_score: float
    uncertainty_score: float
    novelty_score: float
    total_importance: float


@dataclass(slots=True)
class FrameRecord:
    """frame-level 레코드."""

    frame_id: int
    sensor_timestamp: float
    system_timestamp: float
    context_tag: str
    scores: dict[str, float]
    derived_feature: dict[str, Any]
    raw_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "sensor_timestamp": self.sensor_timestamp,
            "system_timestamp": self.system_timestamp,
            "context_tag": self.context_tag,
            "scores": self.scores,
            "derived_feature": self.derived_feature,
            "raw_path": self.raw_path,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class EventRecord:
    """event-level 레코드."""

    event_id: str
    trigger_frame_id: int
    start_frame_id: int
    end_frame_id: int
    importance_score: float
    reason: str
    sensor_start_ts: float
    sensor_end_ts: float
    system_start_ts: float
    system_end_ts: float
    pre_frame_ids: list[int]
    post_frame_ids: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "trigger_frame_id": self.trigger_frame_id,
            "start_frame_id": self.start_frame_id,
            "end_frame_id": self.end_frame_id,
            "importance_score": self.importance_score,
            "reason": self.reason,
            "sensor_start_ts": self.sensor_start_ts,
            "sensor_end_ts": self.sensor_end_ts,
            "system_start_ts": self.system_start_ts,
            "system_end_ts": self.system_end_ts,
            "pre_frame_ids": self.pre_frame_ids,
            "post_frame_ids": self.post_frame_ids,
        }


@dataclass(slots=True)
class CurationDecision:
    """자동 정제 엔진의 정책 결정 결과."""

    storage_policy: str
    finalized_event: EventRecord | None = None
