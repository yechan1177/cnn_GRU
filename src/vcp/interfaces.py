from __future__ import annotations

from abc import ABC, abstractmethod

from .schemas import (
    CurationDecision,
    EventRecord,
    FramePacket,
    FrameRecord,
    PackedFeature,
    ScoreResult,
    TemporalOutput,
)


class CameraStream(ABC):
    """카메라 입력 스트림 인터페이스."""

    @abstractmethod
    def read(self) -> FramePacket | None:
        """다음 프레임을 반환한다. 종료 시 None을 반환한다."""

    @abstractmethod
    def close(self) -> None:
        """스트림 리소스를 해제한다."""


class SpatialEncoder(ABC):
    """공간 특징 추출기 인터페이스."""

    @abstractmethod
    def encode(self, frame: FramePacket) -> list[float]:
        """프레임을 고정 길이 특징 벡터로 변환한다."""


class FeaturePacker(ABC):
    """Feature packing 인터페이스."""

    @abstractmethod
    def pack(self, frame: FramePacket, spatial_vector: list[float]) -> PackedFeature:
        """spatial feature를 temporal encoder 입력 형태로 변환한다."""


class TemporalEncoder(ABC):
    """시계열 맥락 인코더 인터페이스."""

    @abstractmethod
    def encode(self, packed: PackedFeature) -> TemporalOutput:
        """맥락 확률과 경계 신호를 생성한다."""


class EventScorer(ABC):
    """이벤트 점수화 인터페이스."""

    @abstractmethod
    def score(self, packed: PackedFeature, temporal: TemporalOutput) -> ScoreResult:
        """context/boundary/uncertainty/novelty 점수를 계산한다."""


class AutoCurationEngine(ABC):
    """자동 정제 엔진 인터페이스."""

    @abstractmethod
    def process_frame(self, frame_record: FrameRecord, score: ScoreResult) -> CurationDecision:
        """프레임 저장 정책을 결정하고, 필요 시 이벤트를 완결한다."""

    @abstractmethod
    def flush(self) -> list[EventRecord]:
        """스트림 종료 시 미완료 이벤트를 마감해 반환한다."""


class DatasetExporter(ABC):
    """정렬 데이터셋 export 인터페이스."""

    @abstractmethod
    def add_frame(self, frame_record: FrameRecord, storage_policy: str) -> None:
        """frame-level 레코드를 수집한다."""

    @abstractmethod
    def add_event(self, event_record: EventRecord) -> None:
        """event-level 레코드를 수집한다."""

    @abstractmethod
    def finalize(self) -> dict[str, str | int]:
        """수집된 레코드를 파일로 저장하고 실행 요약을 반환한다."""
