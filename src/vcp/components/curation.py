from __future__ import annotations

from collections import deque
from typing import Any

from ..config import CurationConfig, ScoringConfig
from ..interfaces import AutoCurationEngine
from ..schemas import CurationDecision, EventRecord, FrameRecord, ScoreResult


class RingBufferCurationEngine(AutoCurationEngine):
    """ring buffer 기반 자동 정제 엔진."""

    def __init__(self, curation_cfg: CurationConfig, scoring_cfg: ScoringConfig) -> None:
        self._ring_buffer: deque[FrameRecord] = deque(
            maxlen=max(8, int(curation_cfg.ring_buffer_size))
        )
        self._pre_event_frames = max(1, int(curation_cfg.pre_event_frames))
        self._post_event_frames = max(1, int(curation_cfg.post_event_frames))
        self._high_threshold = float(scoring_cfg.high_threshold)
        self._mid_threshold = float(scoring_cfg.mid_threshold)
        self._adaptive_boundary_gate = 0.30
        self._adaptive_novelty_gate = 0.03
        self._event_index = 0
        self._active_event: dict[str, Any] | None = None

    def process_frame(self, frame_record: FrameRecord, score: ScoreResult) -> CurationDecision:
        self._ring_buffer.append(frame_record)

        if self._active_event is not None:
            self._active_event["post_frame_ids"].append(frame_record.frame_id)
            self._active_event["frames_left"] -= 1
            if self._active_event["frames_left"] <= 0:
                finalized_event = self._finalize_event(frame_record)
                self._active_event = None
                return CurationDecision(storage_policy="clip", finalized_event=finalized_event)

        high_trigger = score.total_importance >= self._high_threshold
        adaptive_trigger = (
            score.total_importance >= self._mid_threshold
            and score.boundary_score >= self._adaptive_boundary_gate
            and score.novelty_score >= self._adaptive_novelty_gate
        )

        if self._active_event is None and (high_trigger or adaptive_trigger):
            trigger_mode = "high_threshold" if high_trigger else "adaptive_gate"
            self._start_event(frame_record, score, trigger_mode)
            return CurationDecision(storage_policy="clip", finalized_event=None)

        if score.total_importance >= self._mid_threshold:
            return CurationDecision(storage_policy="keyframe", finalized_event=None)

        return CurationDecision(storage_policy="discard", finalized_event=None)

    def flush(self) -> list[EventRecord]:
        if self._active_event is None:
            return []
        if not self._ring_buffer:
            self._active_event = None
            return []
        last_frame = self._ring_buffer[-1]
        finalized_event = self._finalize_event(last_frame)
        self._active_event = None
        return [finalized_event]

    def _start_event(
        self, frame_record: FrameRecord, score: ScoreResult, trigger_mode: str
    ) -> None:
        self._event_index += 1
        history = list(self._ring_buffer)[:-1]
        pre_frames = history[-self._pre_event_frames :]
        pre_frame_ids = [item.frame_id for item in pre_frames]
        start_frame_id = pre_frame_ids[0] if pre_frame_ids else frame_record.frame_id
        sensor_start_ts = (
            pre_frames[0].sensor_timestamp if pre_frames else frame_record.sensor_timestamp
        )
        system_start_ts = (
            pre_frames[0].system_timestamp if pre_frames else frame_record.system_timestamp
        )

        self._active_event = {
            "event_id": f"event_{self._event_index:05d}",
            "trigger_frame_id": frame_record.frame_id,
            "start_frame_id": start_frame_id,
            "sensor_start_ts": sensor_start_ts,
            "system_start_ts": system_start_ts,
            "importance_score": score.total_importance,
            "reason": (
                f"mode={trigger_mode}, importance={score.total_importance:.3f}, "
                f"boundary={score.boundary_score:.3f}, novelty={score.novelty_score:.3f}, "
                f"context={frame_record.context_tag}"
            ),
            "pre_frame_ids": pre_frame_ids,
            "post_frame_ids": [],
            "frames_left": self._post_event_frames,
        }

    def _finalize_event(self, end_frame_record: FrameRecord) -> EventRecord:
        assert self._active_event is not None
        state = dict(self._active_event)
        post_frame_ids = state["post_frame_ids"][: self._post_event_frames]
        if post_frame_ids:
            end_frame_id = post_frame_ids[-1]
        else:
            end_frame_id = end_frame_record.frame_id
        return EventRecord(
            event_id=state["event_id"],
            trigger_frame_id=state["trigger_frame_id"],
            start_frame_id=state["start_frame_id"],
            end_frame_id=end_frame_id,
            importance_score=float(state["importance_score"]),
            reason=state["reason"],
            sensor_start_ts=float(state["sensor_start_ts"]),
            sensor_end_ts=end_frame_record.sensor_timestamp,
            system_start_ts=float(state["system_start_ts"]),
            system_end_ts=end_frame_record.system_timestamp,
            pre_frame_ids=list(state["pre_frame_ids"]),
            post_frame_ids=list(post_frame_ids),
        )
