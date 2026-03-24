from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .parts import (
    CompositeEventScorer,
    JsonlDatasetExporter,
    RingBufferCurationEngine,
    SimpleFeaturePacker,
    build_camera_stream,
    build_spatial_encoder,
    build_temporal_encoder,
)
from .config import PipelineConfig, load_config
from .schemas import FrameRecord

logger = logging.getLogger(__name__)


class VisionContextPipeline:
    """비전 특징-맥락-자동정제-정렬 export 파이프라인."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.cfg: PipelineConfig = load_config(self.config_path)

        self.camera = build_camera_stream(self.cfg.camera, self.cfg.runtime)
        self.spatial = build_spatial_encoder(self.cfg.spatial)
        self.packer = SimpleFeaturePacker()
        self.temporal = build_temporal_encoder(
            cfg=self.cfg.temporal,
            labels=self.cfg.labels,
            input_dim=max(1, int(self.cfg.spatial.feature_dim)),
        )
        self.scorer = CompositeEventScorer(self.cfg.scoring)
        self.curation = RingBufferCurationEngine(self.cfg.curation, self.cfg.scoring)
        self.exporter = JsonlDatasetExporter(self.cfg.storage)

    def run(self) -> dict[str, Any]:
        logger.info("파이프라인 시작: profile=%s", self.cfg.runtime.profile)
        frame_count = 0
        finalized_event_count = 0

        while True:
            frame = self.camera.read()
            if frame is None:
                break

            spatial_vector = self.spatial.encode(frame)
            packed = self.packer.pack(frame, spatial_vector)
            temporal_out = self.temporal.encode(packed)
            score = self.scorer.score(packed, temporal_out)
            context_tag = max(temporal_out.context_probs, key=temporal_out.context_probs.get)

            frame_record = FrameRecord(
                frame_id=frame.frame_id,
                sensor_timestamp=frame.sensor_timestamp,
                system_timestamp=frame.system_timestamp,
                context_tag=context_tag,
                scores={
                    "context": score.context_score,
                    "boundary": score.boundary_score,
                    "uncertainty": score.uncertainty_score,
                    "novelty": score.novelty_score,
                    "total_importance": score.total_importance,
                },
                derived_feature={
                    "feature_dim": len(spatial_vector),
                    "feature_sample": [round(value, 6) for value in spatial_vector[:8]],
                },
                raw_path=frame.raw_path,
                metadata={
                    "config_profile": self.cfg.runtime.profile,
                    "spatial_model": self.cfg.spatial.model_name,
                    "temporal_model": self.cfg.temporal.model_name,
                },
            )

            decision = self.curation.process_frame(frame_record, score)
            self.exporter.add_frame(frame_record, decision.storage_policy)
            if decision.finalized_event is not None:
                self.exporter.add_event(decision.finalized_event)
                finalized_event_count += 1

            frame_count += 1

        for event in self.curation.flush():
            self.exporter.add_event(event)
            finalized_event_count += 1

        export_result = self.exporter.finalize()
        self.camera.close()
        logger.info(
            "파이프라인 종료: frames=%s, events=%s", frame_count, finalized_event_count
        )

        return {
            "config_path": str(self.config_path),
            "profile": self.cfg.runtime.profile,
            "frames": frame_count,
            "events": finalized_event_count,
            "run_id": export_result["run_id"],
            "run_dir": export_result["run_dir"],
            "exports_dir": export_result["exports_dir"],
        }
