from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import StorageConfig
from ..interfaces import DatasetExporter
from ..schemas import EventRecord, FrameRecord


class JsonlDatasetExporter(DatasetExporter):
    """frame/event/aligned 레코드를 JSONL로 저장한다."""

    def __init__(self, cfg: StorageConfig) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_id = f"{cfg.run_name}_{timestamp}"
        self._base_dir = Path(cfg.base_dir)
        self._run_dir = self._base_dir / "runs" / self._run_id
        self._clips_dir = self._base_dir / "clips" / self._run_id
        self._keyframes_dir = self._base_dir / "keyframes" / self._run_id
        self._exports_dir = self._base_dir / "exports" / self._run_id
        for directory in (
            self._run_dir,
            self._clips_dir,
            self._keyframes_dir,
            self._exports_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self._frame_items: list[tuple[FrameRecord, str]] = []
        self._events: list[EventRecord] = []

    def add_frame(self, frame_record: FrameRecord, storage_policy: str) -> None:
        self._frame_items.append((frame_record, storage_policy))

        if storage_policy == "keyframe":
            keyframe_path = self._keyframes_dir / f"frame_{frame_record.frame_id:06d}.json"
            payload = frame_record.to_dict()
            payload["storage_policy"] = storage_policy
            keyframe_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def add_event(self, event_record: EventRecord) -> None:
        self._events.append(event_record)
        clip_path = self._clips_dir / f"{event_record.event_id}.json"
        clip_path.write_text(
            json.dumps(event_record.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def finalize(self) -> dict[str, str | int]:
        frame_path = self._run_dir / "frame_records.jsonl"
        event_path = self._run_dir / "event_records.jsonl"
        aligned_path = self._exports_dir / "aligned_dataset.jsonl"
        manifest_path = self._exports_dir / "export_manifest.json"

        frame_to_events = self._build_frame_to_event_index()

        frame_lines: list[dict[str, Any]] = []
        aligned_lines: list[dict[str, Any]] = []
        for frame_record, policy in self._frame_items:
            base = frame_record.to_dict()
            base["storage_policy"] = policy
            frame_lines.append(base)

            aligned = {
                "frame_id": frame_record.frame_id,
                "sensor_timestamp": frame_record.sensor_timestamp,
                "system_timestamp": frame_record.system_timestamp,
                "context_tag": frame_record.context_tag,
                "scores": frame_record.scores,
                "event_ids": frame_to_events.get(frame_record.frame_id, []),
                "aligned_modalities": {
                    "vision_feature": frame_record.derived_feature,
                    "eeg": None,
                    "emg": None,
                    "imu": None,
                    "robot_state": None,
                    "trajectory": None,
                },
            }
            aligned_lines.append(aligned)

        event_lines = [event.to_dict() for event in self._events]

        _write_jsonl(frame_path, frame_lines)
        _write_jsonl(event_path, event_lines)
        _write_jsonl(aligned_path, aligned_lines)

        manifest = {
            "run_id": self._run_id,
            "frame_count": len(frame_lines),
            "event_count": len(event_lines),
            "paths": {
                "frame_records": str(frame_path),
                "event_records": str(event_path),
                "aligned_dataset": str(aligned_path),
                "clips_dir": str(self._clips_dir),
                "keyframes_dir": str(self._keyframes_dir),
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return {
            "run_id": self._run_id,
            "run_dir": str(self._run_dir),
            "exports_dir": str(self._exports_dir),
            "frame_count": len(frame_lines),
            "event_count": len(event_lines),
        }

    def _build_frame_to_event_index(self) -> dict[int, list[str]]:
        index: dict[int, list[str]] = {}
        for event in self._events:
            linked_ids = set(event.pre_frame_ids + event.post_frame_ids + [event.trigger_frame_id])
            for frame_id in linked_ids:
                index.setdefault(frame_id, []).append(event.event_id)
        return index


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False))
            file.write("\n")
