from __future__ import annotations

from pathlib import Path

from vcp.tools.brake_review_ui import (
    ReviewSegment,
    build_default_annotation_path,
    build_segment_record,
)


def test_build_default_annotation_path_uses_reviewed_suffix(tmp_path: Path) -> None:
    video_path = Path("data/raw/videos/stopcar.mp4")
    annotation_path = build_default_annotation_path(tmp_path, video_path)
    assert annotation_path == tmp_path / "stopcar_reviewed_context_segments.jsonl"


def test_build_segment_record_contains_review_fields() -> None:
    segment = ReviewSegment(
        start_frame=100,
        end_frame=120,
        context_tag="hard_brake_risk",
        boundary_frames=[100, 108],
        event_active=True,
        status="reviewed",
        confidence="high",
        notes="brake_review_ui",
    )
    row = build_segment_record(
        segment_index=1,
        run_id="stopcar_demo_run",
        video_path=Path("data/raw/videos/stopcar.mp4"),
        segment=segment,
    )
    assert row["segment_id"] == "stopcar_review_001"
    assert row["context_tag"] == "hard_brake_risk"
    assert row["boundary_frames"] == [100, 108]
    assert row["event_active"] is True
    assert row["status"] == "reviewed"
