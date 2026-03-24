from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _import_cv2_np() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ModuleNotFoundError as exc:  # pragma: no cover - 환경 의존
        raise ModuleNotFoundError(
            "브레이크 리뷰 UI 실행에는 opencv-python과 numpy가 필요합니다."
        ) from exc
    return cv2, np


@dataclass(slots=True)
class ReviewLabel:
    """검수 UI에서 사용하는 맥락 라벨 정의."""

    name: str
    event_active: bool
    description: str


@dataclass(slots=True)
class ReviewSegment:
    """UI에서 편집/저장하는 구간 정보."""

    start_frame: int
    end_frame: int
    context_tag: str
    boundary_frames: list[int]
    event_active: bool
    status: str
    confidence: str
    notes: str


@dataclass(slots=True)
class ReviewData:
    """리뷰 UI가 프레임별로 참고하는 run 데이터."""

    run_dir: Path
    video_path: Path
    annotation_path: Path
    frame_rows: list[dict[str, Any]]
    labels: list[ReviewLabel]


@dataclass(slots=True)
class ReviewArgs:
    """브레이크 검수 UI 인자."""

    run_dir: Path | None
    video: Path | None
    annotation_path: Path | None
    annotations_root: Path
    label_schema_path: Path
    start_frame: int
    play_fps: float
    export_preview: Path | None


def parse_args() -> ReviewArgs:
    parser = argparse.ArgumentParser(description="브레이크 전환 구간 검수 UI")
    parser.add_argument("--run-dir", type=str, default="", help="frame_records.jsonl이 있는 run 디렉터리")
    parser.add_argument("--video", type=str, default="", help="원본 비디오 경로")
    parser.add_argument(
        "--annotation-path",
        type=str,
        default="",
        help="저장할 annotation jsonl 경로",
    )
    parser.add_argument(
        "--annotations-root",
        type=str,
        default="data/annotations/context",
        help="기본 annotation 저장 루트",
    )
    parser.add_argument(
        "--label-schema-path",
        type=str,
        default="data/annotations/context/context_label_schema.json",
        help="맥락 라벨 스키마 경로",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="시작 프레임")
    parser.add_argument("--play-fps", type=float, default=15.0, help="재생 FPS")
    parser.add_argument(
        "--export-preview",
        type=str,
        default="",
        help="UI 첫 프레임을 이미지로 저장하고 종료",
    )
    args = parser.parse_args()
    return ReviewArgs(
        run_dir=Path(args.run_dir) if args.run_dir else None,
        video=Path(args.video) if args.video else None,
        annotation_path=Path(args.annotation_path) if args.annotation_path else None,
        annotations_root=Path(args.annotations_root),
        label_schema_path=Path(args.label_schema_path),
        start_frame=max(0, int(args.start_frame)),
        play_fps=max(1.0, float(args.play_fps)),
        export_preview=Path(args.export_preview) if args.export_preview else None,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def find_latest_braking_run(outputs_root: Path) -> Path:
    """브레이크 전환 관련 최신 run을 찾는다."""
    if not outputs_root.exists():
        raise FileNotFoundError(f"outputs/runs를 찾을 수 없습니다: {outputs_root}")
    run_dirs = [
        path
        for path in outputs_root.iterdir()
        if path.is_dir()
        and "braking" in path.name
        and (path / "frame_records.jsonl").exists()
    ]
    if not run_dirs:
        raise FileNotFoundError(
            "브레이크 전환 run을 찾지 못했습니다. 먼저 stopcar 브레이크 추론을 실행해야 합니다."
        )
    run_dirs.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return run_dirs[0]


def load_labels(schema_path: Path) -> list[ReviewLabel]:
    data = json.loads(schema_path.read_text(encoding="utf-8-sig"))
    labels: list[ReviewLabel] = []
    for row in data.get("labels", []):
        labels.append(
            ReviewLabel(
                name=str(row["name"]),
                event_active=bool(row.get("event_active", False)),
                description=str(row.get("description", "")),
            )
        )
    if not labels:
        raise ValueError(f"라벨 스키마가 비어 있습니다: {schema_path}")
    return labels


def infer_video_path(frame_rows: list[dict[str, Any]], explicit_video: Path | None) -> Path:
    """run 데이터에서 원본 영상 경로를 추론한다."""
    if explicit_video is not None:
        return explicit_video
    if not frame_rows:
        raise ValueError("frame_records가 비어 있습니다.")
    metadata = frame_rows[0].get("metadata", {})
    source_video = metadata.get("source_video")
    if source_video:
        return Path(str(source_video))
    raw_path = str(frame_rows[0].get("raw_path", ""))
    if "#frame=" in raw_path:
        return Path("data/raw/videos") / raw_path.split("#frame=", 1)[0]
    raise ValueError("video 경로를 추론하지 못했습니다. --video를 지정하세요.")


def build_default_annotation_path(annotations_root: Path, video_path: Path) -> Path:
    """영상별 reviewed annotation 기본 저장 경로를 생성한다."""
    return annotations_root / f"{video_path.stem}_reviewed_context_segments.jsonl"


def load_existing_segments(annotation_path: Path) -> list[ReviewSegment]:
    """기존 reviewed annotation이 있으면 읽어온다."""
    if not annotation_path.exists():
        return []
    rows = read_jsonl(annotation_path)
    segments: list[ReviewSegment] = []
    for row in rows:
        segments.append(
            ReviewSegment(
                start_frame=int(row.get("start_frame", 0)),
                end_frame=int(row.get("end_frame", 0)),
                context_tag=str(row.get("context_tag", "normal_drive")),
                boundary_frames=sorted({int(item) for item in row.get("boundary_frames", [])}),
                event_active=bool(row.get("event_active", False)),
                status=str(row.get("status", "reviewed")),
                confidence=str(row.get("confidence", "high")),
                notes=str(row.get("notes", "review_ui")),
            )
        )
    return segments


def build_segment_record(
    *,
    segment_index: int,
    run_id: str,
    video_path: Path,
    segment: ReviewSegment,
) -> dict[str, Any]:
    """UI에서 만든 구간을 annotation JSONL 행으로 변환한다."""
    return {
        "segment_id": f"{video_path.stem}_review_{segment_index:03d}",
        "run_id": run_id,
        "video_path": str(video_path),
        "start_frame": int(segment.start_frame),
        "end_frame": int(segment.end_frame),
        "context_tag": segment.context_tag,
        "boundary_frames": sorted({int(item) for item in segment.boundary_frames}),
        "event_active": bool(segment.event_active),
        "status": segment.status,
        "confidence": segment.confidence,
        "notes": segment.notes,
    }


def save_segments(
    *,
    annotation_path: Path,
    run_id: str,
    video_path: Path,
    segments: list[ReviewSegment],
) -> None:
    """reviewed annotation을 jsonl로 저장한다."""
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        build_segment_record(
            segment_index=index + 1,
            run_id=run_id,
            video_path=video_path,
            segment=segment,
        )
        for index, segment in enumerate(sorted(segments, key=lambda item: item.start_frame))
    ]
    with annotation_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False))
            file.write("\n")


def load_review_data(args: ReviewArgs) -> ReviewData:
    """run/annotation/schema를 한 번에 로드한다."""
    outputs_root = Path("outputs") / "runs"
    run_dir = args.run_dir or find_latest_braking_run(outputs_root)
    frame_path = run_dir / "frame_records.jsonl"
    if not frame_path.exists():
        raise FileNotFoundError(f"frame_records.jsonl이 없습니다: {frame_path}")
    frame_rows = read_jsonl(frame_path)
    frame_rows.sort(key=lambda row: int(row.get("frame_id", 0)))
    labels = load_labels(args.label_schema_path)
    video_path = infer_video_path(frame_rows, args.video)
    annotation_path = args.annotation_path or build_default_annotation_path(
        args.annotations_root, video_path
    )
    return ReviewData(
        run_dir=run_dir,
        video_path=video_path,
        annotation_path=annotation_path,
        frame_rows=frame_rows,
        labels=labels,
    )


def draw_text(
    cv2_module: Any,
    image: Any,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int] = (230, 230, 230),
    scale: float = 0.52,
) -> None:
    cv2_module.putText(
        image,
        text,
        (x, y),
        cv2_module.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2_module.LINE_AA,
    )


def draw_series(
    cv2_module: Any,
    np_module: Any,
    canvas: Any,
    *,
    x: int,
    y: int,
    w: int,
    h: int,
    values: list[float],
    color: tuple[int, int, int],
    label: str,
    vmin: float,
    vmax: float,
    ref_line: float | None = None,
) -> None:
    cv2_module.rectangle(canvas, (x, y), (x + w, y + h), (80, 80, 80), 1)
    draw_text(cv2_module, canvas, x + 4, y - 8, label, (200, 200, 200), 0.46)
    if ref_line is not None and vmax > vmin:
        ratio = max(0.0, min(1.0, (ref_line - vmin) / (vmax - vmin)))
        yy = y + h - int(ratio * h)
        cv2_module.line(canvas, (x, yy), (x + w, yy), (100, 150, 255), 1)
    if len(values) < 2:
        return
    pts = []
    denom = max(1, len(values) - 1)
    for idx, value in enumerate(values):
        xr = x + int((idx / denom) * w)
        ratio = 0.0 if vmax <= vmin else max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
        yr = y + h - int(ratio * h)
        pts.append([xr, yr])
    poly = np_module.array(pts, dtype=np_module.int32).reshape((-1, 1, 2))
    cv2_module.polylines(canvas, [poly], False, color, 2)


def render_frame(
    *,
    cv2_module: Any,
    np_module: Any,
    frame_bgr: Any,
    review_data: ReviewData,
    frame_idx: int,
    segments: list[ReviewSegment],
    selected_label_idx: int,
    mark_start: int | None,
    mark_end: int | None,
    pending_boundaries: set[int],
    message: str,
) -> Any:
    """비디오 프레임과 우측 검수 패널을 합친다."""
    row = review_data.frame_rows[frame_idx]
    scores = row.get("scores", {})
    feature_vector = row.get("derived_feature", {}).get("feature_vector", [])
    feature_keys = row.get("derived_feature", {}).get("feature_keys", [])
    feature_map = {
        str(key): float(value)
        for key, value in zip(feature_keys, feature_vector, strict=False)
    }

    video_h, video_w = frame_bgr.shape[:2]
    target_h = 540
    target_w = int((video_w / max(1, video_h)) * target_h)
    target_w = max(720, min(980, target_w))
    video = cv2_module.resize(frame_bgr, (target_w, target_h), interpolation=cv2_module.INTER_AREA)

    panel_w = 620
    canvas = np_module.zeros((target_h, target_w + panel_w, 3), dtype=np_module.uint8)
    canvas[:, :target_w] = video
    panel = canvas[:, target_w:]
    panel[:] = (26, 26, 26)

    active_segments = [
        seg
        for seg in segments
        if seg.start_frame <= frame_idx <= seg.end_frame
    ]
    active_label = active_segments[-1].context_tag if active_segments else "none"
    selected_label = review_data.labels[selected_label_idx]

    draw_text(
        cv2_module,
        canvas,
        16,
        28,
        (
            f"frame={frame_idx} pred={row.get('context_tag', 'unknown')} "
            f"review={active_label} det={row.get('metadata', {}).get('detection_count', 0)}"
        ),
        (90, 255, 120),
        0.62,
    )
    draw_text(
        cv2_module,
        canvas,
        16,
        56,
        (
            f"importance={float(scores.get('total_importance', 0.0)):.3f} "
            f"boundary={float(scores.get('boundary', 0.0)):.3f}"
        ),
        (255, 220, 120),
        0.56,
    )

    y = 28
    draw_text(cv2_module, panel, 20, y, f"run: {review_data.run_dir.name}", (255, 255, 255), 0.54)
    y += 26
    draw_text(cv2_module, panel, 20, y, f"video: {review_data.video_path.name}", (220, 220, 220), 0.52)
    y += 26
    draw_text(cv2_module, panel, 20, y, f"annotation: {review_data.annotation_path.name}", (180, 180, 180), 0.48)
    y += 30

    draw_text(cv2_module, panel, 20, y, f"selected label [{selected_label_idx + 1}]: {selected_label.name}", (100, 180, 255), 0.54)
    y += 24
    draw_text(cv2_module, panel, 20, y, f"mark start: {mark_start if mark_start is not None else '-'}", (210, 210, 210), 0.48)
    y += 22
    draw_text(cv2_module, panel, 20, y, f"mark end: {mark_end if mark_end is not None else '-'}", (210, 210, 210), 0.48)
    y += 22
    draw_text(cv2_module, panel, 20, y, f"pending boundary: {sorted(pending_boundaries)}", (210, 210, 210), 0.48)
    y += 26

    interesting_keys = [
        "roi_risk",
        "motion_delta",
        "center_closeness",
        "looming_score",
        "occlusion_score",
    ]
    for key in interesting_keys:
        value = feature_map.get(key, 0.0)
        draw_text(cv2_module, panel, 20, y, f"{key}: {value:.3f}", (255, 210, 120), 0.5)
        y += 22
    y += 10

    span = 120
    start = max(0, frame_idx - span + 1)
    rows = review_data.frame_rows[start : frame_idx + 1]
    draw_series(
        cv2_module,
        np_module,
        panel,
        x=20,
        y=y,
        w=580,
        h=110,
        values=[float(item.get("scores", {}).get("total_importance", 0.0)) for item in rows],
        color=(70, 220, 255),
        label="total_importance",
        vmin=0.0,
        vmax=1.0,
        ref_line=0.55,
    )
    y += 140
    draw_series(
        cv2_module,
        np_module,
        panel,
        x=20,
        y=y,
        w=580,
        h=110,
        values=[float(item.get("scores", {}).get("boundary", 0.0)) for item in rows],
        color=(110, 255, 120),
        label="boundary",
        vmin=0.0,
        vmax=1.0,
        ref_line=0.85,
    )
    y += 140
    draw_series(
        cv2_module,
        np_module,
        panel,
        x=20,
        y=y,
        w=580,
        h=110,
        values=[float(item.get("derived_feature", {}).get("feature_vector", [0.0] * 16)[14]) for item in rows],
        color=(255, 170, 70),
        label="looming_score",
        vmin=0.0,
        vmax=1.0,
        ref_line=0.35,
    )
    y += 132

    controls = [
        "space 재생/정지",
        "a,d 1프레임 / j,l 30프레임 이동",
        "1~6 라벨 선택",
        "z 시작 / x 끝 / b boundary 토글",
        "c 구간 추가 / r 마지막 구간 삭제",
        "w 저장 / q 종료",
    ]
    for item in controls:
        draw_text(cv2_module, panel, 20, y, item, (170, 170, 170), 0.46)
        y += 20

    draw_text(cv2_module, panel, 20, target_h - 18, message, (120, 220, 255), 0.48)
    return canvas


def run_ui(args: ReviewArgs) -> dict[str, Any]:
    """브레이크 전환 구간 검수 UI 실행."""
    cv2_module, np_module = _import_cv2_np()
    review_data = load_review_data(args)
    segments = load_existing_segments(review_data.annotation_path)

    capture = cv2_module.VideoCapture(str(review_data.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"비디오를 열 수 없습니다: {review_data.video_path}")

    total_frames = min(
        int(capture.get(cv2_module.CAP_PROP_FRAME_COUNT)),
        len(review_data.frame_rows),
    )
    if total_frames <= 0:
        raise RuntimeError("사용 가능한 프레임이 없습니다.")

    state = {
        "frame": max(0, min(args.start_frame, total_frames - 1)),
        "playing": False,
        "selected_label_idx": 0,
        "mark_start": None,
        "mark_end": None,
        "pending_boundaries": set(),
        "message": "ready",
    }

    def read_frame(frame_index: int) -> Any:
        capture.set(cv2_module.CAP_PROP_POS_FRAMES, float(frame_index))
        ret, frame = capture.read()
        if not ret:
            return None
        return frame

    frame = read_frame(state["frame"])
    if frame is None:
        raise RuntimeError("첫 프레임을 읽지 못했습니다.")

    if args.export_preview is not None:
        preview = render_frame(
            cv2_module=cv2_module,
            np_module=np_module,
            frame_bgr=frame,
            review_data=review_data,
            frame_idx=state["frame"],
            segments=segments,
            selected_label_idx=state["selected_label_idx"],
            mark_start=state["mark_start"],
            mark_end=state["mark_end"],
            pending_boundaries=state["pending_boundaries"],
            message=state["message"],
        )
        export_path = args.export_preview
        if not export_path.is_absolute():
            export_path = review_data.run_dir / export_path
        export_path.parent.mkdir(parents=True, exist_ok=True)
        cv2_module.imwrite(str(export_path), preview)
        capture.release()
        return {
            "mode": "preview",
            "preview_path": str(export_path),
            "run_dir": str(review_data.run_dir),
            "annotation_path": str(review_data.annotation_path),
        }

    window_name = "Brake Review UI"
    cv2_module.namedWindow(window_name, cv2_module.WINDOW_NORMAL)
    cv2_module.resizeWindow(window_name, 1560, 760)

    while True:
        frame_idx = max(0, min(state["frame"], total_frames - 1))
        frame = read_frame(frame_idx)
        if frame is None:
            break

        dashboard = render_frame(
            cv2_module=cv2_module,
            np_module=np_module,
            frame_bgr=frame,
            review_data=review_data,
            frame_idx=frame_idx,
            segments=segments,
            selected_label_idx=state["selected_label_idx"],
            mark_start=state["mark_start"],
            mark_end=state["mark_end"],
            pending_boundaries=state["pending_boundaries"],
            message=state["message"],
        )
        cv2_module.imshow(window_name, dashboard)
        delay = max(1, int(1000.0 / args.play_fps))
        key = cv2_module.waitKey(delay) & 0xFF

        if key == ord("q"):
            break
        if key == ord(" "):
            state["playing"] = not state["playing"]
            state["message"] = "playing" if state["playing"] else "paused"
        elif key == ord("a"):
            state["frame"] = max(0, frame_idx - 1)
        elif key == ord("d"):
            state["frame"] = min(total_frames - 1, frame_idx + 1)
        elif key == ord("j"):
            state["frame"] = max(0, frame_idx - 30)
        elif key == ord("l"):
            state["frame"] = min(total_frames - 1, frame_idx + 30)
        elif key == ord("z"):
            state["mark_start"] = frame_idx
            state["message"] = f"mark_start={frame_idx}"
        elif key == ord("x"):
            state["mark_end"] = frame_idx
            state["message"] = f"mark_end={frame_idx}"
        elif key == ord("b"):
            if frame_idx in state["pending_boundaries"]:
                state["pending_boundaries"].remove(frame_idx)
            else:
                state["pending_boundaries"].add(frame_idx)
            state["message"] = f"boundary={sorted(state['pending_boundaries'])}"
        elif key == ord("c"):
            if state["mark_start"] is None or state["mark_end"] is None:
                state["message"] = "start/end mark가 필요합니다."
            else:
                start_frame = min(state["mark_start"], state["mark_end"])
                end_frame = max(state["mark_start"], state["mark_end"])
                label = review_data.labels[state["selected_label_idx"]]
                boundary_frames = sorted(
                    {item for item in state["pending_boundaries"] if start_frame <= item <= end_frame}
                    | {start_frame}
                )
                segments.append(
                    ReviewSegment(
                        start_frame=start_frame,
                        end_frame=end_frame,
                        context_tag=label.name,
                        boundary_frames=boundary_frames,
                        event_active=label.event_active,
                        status="reviewed",
                        confidence="high",
                        notes="brake_review_ui",
                    )
                )
                state["message"] = f"segment added: {label.name} {start_frame}-{end_frame}"
                state["mark_start"] = None
                state["mark_end"] = None
                state["pending_boundaries"] = set()
        elif key == ord("r"):
            if segments:
                removed = segments.pop()
                state["message"] = f"removed: {removed.context_tag} {removed.start_frame}-{removed.end_frame}"
        elif key == ord("w"):
            save_segments(
                annotation_path=review_data.annotation_path,
                run_id=review_data.run_dir.name,
                video_path=review_data.video_path,
                segments=segments,
            )
            state["message"] = f"saved: {review_data.annotation_path.name}"
        elif ord("1") <= key <= ord(str(min(9, len(review_data.labels)))):
            state["selected_label_idx"] = int(chr(key)) - 1
            label = review_data.labels[state["selected_label_idx"]]
            state["message"] = f"selected={label.name}"

        if state["playing"]:
            state["frame"] = min(total_frames - 1, state["frame"] + 1)

    capture.release()
    cv2_module.destroyAllWindows()
    return {
        "run_dir": str(review_data.run_dir),
        "video_path": str(review_data.video_path),
        "annotation_path": str(review_data.annotation_path),
        "segment_count": len(segments),
    }


def main() -> None:
    args = parse_args()
    result = run_ui(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
