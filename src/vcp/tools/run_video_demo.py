from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from vcp.components import (
    CompositeEventScorer,
    JsonlDatasetExporter,
    RingBufferCurationEngine,
    SimpleFeaturePacker,
    build_spatial_encoder,
    build_temporal_encoder,
)
from vcp.config import PipelineConfig, load_config
from vcp.schemas import FramePacket, FrameRecord

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="race.mp4 실시간 모델 테스트/시각화 실행")
    parser.add_argument(
        "--video",
        type=str,
        default="data/raw/videos/race.mp4",
        help="입력 비디오 경로",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/runtime_rtx3080ti_yolo_gru.yaml",
        help="파이프라인 설정 파일",
    )
    parser.add_argument("--display", dest="display", action="store_true", help="실시간 창 표시")
    parser.add_argument("--no-display", dest="display", action="store_false", help="실시간 창 비표시")
    parser.set_defaults(display=True)
    parser.add_argument("--save-video", dest="save_video", action="store_true", help="오버레이 결과 영상 저장")
    parser.add_argument("--no-save-video", dest="save_video", action="store_false", help="결과 영상 미저장")
    parser.set_defaults(save_video=True)
    parser.add_argument("--max-frames", type=int, default=0, help="처리할 최대 프레임 수(0이면 전체)")
    parser.add_argument("--run-name", type=str, default="race_video_demo", help="출력 run 이름 접두사")
    return parser.parse_args()


def _import_cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:  # pragma: no cover - 환경 의존
        raise ModuleNotFoundError(
            "run_video_demo 실행에는 opencv-python이 필요합니다. `pip install opencv-python`"
        ) from exc
    return cv2


def _extract_pixels(cv2_module: Any, frame_bgr: Any) -> list[float]:
    gray = cv2_module.cvtColor(frame_bgr, cv2_module.COLOR_BGR2GRAY)
    small = cv2_module.resize(gray, (8, 4), interpolation=cv2_module.INTER_AREA)
    return [round(float(value) / 255.0, 6) for value in small.reshape(-1).tolist()]


def _draw_overlay(cv2_module: Any, frame_bgr: Any, overlay: dict[str, Any]) -> Any:
    frame = frame_bgr.copy()
    font = cv2_module.FONT_HERSHEY_SIMPLEX
    color_main = (40, 255, 40)
    color_sub = (255, 220, 120)
    color_warn = (90, 170, 255)

    def _class_color(class_id: int) -> tuple[int, int, int]:
        palette = [
            (80, 220, 80),
            (60, 180, 255),
            (255, 180, 60),
            (200, 120, 255),
            (255, 120, 120),
            (120, 240, 240),
            (180, 255, 120),
            (255, 150, 220),
        ]
        return palette[class_id % len(palette)]

    detections = overlay.get("detections", {})
    boxes = detections.get("boxes", []) if isinstance(detections, dict) else []
    for box in boxes:
        xyxy = box.get("xyxy", [])
        if not isinstance(xyxy, list) or len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = [int(value) for value in xyxy]
        cls_id = int(box.get("cls_id", 0))
        cls_name = str(box.get("cls_name", f"class_{cls_id}"))
        conf = float(box.get("conf", 0.0))
        color = _class_color(cls_id)

        cv2_module.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2_module.getTextSize(label, font, 0.5, 1)
        label_top = max(0, y1 - th - 8)
        cv2_module.rectangle(frame, (x1, label_top), (x1 + tw + 8, label_top + th + 6), color, -1)
        cv2_module.putText(
            frame,
            label,
            (x1 + 4, label_top + th + 1),
            font,
            0.5,
            (20, 20, 20),
            1,
            cv2_module.LINE_AA,
        )

    lines = [
        f"frame={overlay['frame_id']} context={overlay['context_tag']} policy={overlay['policy']}",
        (
            "score"
            f" c={overlay['scores']['context']:.3f}"
            f" b={overlay['scores']['boundary']:.3f}"
            f" u={overlay['scores']['uncertainty']:.3f}"
            f" n={overlay['scores']['novelty']:.3f}"
            f" t={overlay['scores']['total_importance']:.3f}"
        ),
        (
            f"events={overlay['events']} fps={overlay['fps']:.2f} "
            f"det={int(detections.get('count', 0))} "
            f"mconf={float(detections.get('mean_conf', 0.0)):.2f}"
        ),
    ]

    y = 28
    for idx, text in enumerate(lines):
        cv2_module.putText(
            frame,
            text,
            (16, y),
            font,
            0.6 if idx == 0 else 0.55,
            color_main if idx == 0 else color_sub,
            2,
            cv2_module.LINE_AA,
        )
        y += 30

    top_contexts = overlay.get("top_contexts", [])
    bar_x = 16
    bar_y = y + 8
    bar_w_max = 220
    bar_h = 16
    cv2_module.putText(
        frame,
        "top-context probs",
        (bar_x, bar_y),
        font,
        0.5,
        color_warn,
        1,
        cv2_module.LINE_AA,
    )
    bar_y += 12
    for label, prob in top_contexts[:3]:
        prob_value = max(0.0, min(1.0, float(prob)))
        fill_w = int(bar_w_max * prob_value)
        cv2_module.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w_max, bar_y + bar_h), (60, 60, 60), 1)
        cv2_module.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (90, 220, 255), -1)
        cv2_module.putText(
            frame,
            f"{label}: {prob_value:.2f}",
            (bar_x + bar_w_max + 8, bar_y + 13),
            font,
            0.45,
            color_sub,
            1,
            cv2_module.LINE_AA,
        )
        bar_y += bar_h + 8

    return frame


def _write_experiment_files(summary: dict[str, Any], cfg: PipelineConfig, args: argparse.Namespace) -> None:
    exp_dir = Path("experiments") / "exp_006_race_video_demo"
    exp_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = {
        "video": str(args.video),
        "config": str(args.config),
        "run_name": str(args.run_name),
        "display": bool(args.display),
        "save_video": bool(args.save_video),
        "max_frames": int(args.max_frames),
        "pipeline": cfg.to_dict(),
    }
    (exp_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(config_snapshot, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    metrics_row = {
        "frames": int(summary["frames"]),
        "events": int(summary["events"]),
        "keyframes": int(summary["policy_counts"].get("keyframe", 0)),
        "clips": int(summary["policy_counts"].get("clip", 0)),
        "discard": int(summary["policy_counts"].get("discard", 0)),
        "det_total": int(summary["detection_count_total"]),
        "det_avg_per_frame": float(summary["detection_count_avg_per_frame"]),
        "det_nonzero_ratio": float(summary["detection_nonzero_frame_ratio"]),
        "avg_fps": float(summary["avg_fps"]),
    }
    with (exp_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics_row.keys()))
        writer.writeheader()
        writer.writerow(metrics_row)

    notes = [
        "# exp_006_race_video_demo 노트",
        "",
        f"- 입력 영상: `{summary['video_path']}`",
        f"- spatial 모델: `{summary['spatial_model']}`",
        f"- temporal 모델: `{summary['temporal_model']}`",
        f"- 평균 처리 FPS: `{summary['avg_fps']:.2f}`",
        f"- 프레임당 평균 검출 수: `{summary['detection_count_avg_per_frame']:.3f}`",
        f"- 검출 발생 프레임 비율: `{summary['detection_nonzero_frame_ratio']:.3f}`",
        "- 실시간 창에서 `q` 키로 중단 가능",
    ]
    (exp_dir / "notes.md").write_text("\n".join(notes), encoding="utf-8")

    result_lines = [
        "# race.mp4 실시간 테스트 결과",
        "",
        f"- frames: `{summary['frames']}`",
        f"- events: `{summary['events']}`",
        f"- avg_fps: `{summary['avg_fps']:.2f}`",
        f"- det_total: `{summary['detection_count_total']}`",
        f"- det_avg_per_frame: `{summary['detection_count_avg_per_frame']:.3f}`",
        f"- det_nonzero_ratio: `{summary['detection_nonzero_frame_ratio']:.3f}`",
        f"- run_dir: `{summary['run_dir']}`",
        f"- exports_dir: `{summary['exports_dir']}`",
        f"- preview_video: `{summary['preview_video']}`",
        f"- screenshot: `{summary['screenshot']}`",
    ]
    (exp_dir / "result_summary.md").write_text("\n".join(result_lines), encoding="utf-8")


def _update_experiment_log(summary: dict[str, Any]) -> None:
    log_path = Path("experiments") / "experiment_log.md"
    if not log_path.exists():
        return

    line = (
        f"| {datetime.now().strftime('%Y-%m-%d')} | exp_006_race_video_demo | {summary['video_path']} | "
        f"race.mp4 실시간 추론/시각화 검증 | {summary['frames']} frame, {summary['events']} event, "
        f"avg_fps {summary['avg_fps']:.2f} | run_id={summary['run_id']} |"
    )
    current = log_path.read_text(encoding="utf-8").rstrip() + "\n"
    if "exp_006_race_video_demo" in current:
        return
    log_path.write_text(current + line + "\n", encoding="utf-8")


def run_demo(args: argparse.Namespace) -> dict[str, Any]:
    cv2 = _import_cv2()

    config_path = Path(args.config)
    cfg = load_config(config_path)
    cfg.storage.run_name = str(args.run_name)

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"비디오를 열 수 없습니다: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0:
        fps = float(max(1, cfg.camera.fps))

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    spatial = build_spatial_encoder(cfg.spatial)
    packer = SimpleFeaturePacker()
    temporal = build_temporal_encoder(cfg.temporal, cfg.labels, max(1, cfg.spatial.feature_dim))
    scorer = CompositeEventScorer(cfg.scoring)
    curation = RingBufferCurationEngine(cfg.curation, cfg.scoring)
    exporter = JsonlDatasetExporter(cfg.storage)

    preview_path: str | None = None
    writer = None
    if args.save_video:
        preview_root = Path(cfg.storage.base_dir) / "runs"
        preview_root.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        preview_file = preview_root / f"{cfg.storage.run_name}_preview_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(preview_file), fourcc, fps, (width, height))
        preview_path = str(preview_file)

    frame_count = 0
    event_count = 0
    policy_counts: dict[str, int] = {"clip": 0, "keyframe": 0, "discard": 0}
    detection_count_total = 0
    detection_count_nonzero_frames = 0
    screenshot_path: str | None = None

    start_time = time.perf_counter()

    try:
        while True:
            ret, frame_bgr = capture.read()
            if not ret:
                break

            if args.max_frames > 0 and frame_count >= int(args.max_frames):
                break

            frame_packet = FramePacket(
                frame_id=frame_count,
                sensor_timestamp=frame_count / fps,
                system_timestamp=time.time(),
                pixels=_extract_pixels(cv2, frame_bgr),
                raw_path=None,
                image=frame_bgr,
            )

            spatial_vector = spatial.encode(frame_packet)
            if hasattr(spatial, "get_feature_keys"):
                feature_keys = list(spatial.get_feature_keys())
            else:
                feature_keys = [f"feature_{idx}" for idx in range(len(spatial_vector))]
            if hasattr(spatial, "get_last_detection"):
                detections = spatial.get_last_detection()
            else:
                detections = {
                    "count": 0,
                    "boxes": [],
                    "orig_shape": None,
                    "mean_conf": 0.0,
                    "max_conf": 0.0,
                }
            det_count = int(detections.get("count", 0))
            detection_count_total += det_count
            if det_count > 0:
                detection_count_nonzero_frames += 1

            packed = packer.pack(frame_packet, spatial_vector)
            temporal_out = temporal.encode(packed)
            score = scorer.score(packed, temporal_out)
            context_tag = max(temporal_out.context_probs, key=temporal_out.context_probs.get)
            top_contexts = sorted(
                temporal_out.context_probs.items(), key=lambda item: item[1], reverse=True
            )[:3]

            frame_record = FrameRecord(
                frame_id=frame_packet.frame_id,
                sensor_timestamp=frame_packet.sensor_timestamp,
                system_timestamp=frame_packet.system_timestamp,
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
                    "feature_vector": [round(value, 6) for value in spatial_vector],
                    "feature_sample": [round(value, 6) for value in spatial_vector[:8]],
                    "feature_keys": feature_keys,
                },
                raw_path=f"{video_path.name}#frame={frame_count}",
                metadata={
                    "source_video": str(video_path),
                    "spatial_model": cfg.spatial.model_name,
                    "temporal_model": cfg.temporal.model_name,
                    "detection_count": det_count,
                    "detection_mean_conf": float(detections.get("mean_conf", 0.0)),
                },
            )

            decision = curation.process_frame(frame_record, score)
            exporter.add_frame(frame_record, decision.storage_policy)
            policy_counts[decision.storage_policy] = policy_counts.get(decision.storage_policy, 0) + 1

            if decision.finalized_event is not None:
                exporter.add_event(decision.finalized_event)
                event_count += 1

            elapsed = max(1e-6, time.perf_counter() - start_time)
            live_fps = (frame_count + 1) / elapsed

            overlay = {
                "frame_id": frame_count,
                "context_tag": context_tag,
                "policy": decision.storage_policy,
                "scores": frame_record.scores,
                "events": event_count,
                "fps": live_fps,
                "detections": detections,
                "top_contexts": top_contexts,
            }
            rendered = _draw_overlay(cv2, frame_bgr, overlay)

            if screenshot_path is None:
                screenshot_dir = Path("artifacts") / "screenshots"
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                screenshot_file = screenshot_dir / f"race_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(screenshot_file), rendered)
                screenshot_path = str(screenshot_file)

            if writer is not None:
                writer.write(rendered)

            if args.display:
                try:
                    cv2.imshow("VCP Race Demo", rendered)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                except Exception as exc:  # pragma: no cover - GUI 환경 의존
                    logger.warning("실시간 창 표시를 비활성화합니다: %s", exc)
                    args.display = False

            frame_count += 1

        for event in curation.flush():
            exporter.add_event(event)
            event_count += 1

        export_result = exporter.finalize()

    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    total_time = max(1e-6, time.perf_counter() - start_time)
    avg_fps = frame_count / total_time

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "video_path": str(video_path),
        "frames": frame_count,
        "events": event_count,
        "avg_fps": round(avg_fps, 4),
        "policy_counts": policy_counts,
        "detection_count_total": int(detection_count_total),
        "detection_count_avg_per_frame": round(detection_count_total / max(1, frame_count), 4),
        "detection_nonzero_frame_ratio": round(
            detection_count_nonzero_frames / max(1, frame_count), 4
        ),
        "run_id": export_result["run_id"],
        "run_dir": export_result["run_dir"],
        "exports_dir": export_result["exports_dir"],
        "preview_video": preview_path,
        "screenshot": screenshot_path,
        "spatial_model": cfg.spatial.model_name,
        "temporal_model": cfg.temporal.model_name,
    }

    summary_path = Path(export_result["run_dir"]) / "race_demo_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_experiment_files(summary=summary, cfg=cfg, args=args)
    _update_experiment_log(summary)

    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    summary = run_demo(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
