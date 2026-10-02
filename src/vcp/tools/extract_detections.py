from __future__ import annotations

"""영상에서 YOLO 검출 결과를 프레임 단위 JSONL로 저장하는 도구.

검출(무거운 단계)과 특징 계산(가벼운 단계)을 분리하기 위해 사용한다.
낮은 confidence 임계값으로 한 번만 검출해 두면, 이후 특징 버전/임계값/FPS를
바꿔 가며 재검출 없이 실험할 수 있다.

출력 형식(한 줄 = 한 프레임)::

    {"frame_id": 0, "t": 0.0, "width": 640, "height": 480,
     "boxes": [[x1, y1, x2, y2, cls_id, conf], ...]}
"""

import argparse
import json
import logging
import time
from pathlib import Path

from vcp.utils.device import resolve_device

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="영상 -> 프레임별 YOLO 검출 JSONL")
    parser.add_argument("--video", required=True, help="입력 영상 경로")
    parser.add_argument("--weights", default="models/checkpoints/yolo3cls_best.pt")
    parser.add_argument("--output", required=True, help="출력 JSONL 경로")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf", type=float, default=0.25, help="저장 최소 confidence")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1, help="N프레임마다 1회 검출")
    parser.add_argument("--max-frames", type=int, default=0, help="0이면 전체")
    parser.add_argument("--resume", action="store_true", help="기존 출력 이어쓰기")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    import cv2
    from ultralytics import YOLO

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"영상을 찾을 수 없습니다: {video_path}")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[int] = set()
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    done.add(int(json.loads(line)["frame_id"]))
        logger.info("resume: 기존 %d 프레임 건너뜀", len(done))

    device = resolve_device(args.device)
    model = YOLO(args.weights)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 20.0

    mode = "a" if args.resume else "w"
    frame_id = -1
    written = 0
    started = time.perf_counter()
    with out_path.open(mode, encoding="utf-8") as out:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_id += 1
            if args.max_frames and frame_id >= args.max_frames:
                break
            if frame_id % max(1, args.stride) != 0 or frame_id in done:
                continue
            result = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                max_det=args.max_det,
                verbose=False,
            )[0]
            h, w = result.orig_shape
            boxes: list[list[float]] = []
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().tolist()
                cls = result.boxes.cls.cpu().tolist()
                conf = result.boxes.conf.cpu().tolist()
                for (x1, y1, x2, y2), c, s in zip(xyxy, cls, conf, strict=True):
                    boxes.append([round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2), int(c), round(s, 4)])
            row = {
                "frame_id": frame_id,
                "t": round(frame_id / fps, 6),
                "width": int(w),
                "height": int(h),
                "boxes": boxes,
            }
            out.write(json.dumps(row) + "\n")
            written += 1
            if written % 500 == 0:
                out.flush()
                elapsed = time.perf_counter() - started
                logger.info("%d 프레임 처리 (%.1f ms/frame)", written, 1000 * elapsed / written)
    capture.release()
    elapsed = time.perf_counter() - started
    logger.info("완료: %d 프레임, %.1f s, 출력=%s", written, elapsed, out_path)


if __name__ == "__main__":
    main()
