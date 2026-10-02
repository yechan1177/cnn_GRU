"""comma.ai speedchallenge 영상 → 64×64 소형 프레임 배열(VLA-lite 실주행 개루프 평가용).

입력: `data/raw/external/comma_speedchallenge/train.mp4` (20400 프레임, 640×480, 20fps)
출력: `data/processed/comma_speedchallenge/frames_64.npz`
- `frames`: uint8 [N, H, W, 3] RGB (기본 [20400, 64, 64, 3])
- `crop`: int32 [4] = (y0, y1, x0, x1), 원본 픽셀 좌표의 잘라낼 영역(끝 미포함)
- `size`: int32 [2] = (H, W)
- `frame_id`: int32 [N] = 0..N-1 (디코딩 순서, `comma_table.npz`의 `frame_id`와 같은 인덱스)

crop 기본값 (y0, y1, x0, x1) = (100, 360, 0, 640)의 근거
- 프레임 0, 5000, 10000, 17000(고속도로 고가도로 아래, 정체 고속도로, 도심 STOP 교차로)을 PNG로 확인했다.
- y ≥ 약 360부터는 모든 장면에서 대시보드·보닛 반사(검은 영역)로 정보가 없다 → 아래쪽 120 px을 버린다.
- 지평선(소실점)은 장면에 따라 y ≈ 210~240에 있고, 그 위 하늘은 주행 결정과 거의 무관하다.
  다만 신호등·표지판·고가도로가 y ≈ 120~200에 보이므로 위쪽은 100 px만 버린다(맨 위 약 20 px은 비네팅으로 검다).
- 옆 차로 차량(끼어들기)과 차선이 좌우 끝까지 보이므로 가로는 자르지 않는다.
- 결과 영역은 640×260(가로:세로 ≈ 2.5:1)이며 64×64로 줄일 때 가로가 더 많이 압축된다(PilotNet류 관행).
  잘린 영역에서 지평선은 위에서 약 45% 높이에 온다.

축소는 `cv2.INTER_AREA`(면적 평균, 축소 시 앨리어싱이 적다)를 쓴다.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_VIDEO = "data/raw/external/comma_speedchallenge/train.mp4"
DEFAULT_OUT = "data/processed/comma_speedchallenge/frames_64.npz"
DEFAULT_CROP: tuple[int, int, int, int] = (100, 360, 0, 640)  # (y0, y1, x0, x1)
DEFAULT_SIZE: tuple[int, int] = (64, 64)  # (H, W)


def crop_resize(frame_bgr: np.ndarray, crop: tuple[int, int, int, int], size: tuple[int, int]) -> np.ndarray:
    """BGR 원본 프레임을 crop 후 size(H, W)로 줄이고 RGB uint8로 반환한다."""

    import cv2

    y0, y1, x0, x1 = crop
    h, w = frame_bgr.shape[:2]
    if not (0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
        raise ValueError(f"crop {crop}가 프레임 크기 {(h, w)}를 벗어납니다.")
    roi = frame_bgr[y0:y1, x0:x1]
    small = cv2.resize(roi, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


def extract(
    video: Path,
    out: Path,
    crop: tuple[int, int, int, int] = DEFAULT_CROP,
    size: tuple[int, int] = DEFAULT_SIZE,
    max_frames: int | None = None,
) -> Path:
    """영상을 처음부터 순서대로 디코딩해 소형 프레임 npz를 저장하고 경로를 반환한다."""

    import cv2

    if not video.exists():
        raise FileNotFoundError(f"영상이 없습니다: {video}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {video}")
    n_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    logger.info("영상 %s: 메타 프레임 수 %d, %.1f fps, crop=%s, size=%s", video, n_meta, fps, crop, size)
    frames: list[np.ndarray] = []
    t0 = time.perf_counter()
    try:
        while max_frames is None or len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(crop_resize(frame, crop, size))
            if len(frames) % 5000 == 0:
                logger.info("  %d 프레임 처리 (%.1fs)", len(frames), time.perf_counter() - t0)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError("디코딩된 프레임이 없습니다.")
    if max_frames is None and n_meta > 0 and len(frames) != n_meta:
        logger.warning("디코딩 프레임 수 %d가 메타데이터 %d와 다릅니다.", len(frames), n_meta)
    arr = np.stack(frames).astype(np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        frames=arr,
        crop=np.asarray(crop, dtype=np.int32),
        size=np.asarray(size, dtype=np.int32),
        frame_id=np.arange(len(arr), dtype=np.int32),
        fps=np.float32(fps),
    )
    logger.info("저장: %s %s (%.1fs)", out, arr.shape, time.perf_counter() - t0)
    return out


def check_alignment(frames_path: Path, table_path: Path) -> bool:
    """frames_64.npz의 프레임 수·순서가 comma_table.npz의 frame_id와 맞는지 확인한다.

    테이블 i행은 frames[table.frame_id[i]]에 대응한다. frame_id가 모두 프레임 범위 안이고,
    테이블이 전체 프레임을 순서대로 담고 있으면(frame_id == arange) True를 반환한다.
    """

    fr = np.load(frames_path)
    tb = np.load(table_path)
    n = fr["frames"].shape[0]
    fid = tb["frame_id"]
    in_range = bool(fid.min() >= 0 and fid.max() < n)
    identity = bool(len(fid) == n and np.array_equal(fid, np.arange(n)))
    logger.info("정렬 확인: 프레임 %d, 테이블 %d행, 범위 내=%s, 항등 정렬=%s", n, len(fid), in_range, identity)
    return in_range and identity


def save_montage(frames_path: Path, out_png: Path, ids: tuple[int, ...] = (0, 5000, 10000, 17000), scale: int = 4) -> Path:
    """샘플 프레임을 가로로 이어 붙이고(최근접 확대) 프레임 번호를 적은 PNG를 저장한다."""

    import cv2

    fr = np.load(frames_path)["frames"]
    tiles = []
    for i in ids:
        tile = cv2.resize(fr[i], None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        cv2.putText(tile, f"#{i}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(np.pad(tile, ((4, 4), (4, 4), (0, 0)), constant_values=255))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), np.concatenate(tiles, axis=1))
    logger.info("몽타주 저장: %s", out_png)
    return out_png


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="comma speedchallenge 영상을 64×64 소형 프레임 npz로 변환")
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--crop", type=int, nargs=4, default=list(DEFAULT_CROP), metavar=("Y0", "Y1", "X0", "X1"))
    parser.add_argument("--size", type=int, nargs=2, default=list(DEFAULT_SIZE), metavar=("H", "W"))
    parser.add_argument("--max-frames", type=int, default=None, help="디버그용 최대 프레임 수")
    parser.add_argument("--table", default="data/processed/comma_speedchallenge/comma_table.npz", help="정렬 확인용 테이블")
    parser.add_argument("--montage", default=None, help="샘플 몽타주 PNG 경로(예: docs/assets/comma_frames_64_sample.png)")
    args = parser.parse_args()
    out = extract(Path(args.video), Path(args.out), tuple(args.crop), tuple(args.size), args.max_frames)
    table = Path(args.table)
    if table.exists():
        if not check_alignment(out, table):
            logger.warning("프레임과 테이블 frame_id 정렬이 항등이 아닙니다. frame_id로 인덱싱하세요.")
    else:
        logger.info("테이블 %s가 없어 정렬 확인을 건너뜁니다.", table)
    if args.montage:
        save_montage(out, Path(args.montage))


if __name__ == "__main__":
    main()
