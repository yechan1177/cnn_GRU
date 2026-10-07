from __future__ import annotations

"""comma.ai speedchallenge 실주행 영상 -> 학습용 프레임 테이블.

입력
- `detections_yolo3cls.jsonl`: `vcp.tools.extract_detections` 결과(프레임별 YOLO 박스)
- `train.txt`: 프레임별 차량 속도(20fps, 원 데이터셋은 단위를 명시하지 않음)

라벨(시각 특징과 독립인 속도 센서에서만 계산)
- 속도 s를 0.5초 이동평균, 가속도 a = ds/dt 를 다시 0.5초 이동평균
- stopped: s < 1.0 / braking: a < -0.6 / accelerating: a > +0.6 / cruise: 그 외
- 행동(action) 타깃: 0.5초, 1.0초 뒤 가속도(VLA 연계 보조 head 학습용)

출력: `<out>.npz` (X_v1, X_v2, y, boundary, t, speed, accel, action, frame_id)
      `<out>_10fps.npz` (짝수 프레임만 사용해 특징을 다시 계산: FPS 변화 실데이터 검증용)
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from vcp.features import build_feature_extractor, frame_from_dict

logger = logging.getLogger(__name__)

EGO_LABELS: list[str] = ["cruise", "accelerating", "braking", "stopped"]


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k))
    pad = k // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(k) / k, mode="valid")[: len(x)]


def speed_labels(speed: np.ndarray, fps: float, accel_thr: float = 0.6, stop_thr: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = max(1, int(round(0.5 * fps)) | 1)
    s = moving_average(speed, k)
    a = moving_average(np.gradient(s) * fps, k)
    y = np.zeros(len(speed), dtype=np.int16)
    y[a > accel_thr] = 1
    y[a < -accel_thr] = 2
    y[s < stop_thr] = 3
    return y, s.astype(np.float32), a.astype(np.float32)


def compute_features(frames: list, version: str, conf: float) -> np.ndarray:
    ext = build_feature_extractor(version, conf_threshold=conf, max_det=30)
    return np.asarray([ext.update(f) for f in frames], dtype=np.float32)


def build(det_path: Path, speed_path: Path, out: Path, conf: float = 0.45, fps: float = 20.0) -> None:
    rows = [json.loads(line) for line in det_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda r: r["frame_id"])
    frames = [frame_from_dict(r) for r in rows]
    speed_all = np.asarray([float(x) for x in speed_path.read_text().split()], dtype=np.float32)
    frame_ids = np.asarray([f.frame_id for f in frames])
    if frame_ids.max() >= len(speed_all):
        raise ValueError("검출 프레임 수가 속도 라벨보다 많습니다.")
    y_all, s_all, a_all = speed_labels(speed_all, fps)
    shift_05, shift_10 = int(round(0.5 * fps)), int(round(1.0 * fps))
    a_pad = np.concatenate([a_all, np.full(shift_10, a_all[-1])])
    action_all = np.stack([a_pad[np.arange(len(a_all)) + shift_05], a_pad[np.arange(len(a_all)) + shift_10]], axis=1)

    def pack(sel_frames: list, ids: np.ndarray, path: Path) -> None:
        y = y_all[ids]
        boundary = np.zeros(len(y), dtype=np.int8)
        boundary[1:] = (y[1:] != y[:-1]).astype(np.int8)
        np.savez_compressed(
            path,
            X_v1=compute_features(sel_frames, "v1", conf),
            X_v2=compute_features(sel_frames, "v2", conf),
            y=y,
            boundary=boundary,
            t=np.asarray([f.t for f in sel_frames], dtype=np.float32),
            speed=s_all[ids],
            accel=a_all[ids],
            action=action_all[ids].astype(np.float32),
            frame_id=ids.astype(np.int32),
        )
        logger.info("저장: %s (%d 프레임, 라벨 분포 %s)", path, len(ids), np.bincount(y, minlength=4).tolist())

    out.parent.mkdir(parents=True, exist_ok=True)
    pack(frames, frame_ids, out.parent / f"{out.name}.npz")
    even = [f for f in frames if f.frame_id % 2 == 0]
    pack(even, np.asarray([f.frame_id for f in even]), out.parent / f"{out.name}_10fps.npz")
    meta = {
        "source": "comma.ai speedchallenge data/train.mp4 + train.txt (https://github.com/commaai/speedchallenge)",
        "fps": fps,
        "labels": EGO_LABELS,
        "label_rule": "0.5s 이동평균 속도/가속도, |a|>0.6 → 가속/제동, 속도<1.0 → 정지",
        "conf_threshold": conf,
        "frames": len(frames),
    }
    (out.parent / f"{out.name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="comma speedchallenge 학습 테이블 생성")
    parser.add_argument("--detections", default="data/processed/comma_speedchallenge/detections_yolo3cls.jsonl")
    parser.add_argument("--speed", default="data/raw/external/comma_speedchallenge/train.txt")
    parser.add_argument("--out", default="data/processed/comma_speedchallenge/comma_table")
    parser.add_argument("--conf", type=float, default=0.45)
    args = parser.parse_args()
    build(Path(args.detections), Path(args.speed), Path(args.out), conf=args.conf)


if __name__ == "__main__":
    main()
