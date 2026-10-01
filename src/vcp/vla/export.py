from __future__ import annotations

"""VLA 학습용 에피소드 내보내기(LeRobot v2 디렉터리 구조를 따름).

구조::

    <out>/meta/info.json        # fps, feature 스키마, 에피소드/프레임 수
    <out>/meta/tasks.jsonl      # 과업 지시문(언어)
    <out>/meta/episodes.jsonl   # 에피소드별 길이/과업/출처
    <out>/data/chunk-000/episode_000000.parquet   # pyarrow가 없으면 .jsonl

주의: LeRobot 공식 라이브러리로 로딩 검증을 하지 않았으므로 '호환 지향' 형식이다.
영상 프레임은 복사하지 않고 원본 영상 경로 + 프레임 번호로 참조한다(`observation.video_ref`).
합성 데이터는 박스 수준 시뮬레이션이라 이미지가 없으며, 검출 박스 자체를 관측으로 저장한다.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VLAEpisode:
    """한 에피소드의 (관측, 언어, 행동) 시퀀스."""

    episode_index: int
    task_ko: str
    task_en: str
    fps: float
    source: str
    timestamp: np.ndarray  # [T]
    state: np.ndarray  # [T, S] 자차 상태(속도, 가속도)
    features: np.ndarray  # [T, 16] 의미 특징
    context_probs: np.ndarray  # [T, K] 맥락 모델 확률
    action: np.ndarray  # [T, A] 행동(미래 가감속)
    context: list[str]  # [T] 맥락 라벨 이름(모델 예측)
    narration_ko: list[str]
    narration_en: list[str]
    event_score: np.ndarray  # [T]
    uncertainty: np.ndarray  # [T]
    boxes: list[list[list[float]]] | None = None  # [T][N][6]
    video_ref: dict[str, Any] | None = None  # {"path": str, "frame_ids": [...]}
    extra: dict[str, Any] = field(default_factory=dict)


def _rows(ep: VLAEpisode, global_offset: int, task_index: int) -> list[dict[str, Any]]:
    rows = []
    frame_ids = (ep.video_ref or {}).get("frame_ids")
    for i in range(len(ep.timestamp)):
        row: dict[str, Any] = {
            "timestamp": float(ep.timestamp[i]),
            "frame_index": i,
            "episode_index": ep.episode_index,
            "index": global_offset + i,
            "task_index": task_index,
            "observation.state": [float(v) for v in ep.state[i]],
            "observation.semantic_features": [float(v) for v in ep.features[i]],
            "observation.context_probs": [float(v) for v in ep.context_probs[i]],
            "action": [float(v) for v in ep.action[i]],
            "annotation.context": ep.context[i],
            "annotation.narration_ko": ep.narration_ko[i],
            "annotation.narration_en": ep.narration_en[i],
            "curation.event_score": float(ep.event_score[i]),
            "curation.uncertainty": float(ep.uncertainty[i]),
        }
        if ep.boxes is not None:
            row["observation.boxes"] = json.dumps(ep.boxes[i])
        if frame_ids is not None:
            row["observation.video_frame"] = int(frame_ids[i])
        rows.append(row)
    return rows


def export_lerobot_like(episodes: list[VLAEpisode], out_dir: Path, name: str, context_labels: list[str]) -> Path:
    """에피소드 목록을 LeRobot v2 구조로 저장한다."""

    out_dir = Path(out_dir)
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    chunk = out_dir / "data" / "chunk-000"
    chunk.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        use_parquet = True
    except ModuleNotFoundError:  # pragma: no cover - 환경 의존
        use_parquet = False

    tasks: dict[tuple[str, str], int] = {}
    episodes_meta = []
    offset = 0
    for ep in episodes:
        key = (ep.task_ko, ep.task_en)
        task_index = tasks.setdefault(key, len(tasks))
        rows = _rows(ep, offset, task_index)
        stem = f"episode_{ep.episode_index:06d}"
        if use_parquet:
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, chunk / f"{stem}.parquet")
        else:
            with (chunk / f"{stem}.jsonl").open("w", encoding="utf-8") as file:
                for row in rows:
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
        episodes_meta.append(
            {
                "episode_index": ep.episode_index,
                "tasks": [ep.task_en],
                "tasks_ko": [ep.task_ko],
                "length": len(rows),
                "source": ep.source,
                **({"video": ep.video_ref.get("path")} if ep.video_ref else {}),
                **ep.extra,
            }
        )
        offset += len(rows)

    first = episodes[0] if episodes else None
    info = {
        "codebase_version": "v2.0-compatible (vcp export, 공식 로더 미검증)",
        "dataset_name": name,
        "fps": first.fps if first else None,
        "total_episodes": len(episodes),
        "total_frames": offset,
        "total_tasks": len(tasks),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}." + ("parquet" if use_parquet else "jsonl"),
        "features": {
            "observation.state": {"dtype": "float32", "shape": [int(first.state.shape[1])] if first else [0], "names": ["ego_speed", "ego_accel"]},
            "observation.semantic_features": {"dtype": "float32", "shape": [int(first.features.shape[1])] if first else [0]},
            "observation.context_probs": {"dtype": "float32", "shape": [len(context_labels)], "names": context_labels},
            "action": {"dtype": "float32", "shape": [int(first.action.shape[1])] if first else [0], "names": ["accel_t+0.5s", "accel_t+1.0s"]},
            "annotation.context": {"dtype": "string"},
            "annotation.narration_ko": {"dtype": "string"},
            "annotation.narration_en": {"dtype": "string"},
            "curation.event_score": {"dtype": "float32"},
            "curation.uncertainty": {"dtype": "float32"},
        },
    }
    (out_dir / "meta" / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    with (out_dir / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as file:
        for (ko, en), idx in tasks.items():
            file.write(json.dumps({"task_index": idx, "task": en, "task_ko": ko}, ensure_ascii=False) + "\n")
    with (out_dir / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as file:
        for row in episodes_meta:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("VLA 데이터셋 내보내기: %s (%d 에피소드, %d 프레임)", out_dir, len(episodes), offset)
    return out_dir
