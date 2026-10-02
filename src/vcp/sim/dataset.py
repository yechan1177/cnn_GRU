from __future__ import annotations

"""합성 에피소드 데이터셋 생성/저장/로드.

저장 형식(`<name>.npz` + `<name>.meta.json`)
- X_v1, X_v2: [N, 16] float32 프레임별 특징(conf 임계값 적용 후 계산)
- y: [N] int16 6맥락 라벨, boundary: [N] int8 라벨 전환 프레임
- ep: [N] int32 에피소드 인덱스, t / ego_v / ego_a / ttc: [N] float32
"""

import json
import logging
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..features import build_feature_extractor
from .camera import DetectorNoiseConfig
from .labels import CONTEXT_LABELS
from .world import SCENARIO_TYPES, simulate_episode

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SynthConfig:
    """합성 데이터셋 생성 설정."""

    n_episodes: int = 700
    fps: float = 15.0
    duration_s: float = 30.0
    noise_level: float = 1.0
    seed_base: int = 0
    conf_threshold: float = 0.45
    workers: int = 4
    domain: str = "driving"


def scenario_for_seed(seed: int, domain: str = "driving") -> str:
    """시드로부터 시나리오 종류를 결정한다(같은 시드 = 같은 시나리오)."""

    from .world import scenario_types_for_domain

    table = scenario_types_for_domain(domain)
    rng = random.Random(seed * 7919 + 17)
    names = list(table)
    weights = [table[n] for n in names]
    return rng.choices(names, weights=weights, k=1)[0]


def _episode_worker(args: tuple[int, int, SynthConfig, bool]) -> dict[str, Any]:
    index, seed, cfg, keep_frames = args
    scenario = scenario_for_seed(seed, cfg.domain)
    episode = simulate_episode(
        scenario,
        seed=seed,
        fps=cfg.fps,
        duration_s=cfg.duration_s,
        noise=DetectorNoiseConfig(level=cfg.noise_level),
    )
    v1 = build_feature_extractor("v1", conf_threshold=cfg.conf_threshold, max_det=30)
    v2 = build_feature_extractor("v2", conf_threshold=cfg.conf_threshold, max_det=30)
    x1 = np.asarray([v1.update(f) for f in episode.frames], dtype=np.float32)
    x2 = np.asarray([v2.update(f) for f in episode.frames], dtype=np.float32)
    y = np.asarray(episode.labels, dtype=np.int16)
    boundary = np.zeros_like(y, dtype=np.int8)
    boundary[1:] = (y[1:] != y[:-1]).astype(np.int8)
    out: dict[str, Any] = {
        "index": index,
        "seed": seed,
        "scenario": scenario,
        "meta": episode.meta,
        "X_v1": x1,
        "X_v2": x2,
        "y": y,
        "boundary": boundary,
        "t": np.asarray([f.t for f in episode.frames], dtype=np.float32),
        "ego_v": np.asarray(episode.ego_v, dtype=np.float32),
        "ego_a": np.asarray(episode.ego_a, dtype=np.float32),
        "ttc": np.asarray(episode.ttc, dtype=np.float32),
    }
    if keep_frames:
        out["frames"] = [
            {
                "frame_id": f.frame_id,
                "t": f.t,
                "width": f.width,
                "height": f.height,
                "boxes": [
                    [round(b.x1, 2), round(b.y1, 2), round(b.x2, 2), round(b.y2, 2), b.cls_id, round(b.conf, 4)]
                    for b in f.boxes
                ],
                "label": CONTEXT_LABELS[int(lbl)],
            }
            for f, lbl in zip(episode.frames, episode.labels, strict=True)
        ]
    return out


def generate_dataset(
    out_path: Path,
    cfg: SynthConfig,
    seeds: list[int] | None = None,
    export_jsonl_episodes: int = 0,
) -> Path:
    """에피소드를 병렬 생성해 npz + meta.json으로 저장한다."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seed_list = seeds if seeds is not None else [cfg.seed_base + i for i in range(cfg.n_episodes)]
    jobs = [(i, s, cfg, i < export_jsonl_episodes) for i, s in enumerate(seed_list)]
    if cfg.workers > 1:
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            results = list(pool.map(_episode_worker, jobs, chunksize=8))
    else:
        results = [_episode_worker(job) for job in jobs]
    results.sort(key=lambda r: r["index"])

    keys = ("X_v1", "X_v2", "y", "boundary", "t", "ego_v", "ego_a", "ttc")
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in (*keys, "ep")}
    episodes_meta = []
    for r in results:
        n = len(r["y"])
        for key in keys:
            arrays[key].append(r[key])
        arrays["ep"].append(np.full(n, r["index"], dtype=np.int32))
        episodes_meta.append({"index": r["index"], "seed": r["seed"], "scenario": r["scenario"], "frames": n, **r["meta"]})
        if "frames" in r:
            sample_dir = out_path.parent / f"{out_path.name}_samples"
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_path = sample_dir / f"episode_{r['index']:04d}_{r['scenario']}.jsonl"
            with sample_path.open("w", encoding="utf-8") as file:
                for row in r["frames"]:
                    file.write(json.dumps(row) + "\n")

    npz_path = out_path.parent / f"{out_path.name}.npz"
    np.savez_compressed(npz_path, **{k: np.concatenate(v) for k, v in arrays.items()})
    meta = {
        "config": asdict(cfg),
        "labels": CONTEXT_LABELS,
        "episodes": episodes_meta,
        "n_frames": int(sum(len(r["y"]) for r in results)),
    }
    (out_path.parent / f"{out_path.name}.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    logger.info("합성 데이터셋 저장: %s (%d 에피소드, %d 프레임)", npz_path, len(results), meta["n_frames"])
    return npz_path


def load_dataset(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """`generate_dataset` 결과를 읽는다(path는 확장자 제외 경로)."""

    path = Path(path)
    data = dict(np.load(path.parent / f"{path.name}.npz"))
    if "X_v1" in data and "X_v2" in data:
        data["X_v1v2"] = np.concatenate([data["X_v1"], data["X_v2"]], axis=1)
    meta = json.loads((path.parent / f"{path.name}.meta.json").read_text(encoding="utf-8"))
    return data, meta
