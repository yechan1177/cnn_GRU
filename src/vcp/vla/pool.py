from __future__ import annotations

"""스타일 지시문이 붙은 큐레이션 풀(에피소드 묶음) 생성·로드.

각 에피소드는 `SimEnv`를 내부 전문가 모드(`step(None)`, IDM + 반응 지연)로 끝까지 돌려 만든다.
- 시나리오: `sim.dataset.scenario_for_seed(seed, domain)` (기존 합성 데이터셋과 같은 규칙)
- 스타일·패러프레이즈: 별도 난수 `random.Random(seed * 104729 + 3)`로 결정(시뮬레이터 난수와 독립)
- 특징: `sim.dataset`과 같은 방식(v1/v2, conf 0.45, max_det 30)으로 노이즈 검출에서 계산.
  폐루프와 같은 계산 경로를 보장하려고 `obs.OnlineFeatureTracker`를 쓴다
- v3 플래그(`collision_pushback`, `decouple_initial_speed`)는 `SimEnv`에 그대로 전달하고 meta config에 기록한다.
  기본값(True, False)이면 기존 풀과 같은 값을 만든다. 저장 형식은 바뀌지 않는다
- 렌더링용 GT 박스: 프레임별 가변 개수를 `box_ptr`(CSR 오프셋)/`boxes`로 압축 저장

저장 형식(`<out>.npz` + `<out>.meta.json`, N=전체 프레임, M=전체 GT 박스)
- X_v1, X_v2 [N,16] float32, y [N] int16, boundary [N] int8, ep [N] int32
- t, ego_v, ego_a, ttc, expert_cmd, horizon_y, v_target [N] float32
- box_ptr [N+1] int64, boxes [M,6] float32 = (x1, y1, x2, y2, cls_id, depth_m)
- style_id [N] int8(STYLE_NAMES 순서), paraphrase [N] int8
"""

import json
import logging
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..sim.camera import DetectorNoiseConfig
from ..sim.dataset import scenario_for_seed
from ..sim.env import SimEnv
from ..sim.world import DRIVING, ROBOT
from .instructions import (
    N_PARAPHRASES,
    STYLE_NAMES,
    DrivingStyle,
    instruction_text,
    sample_style,
    styles_for_domain,
)
from .obs import FEATURE_MAX_DET, OnlineFeatureTracker

logger = logging.getLogger(__name__)

_FLOAT_KEYS = ("t", "ego_v", "ego_a", "ttc", "expert_cmd", "horizon_y", "v_target")


@dataclass
class PoolConfig:
    """풀 생성 설정."""

    n_episodes: int
    seed_base: int
    fps: float = 15.0
    duration_s: float = 30.0
    noise_level: float = 1.0
    domain: str = "driving"
    workers: int = 4
    conf_threshold: float = 0.45
    # v3(docs/33): 충돌 시 자차 되밀기(B3), 스타일 초기 속도 분리(B1). 기본값은 기존 동작
    collision_pushback: bool = True
    decouple_initial_speed: bool = False


def style_for_seed(seed: int, domain: str) -> tuple[DrivingStyle, int]:
    """시드로 스타일과 패러프레이즈 번호를 결정한다(시뮬레이터 난수와 독립된 별도 난수)."""

    rng = random.Random(int(seed) * 104729 + 3)
    style = sample_style(rng, domain)
    paraphrase = rng.randrange(N_PARAPHRASES)
    return style, paraphrase


def _episode_worker(args: tuple[int, int, PoolConfig]) -> dict[str, Any]:
    index, seed, cfg = args
    scenario = scenario_for_seed(seed, cfg.domain)
    style, paraphrase = style_for_seed(seed, cfg.domain)
    env = SimEnv(
        scenario,
        seed,
        fps=cfg.fps,
        duration_s=cfg.duration_s,
        noise=DetectorNoiseConfig(level=cfg.noise_level),
        style=style,
        collision_pushback=cfg.collision_pushback,
        decouple_initial_speed=cfg.decouple_initial_speed,
    )
    tracker = OnlineFeatureTracker(conf_threshold=cfg.conf_threshold, max_det=FEATURE_MAX_DET)
    x1: list[list[float]] = []
    x2: list[list[float]] = []
    cols: dict[str, list[float]] = {k: [] for k in _FLOAT_KEYS}
    labels: list[int] = []
    box_list: list[np.ndarray] = []
    info = env.reset()
    while True:
        f1, f2 = tracker.update(info.frame)
        x1.append(f1)
        x2.append(f2)
        labels.append(info.label)
        cols["t"].append(info.t)
        cols["ego_v"].append(info.ego_v)
        cols["ego_a"].append(info.ego_a)
        cols["ttc"].append(info.ttc)
        cols["expert_cmd"].append(info.expert_cmd)
        cols["horizon_y"].append(info.horizon_y)
        box_list.append(info.gt_boxes)
        if env.done:
            break
        info = env.step(None)
    n = len(labels)
    cols["v_target"] = [env.v_target] * n
    y = np.asarray(labels, dtype=np.int16)
    boundary = np.zeros_like(y, dtype=np.int8)
    boundary[1:] = (y[1:] != y[:-1]).astype(np.int8)
    counts = np.asarray([len(b) for b in box_list], dtype=np.int64)
    meta = env.summary()
    return {
        "index": index,
        "seed": seed,
        "scenario": scenario,
        "style": style.name,
        "paraphrase": paraphrase,
        "v_target_mps": float(env.v_target),
        "meta": meta,
        "X_v1": np.asarray(x1, dtype=np.float32),
        "X_v2": np.asarray(x2, dtype=np.float32),
        "y": y,
        "boundary": boundary,
        **{k: np.asarray(v, dtype=np.float32) for k, v in cols.items()},
        "box_counts": counts,
        "boxes": np.concatenate(box_list, axis=0).astype(np.float32) if n else np.zeros((0, 6), np.float32),
    }


def _stem(path: Path) -> Path:
    path = Path(path)
    if path.name.endswith(".meta.json"):
        return path.with_name(path.name[: -len(".meta.json")])
    return path.with_suffix("") if path.suffix == ".npz" else path


def generate_pool(out_path: Path, cfg: PoolConfig) -> Path:
    """풀을 병렬 생성해 `<out>.npz` + `<out>.meta.json`으로 저장하고 npz 경로를 반환한다."""

    if cfg.n_episodes <= 0:
        raise ValueError(f"n_episodes는 양수여야 한다: {cfg.n_episodes}")
    stem = _stem(out_path)
    stem.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(i, cfg.seed_base + i, cfg) for i in range(cfg.n_episodes)]
    t0 = time.perf_counter()
    if cfg.workers > 1:
        chunk = max(1, min(16, cfg.n_episodes // (cfg.workers * 4)))
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            results = list(pool.map(_episode_worker, jobs, chunksize=chunk))
    else:
        results = [_episode_worker(job) for job in jobs]
    results.sort(key=lambda r: r["index"])
    sim_s = time.perf_counter() - t0

    style_index = {name: i for i, name in enumerate(STYLE_NAMES)}
    arrays: dict[str, list[np.ndarray]] = {
        k: [] for k in ("X_v1", "X_v2", "y", "boundary", *_FLOAT_KEYS, "ep", "style_id", "paraphrase", "boxes")
    }
    counts: list[np.ndarray] = []
    episodes_meta: list[dict[str, Any]] = []
    for r in results:
        n = len(r["y"])
        for key in ("X_v1", "X_v2", "y", "boundary", *_FLOAT_KEYS, "boxes"):
            arrays[key].append(r[key])
        arrays["ep"].append(np.full(n, r["index"], dtype=np.int32))
        arrays["style_id"].append(np.full(n, style_index[r["style"]], dtype=np.int8))
        arrays["paraphrase"].append(np.full(n, r["paraphrase"], dtype=np.int8))
        counts.append(r["box_counts"])
        episodes_meta.append(
            {
                "index": r["index"],
                "seed": r["seed"],
                "scenario": r["scenario"],
                "style": r["style"],
                "paraphrase": r["paraphrase"],
                "v_target": round(r["v_target_mps"], 4),
                "instruction_en": instruction_text(r["style"], r["v_target_mps"], cfg.domain, r["paraphrase"], "en"),
                "instruction_ko": instruction_text(r["style"], r["v_target_mps"], cfg.domain, r["paraphrase"], "ko"),
                "frames": n,
                **r["meta"],
            }
        )
    box_counts = np.concatenate(counts)
    box_ptr = np.zeros(len(box_counts) + 1, dtype=np.int64)
    np.cumsum(box_counts, out=box_ptr[1:])
    out = {k: np.concatenate(v) for k, v in arrays.items()}
    out["boxes"] = out["boxes"].reshape(-1, 6).astype(np.float32)
    out["box_ptr"] = box_ptr

    npz_path = stem.parent / f"{stem.name}.npz"
    np.savez_compressed(npz_path, **out)
    prof = DRIVING if cfg.domain == "driving" else ROBOT
    meta = {
        "config": asdict(cfg),
        "labels": list(prof.label_names),
        "styles": {name: asdict(st) for name, st in styles_for_domain(cfg.domain).items()},
        "style_names": list(STYLE_NAMES),
        "episodes": episodes_meta,
        "n_frames": int(len(out["y"])),
        "n_boxes": int(len(out["boxes"])),
        "generation_s": round(time.perf_counter() - t0, 2),
        "simulation_s": round(sim_s, 2),
    }
    (stem.parent / f"{stem.name}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info(
        "VLA 풀 저장: %s (%d 에피소드, %d 프레임, 박스 %d개, %.1fs)",
        npz_path,
        len(results),
        meta["n_frames"],
        meta["n_boxes"],
        meta["generation_s"],
    )
    return npz_path


def load_pool(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """`generate_pool` 결과를 읽는다. path는 확장자 없는 경로 또는 `.npz` 경로. `X_v1v2`를 파생한다."""

    stem = _stem(path)
    npz_path = stem.parent / f"{stem.name}.npz"
    meta_path = stem.parent / f"{stem.name}.meta.json"
    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"풀 파일이 없다: {npz_path} / {meta_path}")
    with np.load(npz_path) as npz:
        data = {k: npz[k] for k in npz.files}
    data["X_v1v2"] = np.concatenate([data["X_v1"], data["X_v2"]], axis=1)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return data, meta


def frame_boxes(data: dict[str, np.ndarray], idx: int) -> np.ndarray:
    """전체 프레임 인덱스 idx의 GT 박스 [M,6]."""

    ptr = data["box_ptr"]
    return data["boxes"][ptr[idx] : ptr[idx + 1]]
