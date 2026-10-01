from __future__ import annotations

"""실주행 영상(comma speedchallenge) 자동 실험.

- 과제: 시각 특징(YOLO 박스 기반)만으로 자차 운동 상태(정속/가속/제동/정지) 추정
- 라벨: 속도 센서에서만 계산 → 라벨/특징 순환 없음
- 분할: 17분 단일 영상을 10개 시간 블록으로 나눈 블록 교차검증(test 1, val 1, train 8),
  블록 경계 앞뒤 1초는 평가 대상에서 제외(purge)
- 추가: 20fps 학습 → 10fps(짝수 프레임으로 특징 재계산) 테스트, 합성→실데이터 zero-shot 전이
"""

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import blocked_folds, window_index
from .metrics import auroc, macro_f1, summarize
from .rules import RULES, tune_rule
from .trainer import ModelSpec, fit_temperature, predict, softmax, train_model

logger = logging.getLogger(__name__)
EGO_LABELS = ["cruise", "accelerating", "braking", "stopped"]
EVENT = (2,)


@dataclass(slots=True)
class CommaSuiteConfig:
    out_dir: Path
    data_dir: Path
    seeds: tuple[int, ...] = (0, 1, 2)
    n_blocks: int = 10
    purge_s: float = 1.0
    workers: int = 4
    quick: bool = False


def comma_model_specs(quick: bool = False) -> list[ModelSpec]:
    e = 3 if quick else 20
    kw = {"epochs": e, "patience": 5, "batch_size": 256}
    return [
        ModelSpec("mlp_last_v2", "mlp_last", "v2", **kw),
        ModelSpec("gru_v2", "gru", "v2", **kw),
        ModelSpec("cnn_gru_single_v2", "single_channel_cnn_gru", "v2", **kw),
        ModelSpec("mc_cnn_gru_balanced_v2", "multichannel_cnn_gru", "v2", grouping="balanced", **kw),
        ModelSpec("mc_cnn_gru_semantic_v2", "multichannel_cnn_gru", "v2", grouping="semantic", **kw),
        ModelSpec("mc_cnn_gru_semantic_v2_w16", "multichannel_cnn_gru", "v2", grouping="semantic", window=16, **kw),
        ModelSpec("mc_cnn_gru_balanced_v1", "multichannel_cnn_gru", "v1", grouping="balanced", **kw),
        ModelSpec("mc_cnn_gru_semantic_v1", "multichannel_cnn_gru", "v1", grouping="semantic", **kw),
    ]


def _load(path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(path))


def _fold_masks(n: int, fold: dict[str, np.ndarray], block: np.ndarray) -> dict[str, np.ndarray]:
    keep = fold["purge_mask"]
    return {part: np.isin(block, fold[part]) & keep for part in ("train", "val", "test")}


def _comma_job(args: tuple[ModelSpec, int, dict[str, Any]]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    spec, fold_id, ctx = args
    d20 = _load(Path(ctx["path20"]))
    d10 = _load(Path(ctx["path10"]))
    n = len(d20["y"])
    block, folds = blocked_folds(n, ctx["n_blocks"], ctx["purge"])
    fold = folds[fold_id]
    masks = _fold_masks(n, fold, block)
    X = d20[f"X_{spec.feature_version}"]
    idx_tr = window_index(block, spec.window, stride=1, mask=masks["train"])
    idx_va = window_index(block, spec.window, mask=masks["val"])
    idx_te = window_index(block, spec.window, mask=masks["test"])
    seed = 1000 * fold_id + ctx["seed"]
    res = train_model(spec, X, d20["y"], d20["boundary"], block, idx_tr, idx_va, 4, seed)
    vl, _, _ = predict(res.model, X, idx_va)
    temp = fit_temperature(vl, d20["y"][idx_va[:, -1]])
    out: dict[str, Any] = {"model": spec.name, "fold": fold_id, "seed": ctx["seed"], "params": res.params, "best_epoch": res.best_epoch}
    tl, _, _ = predict(res.model, X, idx_te)
    probs = softmax(tl, temp)
    last = idx_te[:, -1]
    out["test_20fps"] = summarize(d20["y"][last], probs.argmax(1), block[last], d20["t"][last], 4, EVENT, probs=probs)
    out["test_20fps"]["braking_auroc"] = auroc(probs[:, 2], d20["y"][last] == 2)

    # 10fps: 같은 test 블록 프레임(짝수)에서 특징을 다시 계산한 테이블로 평가
    block10 = block[d10["frame_id"]]
    keep10 = fold["purge_mask"][d10["frame_id"]] & np.isin(block10, fold["test"])
    X10 = d10[f"X_{spec.feature_version}"]
    idx10 = window_index(block10, spec.window, mask=keep10)
    tl10, _, _ = predict(res.model, X10, idx10)
    p10 = softmax(tl10, temp)
    last10 = idx10[:, -1]
    out["test_10fps"] = summarize(d10["y"][last10], p10.argmax(1), block10[last10], d10["t"][last10], 4, EVENT, probs=p10)
    return out


def _rule_and_majority(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    d20 = _load(Path(ctx["path20"]))
    n = len(d20["y"])
    block, folds = blocked_folds(n, ctx["n_blocks"], ctx["purge"])
    out = []
    fn, grid, ver = RULES["rule_v2_ego"]
    for k, fold in enumerate(folds):
        m = _fold_masks(n, fold, block)
        maj = int(np.bincount(d20["y"][m["train"]]).argmax())
        sel = m["test"]
        y, g, t = d20["y"][sel], block[sel], d20["t"][sel]
        out.append({"model": "majority", "fold": k, "seed": 0, "params": 0, "test_20fps": summarize(y, np.full(len(y), maj), g, t, 4, EVENT)})
        params, _ = tune_rule(fn, grid, d20[f"X_{ver}"][m["val"]], d20["y"][m["val"]], EGO_LABELS)
        pred = fn(d20[f"X_{ver}"][sel], EGO_LABELS, params)
        out.append({"model": "rule_v2_ego", "fold": k, "seed": 0, "params": 0, "test_20fps": summarize(y, pred, g, t, 4, EVENT), "rule_params": params})
    return out


def _sim_to_real(ctx: dict[str, Any], synth_main: Path, quick: bool) -> dict[str, Any]:
    """합성 데이터로만 학습한 6맥락 모델의 실영상 제동 구간 판별력(zero-shot AUROC)."""

    import torch

    from ..sim.dataset import load_dataset
    from .data import split_groups

    torch.set_num_threads(4)
    data, meta = load_dataset(synth_main)
    labels = meta["labels"]
    groups = np.array([e["index"] for e in meta["episodes"]])
    split = split_groups(groups, 0.15, 0.15, 2026)
    d20 = _load(Path(ctx["path20"]))
    is_brake = d20["y"] == 2
    moving = d20["y"] != 3
    out: dict[str, Any] = {}
    w = 8
    for ver, grouping in (("v2", "semantic"), ("v1", "balanced")):
        spec = ModelSpec(f"synth_{grouping}_{ver}", "multichannel_cnn_gru", ver, grouping=grouping, epochs=3 if quick else 10)
        m_tr = np.isin(data["ep"], split["train"])
        m_va = np.isin(data["ep"], split["val"])
        res = train_model(
            spec,
            data[f"X_{ver}"],
            data["y"],
            data["boundary"],
            data["ep"],
            window_index(data["ep"], w, stride=2, mask=m_tr),
            window_index(data["ep"], w, mask=m_va),
            len(labels),
            seed=0,
        )
        idx = window_index(np.zeros(len(d20["y"]), dtype=np.int64), w)
        logits, _, _ = predict(res.model, d20[f"X_{ver}"], idx)
        probs = softmax(logits)
        score = probs[:, labels.index("brake_warning")] + probs[:, labels.index("hard_brake_risk")]
        out[spec.name] = {
            "braking_auroc_all": auroc(score, is_brake),
            "braking_auroc_moving": auroc(score[moving], is_brake[moving]),
        }
    inv_ttc = d20["X_v2"][:, 10]
    out["feature_only_lead_inv_ttc"] = {
        "braking_auroc_all": auroc(inv_ttc, is_brake),
        "braking_auroc_moving": auroc(inv_ttc[moving], is_brake[moving]),
    }
    return out


def run_comma_suite(cfg: CommaSuiteConfig, synth_main: Path | None = None) -> Path:
    started = time.perf_counter()
    path20 = cfg.data_dir / "comma_table.npz"
    path10 = cfg.data_dir / "comma_table_10fps.npz"
    if not path20.exists():
        raise FileNotFoundError(f"{path20} 가 없습니다. 먼저 vcp.tools.build_comma_dataset을 실행하세요.")
    d20 = _load(path20)
    ctx_base = {"path20": str(path20), "path10": str(path10), "n_blocks": cfg.n_blocks, "purge": int(round(cfg.purge_s * 20))}
    records = _rule_and_majority(ctx_base)
    specs = comma_model_specs(cfg.quick)
    folds = range(2) if cfg.quick else range(cfg.n_blocks)
    seeds = cfg.seeds[:1]
    jobs = [(spec, k, {**ctx_base, "seed": s}) for spec in specs for k in folds for s in seeds]
    logger.info("comma 스위트: %d 작업", len(jobs))
    with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
        for rec in pool.map(_comma_job, jobs):
            records.append(rec)
    logger.info("comma 블록 교차검증 완료")

    sim2real = None
    synth_main = synth_main or Path("data/processed/synth/main_15fps_mid")
    if (synth_main.parent / f"{synth_main.name}.npz").exists():
        sim2real = _sim_to_real(ctx_base, synth_main, cfg.quick)

    out = {
        "suite": "comma_speedchallenge_ego_state",
        "labels": EGO_LABELS,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "dataset": {"frames": int(len(d20["y"])), "label_counts": np.bincount(d20["y"], minlength=4).tolist(), "fps": 20},
        "records": records,
        "sim_to_real": sim2real,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    path = cfg.out_dir / "comma_results.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    logger.info("저장: %s", path)
    return path


__all__ = ["CommaSuiteConfig", "run_comma_suite", "macro_f1"]
