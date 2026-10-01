from __future__ import annotations

"""합성 6맥락 벤치마크 자동 실험.

절차
1. 데이터: 메인(15fps, mid 노이즈) + 강건성 테스트(10/30fps, high 노이즈, 같은 테스트 시드)
2. 에피소드 단위 train/val/test 분할(누수 없음)
3. 모델 × 시드 학습 → 검증셋으로만 임계값/온도/평활화 선택 → 테스트 1회 평가
4. 결과 JSON 저장(논문 표/그림은 이 파일에서 생성)
"""

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..sim.dataset import SynthConfig, generate_dataset, load_dataset
from .data import split_groups, window_index
from .metrics import auroc, bootstrap_ci, boundary_f1_tolerance, summarize
from .rules import RULES, hybrid_gate_v1, tune_hybrid, tune_rule
from .trainer import ModelSpec, ema_smooth, fit_temperature, predict, softmax, train_model

logger = logging.getLogger(__name__)

EVENT_CLASS_NAMES = ("brake_warning", "hard_brake_risk")


@dataclass(slots=True)
class SuiteConfig:
    out_dir: Path
    data_dir: Path
    n_episodes: int = 700
    duration_s: float = 30.0
    seeds: tuple[int, ...] = (0, 1, 2)
    split_seed: int = 2026
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    train_stride: int = 2
    workers: int = 4
    quick: bool = False


def default_model_specs(quick: bool = False) -> list[ModelSpec]:
    epochs = 3 if quick else 14
    specs = [
        ModelSpec("mlp_last_v2", "mlp_last", "v2", epochs=epochs),
        ModelSpec("gru_v2", "gru", "v2", epochs=epochs),
        ModelSpec("cnn_gru_single_v2", "single_channel_cnn_gru", "v2", epochs=epochs),
        ModelSpec("mc_cnn_gru_balanced_v2", "multichannel_cnn_gru", "v2", grouping="balanced", epochs=epochs),
        ModelSpec("mc_cnn_gru_semantic_v2", "multichannel_cnn_gru", "v2", grouping="semantic", epochs=epochs),
        ModelSpec("mc_cnn_gru_semantic_v2_focal", "multichannel_cnn_gru", "v2", grouping="semantic", loss="cb_focal", epochs=epochs),
        ModelSpec("mc_cnn_gru_semantic_v2_w16", "multichannel_cnn_gru", "v2", grouping="semantic", window=16, epochs=epochs),
        ModelSpec("mc_cnn_gru_semantic_v2_ema", "multichannel_cnn_gru", "v2", grouping="semantic", ema_smoothing=True, epochs=epochs),
        ModelSpec("mc_cnn_gru_balanced_v1", "multichannel_cnn_gru", "v1", grouping="balanced", epochs=epochs),
        ModelSpec("mc_cnn_gru_balanced_v1_hybrid", "multichannel_cnn_gru", "v1", grouping="balanced", hybrid_gate=True, epochs=epochs),
        ModelSpec("mc_cnn_gru_semantic_v1", "multichannel_cnn_gru", "v1", grouping="semantic", epochs=epochs),
    ]
    return specs


# ----------------------------------------------------------------------
def ensure_datasets(cfg: SuiteConfig) -> dict[str, Path]:
    """메인/강건성 데이터셋이 없으면 생성한다(시드 고정 → 재현 가능)."""

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    n = 60 if cfg.quick else cfg.n_episodes
    main = cfg.data_dir / "main_15fps_mid"
    if not (cfg.data_dir / "main_15fps_mid.npz").exists():
        generate_dataset(main, SynthConfig(n_episodes=n, fps=15.0, duration_s=cfg.duration_s, noise_level=1.0, seed_base=0, workers=cfg.workers), export_jsonl_episodes=3)
    _, meta = load_dataset(main)
    groups = np.array([e["index"] for e in meta["episodes"]])
    split = split_groups(groups, cfg.val_ratio, cfg.test_ratio, cfg.split_seed)
    test_seeds = [meta["episodes"][int(i)]["seed"] for i in split["test"]]
    paths = {"main": main}
    variants = {
        "test_10fps_mid": (10.0, 1.0),
        "test_30fps_mid": (30.0, 1.0),
        "test_15fps_high": (15.0, 2.0),
        "test_15fps_low": (15.0, 0.5),
    }
    for name, (fps, level) in variants.items():
        path = cfg.data_dir / name
        if not (cfg.data_dir / f"{name}.npz").exists():
            generate_dataset(path, SynthConfig(fps=fps, duration_s=cfg.duration_s, noise_level=level, workers=cfg.workers), seeds=test_seeds)
        paths[name] = path
    return paths


def _job(args: tuple[ModelSpec, int, dict[str, Any]]) -> dict[str, Any]:
    """(모델, 시드) 1개 학습 + 모든 테스트셋 평가. 프로세스 풀에서 실행."""

    import torch

    torch.set_num_threads(1)
    spec, seed, ctx = args
    data, meta = load_dataset(Path(ctx["main"]))
    labels: list[str] = meta["labels"]
    n_classes = len(labels)
    event_classes = tuple(labels.index(c) for c in EVENT_CLASS_NAMES)
    X = data[f"X_{spec.feature_version}"]
    split = {k: np.asarray(v) for k, v in ctx["split"].items()}
    mask = {k: np.isin(data["ep"], v) for k, v in split.items()}
    idx_train = window_index(data["ep"], spec.window, stride=ctx["train_stride"], mask=mask["train"])
    idx_val = window_index(data["ep"], spec.window, mask=mask["val"])

    res = train_model(spec, X, data["y"], data["boundary"], data["ep"], idx_train, idx_val, n_classes, seed)
    val_logits, val_bnd, _ = predict(res.model, X, idx_val)
    y_val = data["y"][idx_val[:, -1]]
    temperature = fit_temperature(val_logits, y_val)
    val_probs = softmax(val_logits, temperature)
    val_bprob = 1.0 / (1.0 + np.exp(-val_bnd))
    g_val = data["ep"][idx_val[:, -1]]

    # 검증셋에서만 후처리 파라미터 선택
    chosen: dict[str, Any] = {"temperature": temperature}
    alpha = 1.0
    if spec.ema_smoothing:
        best = -1.0
        for a in (1.0, 0.6, 0.4, 0.25):
            from .metrics import macro_f1

            s = macro_f1(y_val, ema_smooth(val_probs, g_val, a).argmax(1), n_classes)
            if s > best:
                best, alpha = s, a
        chosen["ema_alpha"] = alpha
    hybrid_params = None
    if spec.hybrid_gate:
        hybrid_params, _ = tune_hybrid(val_probs, val_bprob, X[idx_val[:, -1]], y_val, labels)
        chosen["hybrid_params"] = hybrid_params
    bthr_best, bthr = -1.0, 0.5
    for thr in (0.3, 0.5, 0.7, 0.85, 0.95):
        f = boundary_f1_tolerance(data["boundary"][idx_val[:, -1]], val_bprob, g_val, thr)["f1"]
        if f > bthr_best:
            bthr_best, bthr = f, thr
    chosen["boundary_threshold"] = bthr

    results: dict[str, Any] = {}
    for test_name, test_path in ctx["tests"].items():
        tdata, _ = load_dataset(Path(test_path))
        Xt = tdata[f"X_{spec.feature_version}"]
        if test_name == "main":
            tmask = mask["test"]
            idx_t = window_index(tdata["ep"], spec.window, mask=tmask)
        else:
            idx_t = window_index(tdata["ep"], spec.window)
        logits, bnd, _ = predict(res.model, Xt, idx_t)
        probs = softmax(logits, temperature)
        last = idx_t[:, -1]
        g = tdata["ep"][last]
        if spec.ema_smoothing:
            probs = ema_smooth(probs, g, alpha)
        pred = probs.argmax(1)
        bprob = 1.0 / (1.0 + np.exp(-bnd))
        if hybrid_params is not None:
            pred = hybrid_gate_v1(probs, bprob, Xt[last], labels, hybrid_params)
        y_t = tdata["y"][last]
        summary = summarize(y_t, pred, g, tdata["t"][last], n_classes, event_classes, probs=probs)
        summary["boundary"] = boundary_f1_tolerance(tdata["boundary"][last], bprob, g, bthr)
        if test_name == "main":
            summary["ci95"] = bootstrap_ci(y_t, pred, g, n_classes, n_boot=200, seed=seed, event_classes=event_classes)
            err = (pred != y_t).astype(int)
            entropy = -(probs * np.log(np.clip(probs, 1e-12, 1))).sum(1)
            summary["error_auroc_entropy"] = auroc(entropy, err)
            # 큐레이션: 이벤트 점수(브레이크 확률 합) + 불확실성 상위 k% 선택 시 이벤트 프레임 회수율
            ev_score = probs[:, list(event_classes)].sum(1) + 0.5 * entropy / np.log(n_classes)
            is_event = np.isin(y_t, event_classes)
            curve = {}
            for budget in (0.05, 0.1, 0.2, 0.3):
                k = max(1, int(len(ev_score) * budget))
                top = np.argsort(-ev_score)[:k]
                curve[str(budget)] = float(is_event[top].sum() / max(1, is_event.sum()))
            summary["curation_event_recall"] = curve
            summary["confusion"] = np.bincount(y_t * n_classes + pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes).tolist()
        results[test_name] = summary

    return {
        "model": spec.name,
        "spec": spec.to_dict(),
        "seed": seed,
        "params": res.params,
        "best_epoch": res.best_epoch,
        "train_seconds": round(res.train_seconds, 2),
        "history": res.history,
        "chosen": chosen,
        "results": results,
    }


def _rule_jobs(paths: dict[str, Path], split: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    data, meta = load_dataset(paths["main"])
    labels = meta["labels"]
    n_classes = len(labels)
    event_classes = tuple(labels.index(c) for c in EVENT_CLASS_NAMES)
    mask_val = np.isin(data["ep"], split["val"])
    mask_test = np.isin(data["ep"], split["test"])
    out = []
    # 다수 클래스 baseline
    majority = int(np.bincount(data["y"][np.isin(data["ep"], split["train"])]).argmax())
    rows: dict[str, Any] = {}
    for test_name, path in paths.items():
        tdata, _ = load_dataset(path)
        sel = mask_test if test_name == "main" else np.ones(len(tdata["y"]), dtype=bool)
        pred = np.full(int(sel.sum()), majority)
        s = summarize(tdata["y"][sel], pred, tdata["ep"][sel], tdata["t"][sel], n_classes, event_classes)
        if test_name == "main":
            s["ci95"] = bootstrap_ci(tdata["y"][sel], pred, tdata["ep"][sel], n_classes, n_boot=200, event_classes=event_classes)
        rows[test_name] = s
    out.append({"model": "majority", "seed": 0, "params": 0, "results": rows, "chosen": {"class": labels[majority]}})

    for name in ("rule_v1", "rule_v2"):
        fn, grid, ver = RULES[name]
        params, val_score = tune_rule(fn, grid, data[f"X_{ver}"][mask_val], data["y"][mask_val], labels)
        rows = {}
        for test_name, path in paths.items():
            tdata, _ = load_dataset(path)
            sel = mask_test if test_name == "main" else np.ones(len(tdata["y"]), dtype=bool)
            pred = fn(tdata[f"X_{ver}"][sel], labels, params)
            s = summarize(tdata["y"][sel], pred, tdata["ep"][sel], tdata["t"][sel], n_classes, event_classes)
            if test_name == "main":
                s["ci95"] = bootstrap_ci(tdata["y"][sel], pred, tdata["ep"][sel], n_classes, n_boot=200, event_classes=event_classes)
            rows[test_name] = s
        out.append({"model": name, "seed": 0, "params": 0, "results": rows, "chosen": {"rule_params": params, "val_macro_f1": val_score}})
    return out


def run_synthetic_suite(cfg: SuiteConfig, specs: list[ModelSpec] | None = None) -> Path:
    started = time.perf_counter()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    paths = ensure_datasets(cfg)
    data, meta = load_dataset(paths["main"])
    groups = np.array([e["index"] for e in meta["episodes"]])
    split = split_groups(groups, cfg.val_ratio, cfg.test_ratio, cfg.split_seed)
    specs = specs or default_model_specs(cfg.quick)

    records: list[dict[str, Any]] = _rule_jobs(paths, split)
    ctx = {
        "main": str(paths["main"]),
        "split": {k: v.tolist() for k, v in split.items()},
        "train_stride": cfg.train_stride,
        "tests": {k: str(v) for k, v in paths.items()},
    }
    jobs = [(spec, seed, ctx) for spec in specs for seed in cfg.seeds]
    logger.info("합성 스위트: 모델 %d × 시드 %d = %d 작업", len(specs), len(cfg.seeds), len(jobs))
    with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
        for rec in pool.map(_job, jobs):
            logger.info("완료: %s seed=%s main macroF1=%.4f", rec["model"], rec["seed"], rec["results"]["main"]["macro_f1"])
            records.append(rec)

    label_counts = {
        part: np.bincount(data["y"][np.isin(data["ep"], ids)], minlength=len(meta["labels"])).tolist()
        for part, ids in split.items()
    }
    scenario_counts: dict[str, int] = {}
    for e in meta["episodes"]:
        scenario_counts[e["scenario"]] = scenario_counts.get(e["scenario"], 0) + 1
    contact = sum(1 for e in meta["episodes"] if e.get("collisions_steps", 0) > 0)
    out = {
        "suite": "synthetic_6context",
        "labels": meta["labels"],
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "dataset": {
            "episodes": len(meta["episodes"]),
            "frames": meta["n_frames"],
            "split_episodes": {k: len(v) for k, v in split.items()},
            "label_counts": label_counts,
            "scenario_counts": scenario_counts,
            "episodes_with_contact": contact,
        },
        "records": records,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }
    out_path = cfg.out_dir / "synthetic_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    logger.info("저장: %s (%.1f s)", out_path, out["elapsed_s"])
    return out_path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))
