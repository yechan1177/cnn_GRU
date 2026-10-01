from __future__ import annotations

"""피지컬 AI / 로봇 / VLA 연계 실험.

A. 행동 보조 head(실데이터): 같은 경량 모델이 맥락과 함께 미래 가감속(행동)을 예측하는지
B. 자동 큐레이션 효율(실데이터): 저장 예산 대비 제동 이벤트 회수율(모델 점수 vs 무작위)
C. 로봇(AMR) 도메인: 같은 파이프라인의 실내 이동로봇 적용, 주행→로봇 전이(zero-shot / 10% 미세조정)
D. VLA 에피소드 내보내기: (관측, 언어, 행동) LeRobot 구조 데이터셋 생성
"""

import json
import logging
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ..sim.dataset import SynthConfig, generate_dataset, load_dataset
from ..sim.world import ROBOT
from ..vla.curation import event_coverage, select_clips
from ..vla.export import VLAEpisode, export_lerobot_like
from ..vla.language import narrate, task_instruction
from .data import blocked_folds, split_groups, window_index
from .metrics import macro_f1, summarize
from .trainer import ModelSpec, build_model, fit_temperature, predict, softmax, train_model

logger = logging.getLogger(__name__)
EGO_LABELS = ["cruise", "accelerating", "braking", "stopped"]


def future_action(accel: np.ndarray, group: np.ndarray, fps: float, horizons: tuple[float, ...] = (0.5, 1.0)) -> np.ndarray:
    """그룹(에피소드) 안에서 h초 뒤 가속도를 행동 타깃으로 만든다(끝부분은 마지막 값 유지)."""

    out = np.zeros((len(accel), len(horizons)), dtype=np.float32)
    for g in np.unique(group):
        sel = np.where(group == g)[0]
        a = accel[sel]
        for j, h in enumerate(horizons):
            shift = int(round(h * fps))
            idx = np.minimum(np.arange(len(sel)) + shift, len(sel) - 1)
            out[sel, j] = a[idx]
    return out


# ----------------------------------------------------------------------
# A + B: comma 실데이터
# ----------------------------------------------------------------------
def _comma_action_job(args: tuple[int, bool, dict[str, Any]]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    fold_id, with_action, ctx = args
    d = dict(np.load(ctx["path20"]))
    n = len(d["y"])
    block, folds = blocked_folds(n, ctx["n_blocks"], ctx["purge"])
    fold = folds[fold_id]
    m = {p: np.isin(block, fold[p]) & fold["purge_mask"] for p in ("train", "val", "test")}
    spec = ModelSpec(
        "mc_cnn_gru_semantic_v2" + ("_action" if with_action else ""),
        "multichannel_cnn_gru",
        "v2",
        grouping="semantic",
        epochs=ctx["epochs"],
        patience=5,
        batch_size=256,
        action_weight=1.0 if with_action else 0.0,
    )
    X = d["X_v2"]
    action = d["action"]
    idx_tr = window_index(block, 8, mask=m["train"])
    idx_va = window_index(block, 8, mask=m["val"])
    idx_te = window_index(block, 8, mask=m["test"])
    res = train_model(spec, X, d["y"], d["boundary"], block, idx_tr, idx_va, 4, seed=fold_id, action=action if with_action else None)
    vl, _, _ = predict(res.model, X, idx_va)
    temp = fit_temperature(vl, d["y"][idx_va[:, -1]])
    tl, _, act = predict(res.model, X, idx_te)
    probs = softmax(tl, temp)
    last = idx_te[:, -1]
    y = d["y"][last]
    out: dict[str, Any] = {
        "fold": fold_id,
        "model": spec.name,
        "params": res.params,
        "context": summarize(y, probs.argmax(1), block[last], d["t"][last], 4, (2,), probs=probs),
    }
    target = action[last]
    train_mean = action[idx_tr[:, -1]].mean(axis=0)
    out["action_mae_zero"] = np.abs(target).mean(axis=0).tolist()
    out["action_mae_train_mean"] = np.abs(target - train_mean).mean(axis=0).tolist()
    if act is not None:
        out["action_mae_model"] = np.abs(target - act).mean(axis=0).tolist()
        out["action_corr_model"] = [float(np.corrcoef(target[:, j], act[:, j])[0, 1]) for j in range(target.shape[1])]
    # 큐레이션: 테스트 블록에서 2초 클립 선택
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1))).sum(1) / np.log(4)
    score = probs[:, 2] + 0.5 * entropy
    is_event = y == 2
    g = block[last]
    curation = {}
    for budget in (0.1, 0.2, 0.3):
        sel = select_clips(score, g, clip_len=40, budget_ratio=budget)
        rand = [event_coverage(select_clips(score, g, 40, budget, rng=np.random.default_rng(s), random_baseline=True), is_event, g) for s in range(20)]
        curation[str(budget)] = {
            "model": event_coverage(sel, is_event, g),
            "random_frame_coverage": float(np.mean([r["frame_coverage"] for r in rand])),
            "random_event_coverage": float(np.mean([r["event_coverage"] for r in rand])),
        }
    out["curation"] = curation
    if with_action:
        out["export"] = {
            "frame_id": d["frame_id"][last].tolist(),
            "probs": np.round(probs, 4).tolist(),
            "action_pred": np.round(act, 4).tolist() if act is not None else None,
        }
    return out


def run_comma_vla(comma_dir: Path, quick: bool, workers: int = 4) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path20 = comma_dir / "comma_table.npz"
    ctx = {"path20": str(path20), "n_blocks": 10, "purge": 20, "epochs": 3 if quick else 20}
    folds = range(2) if quick else range(10)
    jobs = [(k, w, ctx) for k in folds for w in (False, True)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        recs = list(pool.map(_comma_action_job, jobs))
    exports = [r.pop("export") for r in recs if "export" in r]
    return {"records": recs}, exports


# ----------------------------------------------------------------------
# C: 로봇 도메인
# ----------------------------------------------------------------------
def ensure_robot_dataset(data_dir: Path, quick: bool) -> Path:
    path = data_dir / "robot_main_15fps_mid"
    if not (data_dir / f"{path.name}.npz").exists():
        generate_dataset(path, SynthConfig(n_episodes=60 if quick else 500, fps=15.0, noise_level=1.0, seed_base=50_000, domain="robot", workers=4), export_jsonl_episodes=2)
    return path


def _robot_job(args: tuple[ModelSpec, int, dict[str, Any]]) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    spec, seed, ctx = args
    data, meta = load_dataset(Path(ctx["robot"]))
    n_classes = len(meta["labels"])
    split = {k: np.asarray(v) for k, v in ctx["split"].items()}
    train_eps = split["train"] if not ctx.get("few_shot") else split["train"][: max(2, len(split["train"]) // 10)]
    X = data[f"X_{spec.feature_version}"]
    m_tr = np.isin(data["ep"], train_eps)
    m_va = np.isin(data["ep"], split["val"])
    m_te = np.isin(data["ep"], split["test"])
    idx_tr = window_index(data["ep"], spec.window, stride=2, mask=m_tr)
    idx_va = window_index(data["ep"], spec.window, mask=m_va)
    idx_te = window_index(data["ep"], spec.window, mask=m_te)
    init_state = None
    if ctx.get("init_from"):
        init_state = torch.load(ctx["init_from"], map_location="cpu", weights_only=True)
    if init_state is not None and ctx.get("zero_shot"):
        model = build_model(spec, X.shape[1], n_classes)
        model.load_state_dict(init_state)
        model.eval()
        params = sum(p.numel() for p in model.parameters())
    else:
        if init_state is not None:
            # 사전학습 가중치로 시작하는 미세조정: train_model 내부 초기화 후 덮어쓰기
            spec_ft = ModelSpec(**{**spec.to_dict(), "lr": 5e-4})
            res = _train_with_init(spec_ft, X, data, idx_tr, idx_va, n_classes, seed, init_state)
        else:
            res = train_model(spec, X, data["y"], data["boundary"], data["ep"], idx_tr, idx_va, n_classes, seed)
        model = res.model
        params = res.params
    vl, _, _ = predict(model, X, idx_va)
    temp = fit_temperature(vl, data["y"][idx_va[:, -1]]) if not ctx.get("zero_shot") else 1.0
    tl, _, _ = predict(model, X, idx_te)
    probs = softmax(tl, temp)
    last = idx_te[:, -1]
    ev = (2, 3)
    s = summarize(data["y"][last], probs.argmax(1), data["ep"][last], data["t"][last], n_classes, ev, probs=probs)
    return {"model": ctx.get("tag", spec.name), "seed": seed, "params": params, "test": s}


def _train_with_init(spec: ModelSpec, X: np.ndarray, data: dict[str, np.ndarray], idx_tr: np.ndarray, idx_va: np.ndarray, n_classes: int, seed: int, init_state: dict[str, Any]) -> Any:
    import torch

    from . import trainer as tr

    original = tr.build_model

    def patched(s: ModelSpec, input_dim: int, k: int, action_dim: int = 0) -> torch.nn.Module:
        model = original(s, input_dim, k, action_dim)
        model.load_state_dict(init_state)
        return model

    tr.build_model = patched
    try:
        return tr.train_model(spec, X, data["y"], data["boundary"], data["ep"], idx_tr, idx_va, n_classes, seed)
    finally:
        tr.build_model = original


def run_robot(data_dir: Path, out_dir: Path, quick: bool, workers: int = 4) -> dict[str, Any]:
    import torch

    robot = ensure_robot_dataset(data_dir, quick)
    data, meta = load_dataset(robot)
    groups = np.array([e["index"] for e in meta["episodes"]])
    split = split_groups(groups, 0.15, 0.15, 2026)
    e = 3 if quick else 14
    specs = [
        ModelSpec("mlp_last_v2", "mlp_last", "v2", epochs=e),
        ModelSpec("gru_v2", "gru", "v2", epochs=e),
        ModelSpec("cnn_gru_single_v2", "single_channel_cnn_gru", "v2", epochs=e),
        ModelSpec("mc_cnn_gru_balanced_v2", "multichannel_cnn_gru", "v2", grouping="balanced", epochs=e),
        ModelSpec("mc_cnn_gru_semantic_v2", "multichannel_cnn_gru", "v2", grouping="semantic", epochs=e),
        ModelSpec("mc_cnn_gru_balanced_v1", "multichannel_cnn_gru", "v1", grouping="balanced", epochs=e),
    ]
    seeds = (0,) if quick else (0, 1, 2)
    ctx = {"robot": str(robot), "split": {k: v.tolist() for k, v in split.items()}}
    jobs = [(s, seed, ctx) for s in specs for seed in seeds]

    # 주행 사전학습 모델(전이 실험용)
    torch.set_num_threads(4)
    drv, drv_meta = load_dataset(data_dir / "main_15fps_mid")
    dg = np.array([x["index"] for x in drv_meta["episodes"]])
    dsplit = split_groups(dg, 0.15, 0.15, 2026)
    sem = ModelSpec("mc_cnn_gru_semantic_v2", "multichannel_cnn_gru", "v2", grouping="semantic", epochs=e)
    dres = train_model(
        sem,
        drv["X_v2"],
        drv["y"],
        drv["boundary"],
        drv["ep"],
        window_index(drv["ep"], 8, stride=2, mask=np.isin(drv["ep"], dsplit["train"])),
        window_index(drv["ep"], 8, mask=np.isin(drv["ep"], dsplit["val"])),
        6,
        seed=0,
    )
    pre_path = out_dir.parent / "checkpoints" / "driving_semantic_v2_pretrain.pt"
    pre_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dres.model.state_dict(), pre_path)
    for seed in seeds:
        jobs.append((sem, seed, {**ctx, "init_from": str(pre_path), "zero_shot": True, "tag": "driving_pretrained_zero_shot"}))
        jobs.append((sem, seed, {**ctx, "few_shot": True, "tag": "robot_10pct_scratch"}))
        jobs.append((sem, seed, {**ctx, "few_shot": True, "init_from": str(pre_path), "tag": "robot_10pct_finetune_from_driving"}))

    with ProcessPoolExecutor(max_workers=workers) as pool:
        recs = list(pool.map(_robot_job, jobs))
    label_counts = {p: np.bincount(data["y"][np.isin(data["ep"], ids)], minlength=6).tolist() for p, ids in split.items()}
    scen: dict[str, int] = {}
    for ep in meta["episodes"]:
        scen[ep["scenario"]] = scen.get(ep["scenario"], 0) + 1
    return {
        "labels": list(ROBOT.label_names),
        "dataset": {
            "episodes": len(meta["episodes"]),
            "frames": meta["n_frames"],
            "label_counts": label_counts,
            "scenario_counts": scen,
            "episodes_with_contact": sum(1 for ep in meta["episodes"] if ep.get("collisions_steps", 0) > 0),
        },
        "records": recs,
    }


# ----------------------------------------------------------------------
# D: VLA 내보내기
# ----------------------------------------------------------------------
def _synthetic_vla_episodes(path: Path, domain: str, labels: list[str], n_eps: int, model: Any | None = None) -> list[VLAEpisode]:
    data, meta = load_dataset(path)
    fps = float(meta["config"]["fps"])
    action = future_action(data["ego_a"], data["ep"], fps)
    task_ko, task_en = task_instruction(domain)
    episodes = []
    for e in meta["episodes"][:n_eps]:
        sel = np.where(data["ep"] == e["index"])[0]
        if model is not None:
            idx = window_index(data["ep"][sel], 8)
            logits, _, _ = predict(model, data["X_v2"][sel], idx)
            probs = softmax(logits)
        else:
            probs = np.eye(len(labels), dtype=np.float32)[data["y"][sel]]
        ctx_names = [labels[i] for i in probs.argmax(1)]
        ent = -(probs * np.log(np.clip(probs, 1e-12, 1))).sum(1) / np.log(len(labels))
        ttc = data["ttc"][sel]
        texts = [narrate(c, float(v), float(tc) if tc > 0 else None, float(p.max())) for c, v, tc, p in zip(ctx_names, data["ego_v"][sel], ttc, probs, strict=True)]
        episodes.append(
            VLAEpisode(
                episode_index=len(episodes),
                task_ko=task_ko,
                task_en=task_en,
                fps=fps,
                source=f"synthetic:{domain}:{e['scenario']}:seed{e['seed']}",
                timestamp=data["t"][sel],
                state=np.stack([data["ego_v"][sel], data["ego_a"][sel]], axis=1),
                features=data["X_v2"][sel],
                context_probs=probs,
                action=action[sel],
                context=ctx_names,
                narration_ko=[t[0] for t in texts],
                narration_en=[t[1] for t in texts],
                event_score=probs[:, [2, 3]].sum(1),
                uncertainty=ent,
                extra={
                    "gt_context": [labels[i] for i in data["y"][sel]],
                    "annotation_source": "context_model" if model is not None else "simulator_ground_truth",
                },
            )
        )
    return episodes


def _comma_vla_episodes(comma_dir: Path, exports: list[dict[str, Any]], budget: float = 0.2) -> list[VLAEpisode]:
    d = dict(np.load(comma_dir / "comma_table.npz"))
    pos = {int(f): i for i, f in enumerate(d["frame_id"])}
    task_ko, task_en = task_instruction("real_driving")
    episodes: list[VLAEpisode] = []
    for ex in exports:
        fids = np.asarray(ex["frame_id"])
        probs = np.asarray(ex["probs"], dtype=np.float32)
        ent = -(probs * np.log(np.clip(probs, 1e-12, 1))).sum(1) / np.log(4)
        score = probs[:, 2] + 0.5 * ent
        group = np.zeros(len(fids), dtype=np.int64)
        group[1:] = np.cumsum(np.diff(fids) != 1)
        mask = select_clips(score, group, clip_len=40, budget_ratio=budget)
        idx = np.where(mask)[0]
        # 연속 구간마다 하나의 에피소드
        breaks = np.where(np.diff(idx) != 1)[0] + 1
        for seg in np.split(idx, breaks):
            if len(seg) < 10:
                continue
            rows = np.array([pos[int(f)] for f in fids[seg]])
            names = [EGO_LABELS[i] for i in probs[seg].argmax(1)]
            texts = [narrate(c, float(v), None, float(p.max())) for c, v, p in zip(names, d["speed"][rows], probs[seg], strict=True)]
            episodes.append(
                VLAEpisode(
                    episode_index=len(episodes),
                    task_ko=task_ko,
                    task_en=task_en,
                    fps=20.0,
                    source="comma_speedchallenge:train.mp4",
                    timestamp=d["t"][rows],
                    state=np.stack([d["speed"][rows], d["accel"][rows]], axis=1),
                    features=d["X_v2"][rows],
                    context_probs=probs[seg],
                    action=d["action"][rows],
                    context=names,
                    narration_ko=[t[0] for t in texts],
                    narration_en=[t[1] for t in texts],
                    event_score=probs[seg][:, 2],
                    uncertainty=ent[seg],
                    video_ref={"path": "data/raw/external/comma_speedchallenge/train.mp4", "frame_ids": fids[seg].tolist()},
                    extra={"gt_context": [EGO_LABELS[i] for i in d["y"][rows]]},
                )
            )
    return episodes


def run_vla_suite(out_dir: Path, data_dir: Path, comma_dir: Path, quick: bool = False, workers: int = 4) -> Path:
    started = time.perf_counter()
    result: dict[str, Any] = {"suite": "vla_physical_ai"}
    comma_res, exports = run_comma_vla(comma_dir, quick, workers)
    result["comma_action_curation"] = comma_res
    logger.info("A/B(comma 행동·큐레이션) 완료")
    result["robot"] = run_robot(data_dir, out_dir, quick, workers)
    logger.info("C(로봇 도메인) 완료")

    export_root = Path("data/processed/vla_export")
    from ..sim.labels import CONTEXT_LABELS

    # 주행 내보내기: 사전학습 맥락 모델의 예측으로 주석(자동 구축 시나리오), 로봇은 GT 라벨 주석
    import torch

    pre = out_dir.parent / "checkpoints" / "driving_semantic_v2_pretrain.pt"
    drv_model = None
    if pre.exists():
        drv_model = build_model(ModelSpec("m", "multichannel_cnn_gru", "v2", grouping="semantic"), 16, 6)
        drv_model.load_state_dict(torch.load(pre, map_location="cpu", weights_only=True))
        drv_model.eval()
    drv_eps = _synthetic_vla_episodes(data_dir / "test_15fps_mid", "driving", CONTEXT_LABELS, 20, model=drv_model)
    export_lerobot_like(drv_eps, export_root / "synthetic_driving", "vcp_synthetic_driving", CONTEXT_LABELS)
    rob_eps = _synthetic_vla_episodes(data_dir / "robot_main_15fps_mid", "robot", list(ROBOT.label_names), 20)
    export_lerobot_like(rob_eps, export_root / "synthetic_robot", "vcp_synthetic_robot", list(ROBOT.label_names))
    com_eps = _comma_vla_episodes(comma_dir, exports)
    if com_eps:
        export_lerobot_like(com_eps, export_root / "comma_curated", "vcp_comma_curated", EGO_LABELS)
    result["export"] = {
        "synthetic_driving": {"episodes": len(drv_eps), "frames": int(sum(len(e.timestamp) for e in drv_eps))},
        "synthetic_robot": {"episodes": len(rob_eps), "frames": int(sum(len(e.timestamp) for e in rob_eps))},
        "comma_curated": {"episodes": len(com_eps), "frames": int(sum(len(e.timestamp) for e in com_eps)), "budget": 0.2},
    }
    # 요약 폴더에는 작은 샘플만 복사(저장소 추적용)
    sample = out_dir / "vla_sample"
    if sample.exists():
        shutil.rmtree(sample)
    for name in ("synthetic_driving", "comma_curated", "synthetic_robot"):
        src = export_root / name
        if not src.exists():
            continue
        (sample / name / "meta").mkdir(parents=True, exist_ok=True)
        for f in (src / "meta").iterdir():
            shutil.copy(f, sample / name / "meta" / f.name)
    if com_eps:
        ep = com_eps[0]
        with (sample / "comma_curated" / "episode_000000_preview.jsonl").open("w", encoding="utf-8") as f:
            for i in range(0, len(ep.timestamp), 4):
                f.write(json.dumps({
                    "timestamp": round(float(ep.timestamp[i]), 3),
                    "video_frame": ep.video_ref["frame_ids"][i] if ep.video_ref else None,
                    "state": [round(float(v), 3) for v in ep.state[i]],
                    "action": [round(float(v), 3) for v in ep.action[i]],
                    "context": ep.context[i],
                    "narration_ko": ep.narration_ko[i],
                    "narration_en": ep.narration_en[i],
                }, ensure_ascii=False) + "\n")
    result["elapsed_s"] = round(time.perf_counter() - started, 1)
    path = out_dir / "vla_results.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    logger.info("저장: %s", path)
    return path


__all__ = ["run_vla_suite", "future_action", "macro_f1"]
