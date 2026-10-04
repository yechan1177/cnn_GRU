from __future__ import annotations

"""실주행 영상(comma.ai speedchallenge) 개루프 큐레이션 실험.

설계(10fps 테이블, 5-fold 시간 블록 교차검증)
- fold k: 블록 k = 테스트, 블록 k+1 = 점수기 학습용 "부트스트랩 라벨" 블록, 나머지 3블록 = 큐레이션 풀.
- 점수기
  - real: 부트스트랩 블록의 자차 운동 4상태 라벨로 학습한다.
  - sim: 합성 주행 데이터로만 학습한 6맥락 점수기를 그대로 쓴다(실라벨 0개).
- 정책: 64×64 실영상 2프레임 + 속도 → 미래 1초(10스텝) 가속도 청크. 언어는 고정 지시문이다.
- 평가(테스트 블록 전체)
  - 청크 MAE: 전체 / 제동 프레임
  - 제동 시작 예측 AUROC: 현재 제동이 아닌 프레임에서 1초 안에 제동이 시작되는지를 예측 청크 평균의 음수로 판별
- 블록 경계 ±purge 프레임은 어느 분할의 대상에서도 뺀다.
"""

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .vla_curation_suite import CurationSuiteConfig, _json_default, chunk_targets, clip_ids, entropy_norm

logger = logging.getLogger(__name__)

COMMA_METHODS = ("random", "uniform", "action_trigger", "uncertainty", "event", "event_sim", "ours", "oracle")
BRAKING = 2
N_BLOCKS = 5
PURGE = 10
CHUNK = 10
CLIP_LEN = 20
INSTRUCTION = "drive safely along the road"

_C: dict[str, Any] = {}


def _load(cfg: CurationSuiteConfig) -> dict[str, Any]:
    if "d" in _C:
        return _C
    d = dict(np.load(cfg.comma_dir / "comma_table_10fps.npz"))
    d["X_v1v2"] = np.concatenate([d["X_v1"], d["X_v2"]], axis=1)
    frames = np.load(cfg.comma_dir / "frames_64.npz")["frames"]
    d["frames"] = frames[d["frame_id"]]
    _C["d"] = d
    return _C


def _scorer_real(d: dict[str, np.ndarray], seed_mask: np.ndarray, quick: bool) -> tuple[np.ndarray, dict[str, Any]]:
    """부트스트랩 블록으로 4상태 점수기를 학습하고 전체 프레임 확률을 낸다."""

    from .data import window_index
    from .metrics import macro_f1
    from .trainer import fit_temperature, predict, softmax, train_model
    from .vla_curation_suite import scorer_spec

    spec = replace(scorer_spec(quick), patience=4)
    X = d["X_v1v2"]
    pos = np.where(seed_mask)[0]
    cut = pos[int(len(pos) * 0.8)]
    tr = seed_mask & (np.arange(len(X)) < cut - spec.window)
    va = seed_mask & (np.arange(len(X)) >= cut)
    group = np.zeros(len(X), dtype=np.int64)
    idx_tr = window_index(group, spec.window, mask=tr)
    idx_va = window_index(group, spec.window, mask=va)
    res = train_model(spec, X, d["y"], d["boundary"], group, idx_tr, idx_va, 4, seed=0)
    logits_va, _, _ = predict(res.model, X, idx_va)
    temp = fit_temperature(logits_va, d["y"][idx_va[:, -1]])
    logits, _, _ = predict(res.model, X, window_index(group, spec.window))
    probs = softmax(logits, temp)
    return probs, {"val_macro_f1": macro_f1(d["y"][idx_va[:, -1]], logits_va.argmax(1), 4), "temperature": temp}


def _scorer_sim(cfg: CurationSuiteConfig, d: dict[str, np.ndarray]) -> np.ndarray:
    from .data import window_index
    from .trainer import predict, softmax
    from .vla_curation_suite import HAZARD_IDS, load_scorer

    model, info = load_scorer(cfg.cache / "scorer_driving.pt")
    idx = window_index(np.zeros(len(d["y"]), dtype=np.int64), int(info["spec"]["window"]))
    logits, _, _ = predict(model, d["X_v1v2"], idx)
    return softmax(logits, info["temperature"])[:, list(HAZARD_IDS)].sum(1)


def prepare_fold(cfg: CurationSuiteConfig, fold_id: int) -> Path:
    """fold별 점수(실/시뮬 점수기)와 분할 마스크를 캐시한다."""

    from ..features.semantic_v2 import V2_KEYS
    from .data import blocked_folds

    out = cfg.cache / f"comma_fold{fold_id}.npz"
    if out.exists():
        return out
    import torch

    torch.set_num_threads(4)
    d = _load(cfg)["d"]
    n = len(d["y"])
    block, folds = blocked_folds(n, N_BLOCKS, PURGE)
    f = folds[fold_id]
    keep = f["purge_mask"]
    seed_block = int(f["val"][0])
    test_mask = (block == fold_id) & keep
    seed_mask = (block == seed_block) & keep
    pool_mask = np.isin(block, f["train"]) & keep
    probs, info = _scorer_real(d, seed_mask, cfg.quick)
    v2 = d["X_v2"]
    np.savez_compressed(
        out,
        block=block,
        test_mask=test_mask,
        seed_mask=seed_mask,
        pool_mask=pool_mask,
        event_score=probs[:, BRAKING].astype(np.float32),
        probs=probs.astype(np.float32),  # v4 위험 맥락 보조 헤드 타깃(docs/36 P3)
        entropy=entropy_norm(probs),
        event_sim=_scorer_sim(cfg, d).astype(np.float32),
        ittc=np.maximum(v2[:, V2_KEYS.index("lead_inv_ttc")], v2[:, V2_KEYS.index("vru_approach")]).astype(np.float32),
        info=json.dumps(info),
    )
    return out


def _segments(mask: np.ndarray) -> np.ndarray:
    """마스크의 연속 구간마다 다른 그룹 id(-1 = 마스크 밖)."""

    grp = np.full(len(mask), -1, dtype=np.int64)
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    diff = np.diff(padded)
    for i, (s, e) in enumerate(zip(np.where(diff == 1)[0], np.where(diff == -1)[0], strict=True)):
        grp[s:e] = i
    return grp


def run_comma_job(job: dict[str, Any]) -> dict[str, Any]:
    import torch

    from ..vla.curation import select, selection_stats
    from ..vla.instructions import encode_instruction
    from ..vla.policy import ArrayImageSource, PolicyConfig, PolicyData, predict_open_loop, train_policy
    from .metrics import auroc

    torch.set_num_threads(1)
    cfg: CurationSuiteConfig = job["cfg"]
    out_path = cfg.runs / f"{job['key']}.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    d = _load(cfg)["d"]
    F = dict(np.load(cfg.cache / f"comma_fold{job['fold']}.npz"))
    n = len(d["y"])
    pool_seg = _segments(F["pool_mask"])
    pool_clip = clip_ids(np.where(pool_seg >= 0, pool_seg, -1 - np.arange(n)), CLIP_LEN)
    pool_clip[pool_seg < 0] = -1
    oracle = d["y"] == BRAKING
    method = job["method"]
    if method == "full":
        mask = pool_clip >= 0
    elif method in ("random_shared", "care", "mix_trigger"):
        # v3 공유 저장소 선별(docs/33 M1). 풀 밖 프레임은 프레임마다 다른 그룹이라 클립이 생기지 않는다.
        from ..vla.curation import select_shared

        rng = np.random.default_rng(1000 + job["seed"])
        group = np.where(pool_seg >= 0, pool_seg, -1 - np.arange(n))
        score = {"random_shared": None, "care": F["event_score"], "mix_trigger": (-d["accel"]).astype(np.float32)}[method]
        mask = select_shared(
            score,
            job["budget"] * F["pool_mask"].sum() / n,
            group,
            CLIP_LEN,
            rng,
            reservoir=float(job.get("reservoir", cfg.reservoir)),
            entropy=F["entropy"] if method == "care" else None,
            lam=float(job.get("lam", cfg.lam)) if method == "care" else 0.0,
            **({"per_group_cap": job["per_group_cap"]} if job.get("per_group_cap") is not None else {}),  # v4 S1
        )
        mask &= pool_clip >= 0
    else:
        rng = np.random.default_rng(1000 + job["seed"])
        group = np.where(pool_seg >= 0, pool_seg, -1 - np.arange(n))
        kw: dict[str, Any] = {
            "event_score": F["event_sim"] if method == "event_sim" else F["event_score"],
            "entropy": F["entropy"],
            "features": d["X_v1v2"],
            "action": d["accel"],
            "ittc": F["ittc"],
            "oracle": oracle,
            "lam": job.get("lam", cfg.lam),
            "reservoir": job.get("reservoir", cfg.reservoir),
        }
        # 풀 밖 프레임은 점수를 최저로 두어 선택되지 않게 한다(그룹 id도 프레임마다 달라 클립이 생기지 않는다).
        mask = select("event" if method == "event_sim" else method, job["budget"] * F["pool_mask"].sum() / n, group, CLIP_LEN, rng, **kw)
        mask &= pool_clip >= 0
    train_idx = np.where(mask)[0]
    tokens = np.tile(encode_instruction(INSTRUCTION), (n, 1))
    use_feat = bool(job.get("use_features", False))  # v3: 검출 특징 토큰(YOLO 검출 기반 v1+v2) 입력
    # v4(docs/36): 정책 블록 설정(시간 특징 인코더·보조 헤드·지시문 드롭아웃). 비어 있으면 v3와 같다.
    pparams = dict(job.get("policy_params") or {})
    extra_data: dict[str, Any] = {}
    if pparams.get("aux_weight", 0.0) > 0:
        if "probs" not in F:
            raise ValueError("보조 헤드에는 fold 캐시의 점수기 확률(probs)이 필요합니다. prepare_fold를 다시 실행하세요.")
        extra_data["aux_targets"] = F["probs"].astype(np.float32)
        pparams["aux_classes"] = int(F["probs"].shape[1])
    data = PolicyData(
        images=ArrayImageSource(d["frames"]),
        group=pool_clip,
        ego_v=d["speed"],
        action=d["accel"],
        tokens=tokens,
        domain="driving",
        **({"features": d["X_v1v2"].astype(np.float32)} if use_feat else {}),
        **extra_data,
    )
    pcfg = PolicyConfig(chunk=CHUNK, steps=job["steps"], batch=cfg.batch, seed=job["seed"], threads=1, **({"use_features": True} if use_feat else {}), **pparams)
    model, log = train_policy(data, train_idx, pcfg)
    # 평가: 테스트 블록(연속 구간)을 그룹으로
    test_seg = _segments(F["test_mask"])
    test_idx = np.where(test_seg >= 0)[0]
    eval_data = replace(data, group=np.where(test_seg >= 0, test_seg, -1 - np.arange(n)))
    pred = predict_open_loop(model, eval_data, test_idx)
    target = chunk_targets(d["accel"], eval_data.group, test_idx, CHUNK)
    err = np.abs(pred - target)
    brake_now = oracle[test_idx]
    future_brake = np.zeros(len(test_idx), dtype=bool)
    for j in range(1, CHUNK + 1):
        nxt = np.minimum(test_idx + j, n - 1)
        future_brake |= oracle[nxt] & (eval_data.group[nxt] == eval_data.group[test_idx])
    onset_mask = ~brake_now
    result = {
        "key": job["key"],
        "job": {k: v for k, v in job.items() if k != "cfg"},
        "policy_params": pparams,
        "selection": selection_stats(mask, d["y"], np.where(pool_seg >= 0, pool_seg, -1), 4, (BRAKING,)),
        "n_train_frames": int(len(train_idx)),
        "train_log": {k: v for k, v in log.items() if k != "loss_curve"},
        "open_loop": {
            "mae": float(err.mean()),
            "mae_first": float(err[:, 0].mean()),
            "mae_braking": float(err[brake_now].mean()) if brake_now.any() else float("nan"),
            "brake_onset_auroc": auroc(-pred[onset_mask].mean(1), future_brake[onset_mask]),
            "n_test": int(len(test_idx)),
        },
        "seconds": time.perf_counter() - t0,
    }
    cfg.runs.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, default=_json_default), encoding="utf-8")
    logger.info("완료 %s: MAE %.3f, 제동 MAE %.3f, 시작 AUROC %.3f", job["key"], result["open_loop"]["mae"], result["open_loop"]["mae_braking"], result["open_loop"]["brake_onset_auroc"])
    return result


def run_comma_curation(
    cfg: CurationSuiteConfig,
    methods: tuple[str, ...] = COMMA_METHODS,
    budgets: tuple[float, ...] = (0.10, 0.20),
    seeds: tuple[int, ...] = (0, 1),
    steps: int = 1500,
) -> list[dict[str, Any]]:
    from .vla_curation_suite import tuned

    cfg = tuned(replace(cfg, domain="driving"))
    folds = range(N_BLOCKS if not cfg.quick else 1)
    for k in folds:
        prepare_fold(cfg, k)
    jobs = []
    for k in folds:
        for s in seeds:
            jobs.append({"cfg": cfg, "fold": k, "method": "full", "budget": 1.0, "seed": s, "steps": steps})
            for b in budgets:
                for m in methods:
                    jobs.append({"cfg": cfg, "fold": k, "method": m, "budget": b, "seed": s, "steps": steps})
    for j in jobs:
        j["key"] = f"comma__f{j['fold']}__{j['method']}__b{j['budget']:.2f}__s{j['seed']}__st{j['steps']}"
    pending = [j for j in jobs if not (cfg.runs / f"{j['key']}.json").exists()]
    logger.info("comma 작업 %d개(남은 것 %d개)", len(jobs), len(pending))
    if pending:
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            list(ex.map(run_comma_job, pending))
    return [json.loads((cfg.runs / f"{j['key']}.json").read_text(encoding="utf-8")) for j in jobs]
