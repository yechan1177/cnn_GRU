from __future__ import annotations

"""VLA 연계 큐레이션 주 실험(논문 v2).

질문: 저장 예산 안에서 어떤 데이터를 남겨야, 그 데이터로 학습한 소형 VLA 정책이 드문 위험 상황에서
실패하지 않는가?

단계(모두 디스크에 캐시되어 중단 후 재실행하면 이어서 돈다)
1. scorer   : 부트스트랩 라벨 셋(기존 합성 데이터)으로 맥락 점수기(CNN-GRU, v1+v2 특징, 창 16) 학습
2. pool     : 지시문(주행 스타일·목표 속도)이 붙은 새 큐레이션 풀 + 개루프 테스트 풀 생성
3. score    : 풀 전체에 점수기 적용 → 이벤트 점수·엔트로피·역 TTC·자차 가속도·GT 위험 라벨
4. pilot    : 검증 시나리오에서 학습 단계 수·시드 분산 측정(변수 측정 단계)
5. tune     : 검증 시나리오에서 ours의 λ(불확실성 가중), ρ(저장소 비율) 선택
6. main     : 방법 × 예산 × 시드 → 정책 학습 → 폐루프(테스트 시나리오) + 개루프 평가
7. lang     : 언어 입력 절제(지시 준수 지표)
8. robot    : 실내 이동로봇(AMR) 도메인 축소 반복
9. comma    : 실주행 영상 개루프(블록 교차검증)

모든 정책 학습은 같은 구조·경사 단계 수·배치·학습률·평가 시드를 쓴다. 조건마다 바뀌는 것은
선별된 프레임 집합뿐이다(통제 변수). 선택 단위는 고정 길이 클립이며, 정책의 과거 프레임(t-2)과
행동 청크 타깃도 클립 안에서만 만든다(저장되지 않은 프레임은 학습에 쓸 수 없다).
"""

import json
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

HAZARD_IDS = (2, 3)  # 주행: brake_warning, hard_brake_risk / 로봇: slow_down, safety_stop
SCORER_DATA = {"driving": "main_15fps_mid", "robot": "robot_main_15fps_mid"}
SEED_BASES = {
    "driving": {"pool": 100000, "val": 150000, "test": 200000, "testpool": 300000},
    "robot": {"pool": 400000, "val": 450000, "test": 500000, "testpool": 600000},
}


@dataclass(slots=True)
class CurationSuiteConfig:
    root: Path = Path("experiments/exp_110_vla_curation")
    synth_dir: Path = Path("data/processed/synth")
    comma_dir: Path = Path("data/processed/comma_speedchallenge")
    domain: str = "driving"
    pool_episodes: int = 1200
    testpool_episodes: int = 150
    val_per_cell: int = 2
    test_per_cell: int = 7
    budgets: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)
    main_budget: float = 0.02  # 모든 방법을 비교하는 주 예산(나머지 예산은 core_methods만)
    core_methods: tuple[str, ...] = ("random", "action_trigger", "uncertainty", "event", "oracle", "ours")
    methods: tuple[str, ...] = (
        "random",
        "uniform",
        "action_trigger",
        "rule_ittc",
        "uncertainty",
        "event",
        "coreset",
        "offline_loss",
        "oracle",
        "ours",
    )
    seeds: tuple[int, ...] = (0, 1, 2)
    clip_len: int = 30
    steps: int = 6000  # 파일럿(검증)에서 전체 데이터가 3000단계에 미수렴(성공률 0.88→6000단계 1.00)해 6000으로 확정
    batch: int = 128
    lam: float = 0.5
    reservoir: float = 0.3
    workers: int = 4
    quick: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def summary(self) -> Path:
        return self.root / "summary"


# ----------------------------------------------------------------------
# 1. 맥락 점수기
# ----------------------------------------------------------------------
def scorer_spec(quick: bool = False) -> Any:
    from .trainer import ModelSpec

    return ModelSpec(
        "scorer_v1v2_w16",
        "multichannel_cnn_gru",
        "v1v2",
        grouping="semantic",
        window=16,
        epochs=3 if quick else 14,
    )


def train_scorer(cfg: CurationSuiteConfig) -> Path:
    """기존 합성 데이터(부트스트랩 라벨 셋)로 점수기를 학습해 저장한다. 풀과 시드가 겹치지 않는다."""

    import torch

    from ..sim.dataset import load_dataset
    from .data import split_groups, window_index
    from .metrics import macro_f1
    from .trainer import fit_temperature, predict, softmax, train_model

    path = cfg.cache / f"scorer_{cfg.domain}.pt"
    if path.exists():
        return path
    cfg.cache.mkdir(parents=True, exist_ok=True)
    data, meta = load_dataset(cfg.synth_dir / SCORER_DATA[cfg.domain])
    spec = scorer_spec(cfg.quick)
    X = data[f"X_{spec.feature_version}"]
    group = data["ep"]
    split = split_groups(group, 0.15, 0.0, seed=2026)
    train_groups = np.concatenate([split["train"], split["test"]])
    idx_tr = window_index(group, spec.window, mask=np.isin(group, train_groups))
    idx_va = window_index(group, spec.window, mask=np.isin(group, split["val"]))
    torch.set_num_threads(4)
    res = train_model(spec, X, data["y"], data["boundary"], group, idx_tr, idx_va, 6, seed=0)
    logits, _, _ = predict(res.model, X, idx_va)
    y_va = data["y"][idx_va[:, -1]]
    temp = fit_temperature(logits, y_va)
    probs = softmax(logits, temp)
    hazard = np.isin(y_va, HAZARD_IDS)
    from .metrics import auroc

    info = {
        "spec": spec.to_dict(),
        "temperature": temp,
        "val_macro_f1": macro_f1(y_va, logits.argmax(1), 6),
        "val_hazard_auroc": auroc(probs[:, list(HAZARD_IDS)].sum(1), hazard),
        "params": res.params,
        "train_seconds": res.train_seconds,
        "train_episodes": int(len(train_groups)),
        "source": SCORER_DATA[cfg.domain],
        "labels": meta["labels"] if "labels" in meta else None,
    }
    torch.save({"state": res.model.state_dict(), "info": info}, path)
    (cfg.cache / f"scorer_{cfg.domain}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("점수기 저장: %s (val macro-F1 %.3f, 위험 AUROC %.3f)", path, info["val_macro_f1"], info["val_hazard_auroc"])
    return path


def load_scorer(path: Path) -> tuple[Any, dict[str, Any]]:
    import torch

    from .trainer import ModelSpec, build_model

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    info = ckpt["info"]
    spec = ModelSpec(**info["spec"])
    from ..features.registry import get_feature_spec

    model = build_model(spec, get_feature_spec(spec.feature_version).dim, 6)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, info


# ----------------------------------------------------------------------
# 2~3. 풀 생성과 점수화
# ----------------------------------------------------------------------
def pool_path(cfg: CurationSuiteConfig, kind: str = "pool") -> Path:
    return cfg.cache / f"{kind}_{cfg.domain}"


def ensure_pools(cfg: CurationSuiteConfig) -> None:
    from ..vla.pool import PoolConfig, generate_pool

    bases = SEED_BASES[cfg.domain]
    for kind, n in (("pool", cfg.pool_episodes), ("testpool", cfg.testpool_episodes)):
        path = pool_path(cfg, kind)
        if (path.parent / f"{path.name}.npz").exists():
            continue
        t0 = time.perf_counter()
        generate_pool(path, PoolConfig(n_episodes=n, seed_base=bases[kind], domain=cfg.domain, workers=cfg.workers))
        logger.info("%s 생성 %.1fs", path, time.perf_counter() - t0)


def clip_ids(group: np.ndarray, clip_len: int) -> np.ndarray:
    """프레임별 클립 id(-1 = 어느 클립에도 속하지 않음). curation.clip_starts와 같은 분할."""

    from ..vla.curation import clip_starts

    out = np.full(len(group), -1, dtype=np.int64)
    for i, s in enumerate(clip_starts(group, clip_len)):
        out[s : s + clip_len] = i
    return out


def entropy_norm(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return (-(p * np.log(p)).sum(1) / math.log(probs.shape[1])).astype(np.float32)


def score_pool(cfg: CurationSuiteConfig) -> Path:
    """점수기를 풀 전체에 인과적으로(과거 16프레임 창) 적용해 선별 신호를 저장한다."""

    import torch

    from ..features.semantic_v2 import V2_KEYS
    from ..vla.pool import load_pool
    from .data import window_index
    from .trainer import predict, softmax

    out = cfg.cache / f"scores_{cfg.domain}.npz"
    if out.exists():
        return out
    pool, _ = load_pool(pool_path(cfg))
    model, info = load_scorer(cfg.cache / f"scorer_{cfg.domain}.pt")
    spec_window = int(info["spec"]["window"])
    X = pool[f"X_{info['spec']['feature_version']}"]
    idx = window_index(pool["ep"], spec_window)
    torch.set_num_threads(4)
    t0 = time.perf_counter()
    logits, _, _ = predict(model, X, idx)
    elapsed = time.perf_counter() - t0
    probs = softmax(logits, info["temperature"])
    v2 = pool["X_v2"]
    ittc = np.maximum(v2[:, V2_KEYS.index("lead_inv_ttc")], v2[:, V2_KEYS.index("vru_approach")])
    np.savez_compressed(
        out,
        probs=probs.astype(np.float32),
        event_score=probs[:, list(HAZARD_IDS)].sum(1).astype(np.float32),
        entropy=entropy_norm(probs),
        ittc=ittc.astype(np.float32),
        action=pool["ego_a"].astype(np.float32),
        oracle=np.isin(pool["y"], HAZARD_IDS),
        batch_ms_per_frame=np.float32(1000.0 * elapsed / max(1, len(idx))),
    )
    logger.info("풀 점수화 완료: %d 프레임, %.3f ms/프레임(배치)", len(idx), 1000.0 * elapsed / max(1, len(idx)))
    return out


# ----------------------------------------------------------------------
# 작업자(프로세스) 전역 상태
# ----------------------------------------------------------------------
_W: dict[str, Any] = {}


def _worker_init(cfg_dict: dict[str, Any]) -> None:
    import torch

    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _W.clear()
    _W["cfg"] = cfg_dict


def _get_pool(cfg: CurationSuiteConfig) -> dict[str, Any]:
    key = f"pool:{cfg.domain}"
    if key not in _W:
        from ..vla.pool import load_pool

        pool, meta = load_pool(pool_path(cfg))
        scores = dict(np.load(cfg.cache / f"scores_{cfg.domain}.npz"))
        cid = clip_ids(pool["ep"], cfg.clip_len)
        loss_path = cfg.cache / f"offline_loss_{cfg.domain}.npy"
        _W[key] = {"pool": pool, "meta": meta, "scores": scores, "clip": cid, "loss_path": loss_path}
    return _W[key]


def _get_testpool(cfg: CurationSuiteConfig) -> dict[str, Any]:
    key = f"testpool:{cfg.domain}"
    if key not in _W:
        from ..vla.pool import load_pool

        pool, meta = load_pool(pool_path(cfg, "testpool"))
        cid = clip_ids(pool["ep"], cfg.clip_len)
        _W[key] = {"pool": pool, "meta": meta, "clip": cid}
    return _W[key]


def episode_tokens(pool: dict[str, np.ndarray], meta: dict[str, Any]) -> np.ndarray:
    """프레임별 지시문 토큰(에피소드 영어 지시문을 인코딩)."""

    from ..vla.instructions import encode_instruction

    table = np.stack([encode_instruction(ep["instruction_en"]) for ep in meta["episodes"]])
    return table[pool["ep"]]


def make_policy_data(pool: dict[str, np.ndarray], meta: dict[str, Any], group: np.ndarray, domain: str) -> Any:
    from ..vla.policy import PolicyData, PoolImageSource

    return PolicyData(
        images=PoolImageSource.from_pool(pool, domain),
        group=group,
        ego_v=pool["ego_v"],
        action=pool["expert_cmd"],
        tokens=episode_tokens(pool, meta),
        domain=domain,
    )


def select_mask(cfg: CurationSuiteConfig, job: dict[str, Any], P: dict[str, Any]) -> np.ndarray:
    from ..vla.curation import select

    n = len(P["clip"])
    if job["method"] == "full":
        return P["clip"] >= 0
    s = P["scores"]
    kwargs: dict[str, Any] = {
        "event_score": s["event_score"],
        "entropy": s["entropy"],
        "features": P["pool"]["X_v1v2"],
        "action": s["action"],
        "ittc": s["ittc"],
        "oracle": s["oracle"],
        "lam": job.get("lam", cfg.lam),
        "reservoir": job.get("reservoir", cfg.reservoir),
    }
    if job["method"] == "offline_loss":
        kwargs["loss"] = np.load(P["loss_path"])
    rng = np.random.default_rng(1000 + int(job["seed"]))
    mask = select(job["method"], job["budget"], P["pool"]["ep"], cfg.clip_len, rng, **kwargs)
    assert len(mask) == n
    return mask


def open_loop_metrics(model: Any, cfg: CurationSuiteConfig, stride: int = 3) -> dict[str, float]:
    """전문가 시연(테스트 풀)에 대한 행동 청크 MAE(전체 / 위험 프레임)."""

    from ..vla.policy import predict_open_loop

    T = _get_testpool(cfg)
    pool = T["pool"]
    data = make_policy_data(pool, T["meta"], T["clip"], cfg.domain)
    valid = np.where(T["clip"] >= 0)[0][::stride]
    pred = predict_open_loop(model, data, valid)
    target = chunk_targets(pool["expert_cmd"], T["clip"], valid, pred.shape[1])
    err = np.abs(pred - target)
    hazard = np.isin(pool["y"][valid], HAZARD_IDS)
    hard_brake = target[:, 0] <= -2.0 if cfg.domain == "driving" else target[:, 0] <= -0.6
    return {
        "mae": float(err.mean()),
        "mae_first": float(err[:, 0].mean()),
        "mae_hazard": float(err[hazard].mean()) if hazard.any() else float("nan"),
        "mae_hard_brake": float(err[hard_brake].mean()) if hard_brake.any() else float("nan"),
        "n_frames": int(len(valid)),
        "n_hazard": int(hazard.sum()),
    }


def chunk_targets(action: np.ndarray, group: np.ndarray, idx: np.ndarray, chunk: int) -> np.ndarray:
    """action[t:t+chunk]을 같은 그룹(연속 구간) 안에서 만든다. 그룹 끝을 넘으면 그룹 마지막 값으로 채운다."""

    n = len(group)
    change = np.ones(n, dtype=bool)
    change[1:] = group[1:] != group[:-1]
    starts = np.where(change)[0]
    ends = np.append(starts[1:], n) - 1
    last = ends[np.searchsorted(starts, idx, side="right") - 1]
    cols = [action[np.minimum(idx + j, last)] for j in range(chunk)]
    return np.stack(cols, 1).astype(np.float32)


def job_key(job: dict[str, Any]) -> str:
    parts = [job["domain"], job["eval"], job["method"], f"b{job['budget']:.2f}", f"s{job['seed']}", f"st{job['steps']}"]
    if not job.get("use_language", True):
        parts.append("nolang")
    if job["method"] == "ours" and ("lam" in job or "reservoir" in job):
        parts.append(f"l{job.get('lam', 'd')}_r{job.get('reservoir', 'd')}")
    return "__".join(str(p) for p in parts)


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    """정책 1개를 학습하고 평가한다(작업자 프로세스에서 실행)."""

    from ..vla.closed_loop import make_test_specs, run_closed_loop
    from ..vla.curation import selection_stats
    from ..vla.policy import PolicyConfig, make_policy_fn, train_policy

    cfg = CurationSuiteConfig(**{**_W["cfg"], "domain": job["domain"]})
    out_path = cfg.runs / f"{job_key(job)}.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    P = _get_pool(cfg)
    mask = select_mask(cfg, job, P) & (P["clip"] >= 0)
    train_idx = np.where(mask)[0]
    stats = selection_stats(mask, P["pool"]["y"], P["pool"]["ep"], 6, HAZARD_IDS)
    data = make_policy_data(P["pool"], P["meta"], P["clip"], cfg.domain)
    pcfg = PolicyConfig(steps=job["steps"], batch=cfg.batch, seed=int(job["seed"]), use_language=job.get("use_language", True), threads=1)
    model, log = train_policy(data, train_idx, pcfg)
    if job.get("save_model"):
        import torch

        torch.save(model, cfg.cache / f"policy_{job_key(job)}.pt")
    bases = SEED_BASES[cfg.domain]
    per_cell = cfg.val_per_cell if job["eval"] == "val" else cfg.test_per_cell
    specs = make_test_specs(cfg.domain, per_cell, bases[job["eval"]])
    closed = run_closed_loop(make_policy_fn(model, cfg.domain), specs, domain=cfg.domain)
    result = {
        "job": job,
        "key": job_key(job),
        "selection": stats,
        "n_train_frames": int(len(train_idx)),
        "train_log": {k: v for k, v in log.items() if k != "loss_curve"} | {"loss_curve": log.get("loss_curve", [])[-20:]},
        "closed_loop": closed,
        "open_loop": open_loop_metrics(model, cfg) if job["eval"] == "test" else {},
        "seconds": time.perf_counter() - t0,
    }
    cfg.runs.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return result


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def run_jobs(cfg: CurationSuiteConfig, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pending = [j for j in jobs if not (cfg.runs / f"{job_key(j)}.json").exists()]
    logger.info("작업 %d개(남은 것 %d개)", len(jobs), len(pending))
    cfg_dict = _cfg_dict(cfg)
    if pending:
        if cfg.workers > 1:
            with ProcessPoolExecutor(max_workers=cfg.workers, initializer=_worker_init, initargs=(cfg_dict,)) as ex:
                for res in ex.map(run_job, pending):
                    _log_result(res)
        else:
            _worker_init(cfg_dict)
            for job in pending:
                _log_result(run_job(job))
    return [json.loads((cfg.runs / f"{job_key(j)}.json").read_text(encoding="utf-8")) for j in jobs]


def _cfg_dict(cfg: CurationSuiteConfig) -> dict[str, Any]:
    d = asdict(cfg)
    return d


def _log_result(res: dict[str, Any]) -> None:
    agg = res["closed_loop"].get("overall", {})
    logger.info(
        "완료 %s: 충돌률 %s, 학습 프레임 %d, %.0fs",
        res["key"],
        agg.get("collision_rate"),
        res["n_train_frames"],
        res["seconds"],
    )


def make_job(cfg: CurationSuiteConfig, method: str, budget: float, seed: int, eval_set: str = "test", **kw: Any) -> dict[str, Any]:
    job = {"domain": cfg.domain, "eval": eval_set, "method": method, "budget": float(budget), "seed": int(seed), "steps": int(kw.pop("steps", cfg.steps))}
    job.update(kw)
    return job


# ----------------------------------------------------------------------
# 4. 오프라인 손실(정책 의존 오프라인 기준선)
# ----------------------------------------------------------------------
def ensure_offline_loss(cfg: CurationSuiteConfig) -> Path:
    """전체 풀로 학습한 정책(시드 0)의 프레임별 개루프 손실. 오프라인 큐레이션 계열의 대리 기준선."""

    import torch

    from ..vla.policy import predict_open_loop

    path = cfg.cache / f"offline_loss_{cfg.domain}.npy"
    if path.exists():
        return path
    job = make_job(cfg, "full", 1.0, 0, "test", save_model=True)
    run_jobs(cfg, [job])
    _worker_init(_cfg_dict(cfg))
    torch.set_num_threads(4)
    P = _get_pool(cfg)
    data = make_policy_data(P["pool"], P["meta"], P["clip"], cfg.domain)
    model = torch.load(cfg.cache / f"policy_{job_key(job)}.pt", map_location="cpu", weights_only=False)
    clip = P["clip"]
    valid = np.where(clip >= 0)[0]
    pred = predict_open_loop(model, data, valid)
    target = chunk_targets(P["pool"]["expert_cmd"], clip, valid, pred.shape[1])
    loss = np.zeros(len(clip), dtype=np.float32)
    loss[valid] = np.abs(pred - target).mean(1)
    np.save(path, loss)
    return path


# ----------------------------------------------------------------------
# 단계 실행
# ----------------------------------------------------------------------
def stage_pilot(cfg: CurationSuiteConfig) -> dict[str, Any]:
    """변수 측정: 학습 단계 수에 따른 성능, 시드 분산(검증 시나리오만 사용)."""

    steps_grid = (1000, 3000, 6000) if not cfg.quick else (50, 100)
    jobs = []
    for st in steps_grid:
        for method, budget in (("full", 1.0), ("random", 0.10)):
            jobs.append(make_job(cfg, method, budget, 0, "val", steps=st))
    pilot_steps = 3000 if not cfg.quick else cfg.steps  # 파일럿 1차는 3000단계에서 시드 분산을 쟀다(기록 그대로 재현)
    for seed in (cfg.seeds if not cfg.quick else (0,)):
        for method, budget in (("full", 1.0), ("random", 0.10), ("oracle", 0.10)):
            jobs.append(make_job(cfg, method, budget, seed, "val", steps=pilot_steps))
    # 파일럿 2차: 예산 규모(1/2/5%)에서 무작위·오라클·CARE·감속 트리거의 격차(주 예산 결정용)
    for b in ((0.01, 0.02, 0.05) if not cfg.quick else (0.05,)):
        for m in ("random", "oracle", "ours", "action_trigger"):
            jobs.append(make_job(cfg, m, b, 0, "val"))  # 2차는 확정한 T(cfg.steps)로 실행
    res = run_jobs(cfg, jobs)
    return {"jobs": [r["key"] for r in res]}


def expert_reference(cfg: CurationSuiteConfig, eval_set: str) -> dict[str, Any]:
    """같은 평가 시나리오를 전문가(지연 없는 IDM 스타일 명령)로 돈 기준 결과(캐시)."""

    from ..vla.closed_loop import make_test_specs, run_closed_loop

    path = cfg.cache / f"expert_{cfg.domain}_{eval_set}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    per_cell = cfg.val_per_cell if eval_set == "val" else cfg.test_per_cell
    specs = make_test_specs(cfg.domain, per_cell, SEED_BASES[cfg.domain][eval_set])
    res = run_closed_loop(None, specs, domain=cfg.domain)
    cfg.cache.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return res


SUCCESS_PROGRESS = 0.8


def episode_success(episodes: list[dict[str, Any]], expert: dict[str, Any], moving_only: bool = False) -> np.ndarray:
    """에피소드 성공 = 충돌 없음 AND 이동 거리 ≥ 0.8 × 같은 시나리오 전문가 이동 거리.

    충돌률만 보면 멈춰 서는 정책이 최고점이 되므로, CARLA 주행 점수처럼 진행률을 함께 요구한다.
    로봇 도메인은 정지 중 피충돌(작업자가 다가와 부딪힘)을 빼기 위해 moving_only=True로 주행 중 충돌만 본다.
    """

    ref = {(e["scenario"], e["seed"]): e["distance"] for e in expert["episodes"]}
    key = "collision_moving" if moving_only else "collision"
    out = []
    for e in episodes:
        d_ref = ref[(e["scenario"], e["seed"])]  # 평가 사양이 다르면 KeyError(조용한 오판정 방지)
        progress_ok = e["distance"] >= SUCCESS_PROGRESS * d_ref
        out.append((not e[key]) and progress_ok)
    return np.asarray(out, dtype=bool)


def stage_tune(cfg: CurationSuiteConfig) -> dict[str, Any]:
    grid_l = (0.0, 0.5, 1.0) if not cfg.quick else (0.5,)
    grid_r = (0.25, 0.5, 0.75) if not cfg.quick else (0.3,)
    jobs = [make_job(cfg, "ours", cfg.main_budget, 0, "val", lam=lam, reservoir=r) for lam in grid_l for r in grid_r]
    res = run_jobs(cfg, jobs)
    expert = expert_reference(cfg, "val")
    moving = cfg.domain == "robot"
    scored = [(float(episode_success(r["closed_loop"]["episodes"], expert, moving).mean()), -float(r["closed_loop"]["overall"].get("speed_error") or 0.0), r) for r in res]
    best = max(scored, key=lambda x: (x[0], x[1]))[2]
    choice = {
        "lam": best["job"]["lam"],
        "reservoir": best["job"]["reservoir"],
        "criterion": "검증 성공률(충돌 없음 + 전문가 대비 진행 80% 이상) 최대, 동률이면 속도 오차 최소",
        "grid": [{"lam": r["job"]["lam"], "reservoir": r["job"]["reservoir"], "success": sc, "speed_error": -se} for sc, se, r in scored],
    }
    (cfg.cache / f"tune_{cfg.domain}.json").write_text(json.dumps(choice, ensure_ascii=False), encoding="utf-8")
    return choice


def tuned(cfg: CurationSuiteConfig) -> CurationSuiteConfig:
    path = cfg.cache / f"tune_{cfg.domain}.json"
    if path.exists():
        t = json.loads(path.read_text(encoding="utf-8"))
        return replace(cfg, lam=float(t["lam"]), reservoir=float(t["reservoir"]))
    return cfg


def stage_main(cfg: CurationSuiteConfig, methods: tuple[str, ...] | None = None, budgets: tuple[float, ...] | None = None) -> list[dict[str, Any]]:
    cfg = tuned(cfg)
    methods = methods or cfg.methods
    if "offline_loss" in methods:
        ensure_offline_loss(cfg)
    jobs = [make_job(cfg, "full", 1.0, s) for s in cfg.seeds]
    for b in budgets or cfg.budgets:
        ms = methods if (b == cfg.main_budget or budgets is not None) else tuple(m for m in methods if m in cfg.core_methods)
        for m in ms:
            for s in cfg.seeds:
                jobs.append(make_job(cfg, m, b, s))
    # 주 예산 조건을 먼저 끝내도록 정렬(중간 점검이 쉽다)
    jobs.sort(key=lambda j: (j["method"] != "full", j["budget"] != cfg.main_budget, j["budget"], j["method"], j["seed"]))
    return run_jobs(cfg, jobs)


def stage_lang(cfg: CurationSuiteConfig) -> list[dict[str, Any]]:
    cfg = tuned(cfg)
    jobs = []
    for s in cfg.seeds:
        for m, b in (("full", 1.0), ("ours", cfg.main_budget), ("random", cfg.main_budget)):
            jobs.append(make_job(cfg, m, b, s, use_language=False))
            jobs.append(make_job(cfg, m, b, s))
    return run_jobs(cfg, jobs)


def write_summary(cfg: CurationSuiteConfig) -> Path:
    """runs/*.json → summary/curation_runs.json(정책 가중치 없이 지표만, 추적 대상)."""

    cfg.summary.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(cfg.runs.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        cl = r["closed_loop"]
        rows.append(
            {
                "key": r["key"],
                "job": r["job"],
                "selection": r["selection"],
                "n_train_frames": r["n_train_frames"],
                "train_log": r["train_log"],
                "closed_loop": {k: v for k, v in cl.items() if k != "episodes"},
                "closed_loop_episodes": cl.get("episodes", []),
                "open_loop": r.get("open_loop", {}),
                "seconds": r["seconds"],
            }
        )
    path = cfg.summary / "curation_runs.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, default=_json_default), encoding="utf-8")
    meta = {}
    for name in ("scorer_driving.json", "scorer_robot.json", "tune_driving.json", "tune_robot.json"):
        p = cfg.cache / name
        if p.exists():
            meta[name.removesuffix(".json")] = json.loads(p.read_text(encoding="utf-8"))
    (cfg.summary / "curation_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("요약 저장: %s (%d 실행)", path, len(rows))
    return path


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="VLA 연계 큐레이션 실험")
    ap.add_argument("stage", choices=["prepare", "pilot", "tune", "main", "lang", "robot", "comma", "summary", "all"])
    ap.add_argument("--root", default="experiments/exp_110_vla_curation")
    ap.add_argument("--domain", default="driving")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = CurationSuiteConfig(root=Path(args.root), domain=args.domain, workers=args.workers, quick=args.quick)
    if args.quick:
        cfg = replace(cfg, pool_episodes=24, testpool_episodes=6, val_per_cell=1, test_per_cell=1, seeds=(0,), steps=60, batch=32)
    if args.steps:
        cfg = replace(cfg, steps=args.steps)
    if args.domain == "robot" and not args.quick:
        cfg = replace(cfg, pool_episodes=600, testpool_episodes=90, test_per_cell=4)

    if args.stage in {"prepare", "all"}:
        train_scorer(cfg)
        ensure_pools(cfg)
        score_pool(cfg)
    if args.stage in {"pilot", "all"}:
        stage_pilot(cfg)
    if args.stage in {"tune", "all"}:
        stage_tune(cfg)
    if args.stage in {"main", "all"}:
        stage_main(cfg)
    if args.stage in {"lang", "all"}:
        stage_lang(cfg)
    if args.stage == "robot":
        stage_main(cfg, methods=("random", "uniform", "action_trigger", "uncertainty", "event", "oracle", "ours"), budgets=(cfg.main_budget,))
    if args.stage == "comma":
        from .comma_curation import run_comma_curation

        run_comma_curation(cfg)
    if args.stage in {"summary", "all"}:
        write_summary(cfg)


if __name__ == "__main__":
    main()
