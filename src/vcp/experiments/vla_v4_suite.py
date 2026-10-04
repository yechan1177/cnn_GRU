from __future__ import annotations

"""v4 실험 스위트: 미달 KPI 원인별 개선의 개발 세트 선택과 새 테스트 세트 확증(docs/36, 사전 등록 docs/37).

v3(`vla_v3_suite.py`, exp_120)에서 바뀐 점
- 정책 블록: 시간 특징 인코더(P2), 위험 맥락 보조 헤드(P3), 지시문 드롭아웃(T1). 모두 설정값으로 켜고 끈다.
- 선별: 공유 저장소 점수 몫의 에피소드 상한(S1, `per_group_cap`).
- 평가 세트: 새 개발 세트(160000번대·반사실 170000번대)와 처음 보는 새 테스트 세트(1000000번대).
- 풀·점수기·풀 점수는 v3와 같다(v3 캐시를 복사해 쓴다). 따라서 v3·v4 차이는 정책·선별 설정과 평가 시드뿐이다.

작업 키에는 설정 이름(variant)을 넣는다. 같은 작업 키는 같은 결과를 뜻한다(캐시).
"""

import json
import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .vla_curation_suite import HAZARD_IDS, _json_default, chunk_targets, clip_ids, episode_tokens

logger = logging.getLogger(__name__)

SEED_BASES_V4 = {
    "driving": {"dev": 160000, "devcf": 170000, "test": 1000000, "cf": 1050000},
    "robot": {"dev": 460000, "devcf": 470000, "test": 1100000, "cf": 1150000},  # AMR 개발 세트는 셀당 4개
}

# 설정 묶음(variant). v3 = v3 정책·선별 그대로, v4 = 개발 세트에서 채택한 조합(stage_dev 이후 cache/v4_choice.json로 확정)
VARIANTS: dict[str, dict[str, Any]] = {
    "v3": {},
    "p2": {"feature_history": 8, "feature_stride": 2, "feature_encoder": "gru"},
    "p3": {"aux_weight": 0.2},
    "t1": {"lang_dropout": 0.15},
    "p4": {"hazard_weight": 2.0},
    "p5": {"proprio_history": 8, "proprio_stride": 2},  # 자차 운동 이력(v3 실영상 진단 후 추가, docs/36 2.0b)
    # S1(점수 몫 에피소드 상한)은 개발 실험 전에 기각했다: v3 CARE 2% 점수 몫 36클립이 이미 서로 다른 36개 에피소드에서
    # 나와 c=1이 선택을 바꾸지 않는다(A15 측정, docs/30 7절). 트리거 혼합만 바뀌어 비교 기준선만 달라진다.
    "all": {"feature_history": 8, "feature_stride": 2, "feature_encoder": "gru", "aux_weight": 0.2, "lang_dropout": 0.15, "hazard_weight": 2.0},
}
POLICY_KEYS = ("feature_history", "feature_stride", "feature_encoder", "aux_weight", "lang_dropout", "proprio_history", "proprio_stride")


@dataclass(slots=True)
class V4Config:
    root: Path = Path("experiments/exp_130_vla_v4")
    v3_root: Path = Path("experiments/exp_120_vla_v3")
    comma_dir: Path = Path("data/processed/comma_speedchallenge")
    domain: str = "driving"
    dev_per_cell: int = 4
    devcf_per_cell: int = 2
    test_per_cell: int = 7
    cf_per_cell: int = 3
    clip_len: int = 30
    steps: int = 6000
    batch: int = 128
    workers: int = 4
    quick: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    def for_domain(self, domain: str) -> "V4Config":
        if domain == "robot" and not self.quick:
            return replace(self, domain="robot", test_per_cell=4, cf_per_cell=2)
        return replace(self, domain=domain)

    def v3(self) -> Any:
        """v3 설정(선별 λ·ρ는 v3 튜닝값). 풀 경로·전문가 계산 등 v3 유틸리티에 넘긴다."""

        from .vla_v3_suite import V3Config, tuned

        c = V3Config(root=self.v3_root, domain=self.domain, workers=self.workers, quick=self.quick)
        c = c.for_domain(self.domain) if self.domain != "driving" else c
        return tuned(c)


# ----------------------------------------------------------------------
# 준비: v3 캐시(풀·점수기·점수·튜닝값) 복사
# ----------------------------------------------------------------------
def prepare(cfg: V4Config) -> None:
    cfg.cache.mkdir(parents=True, exist_ok=True)
    src = cfg.v3_root / "cache"
    names = [f"pool_{cfg.domain}.npz", f"pool_{cfg.domain}.meta.json", f"testpool_{cfg.domain}.npz", f"testpool_{cfg.domain}.meta.json", f"scores_{cfg.domain}.npz", f"tune_{cfg.domain}.json", f"scorer_{cfg.domain}.pt", f"scorer_{cfg.domain}.json"]
    for n in names:
        if (src / n).exists() and not (cfg.cache / n).exists():
            shutil.copy2(src / n, cfg.cache / n)
    missing = [n for n in names if not (cfg.cache / n).exists() and not n.startswith(("tune_", "scorer_"))]
    if missing:
        raise FileNotFoundError(f"v3 캐시가 없습니다(먼저 v3 prepare): {missing}")


def eval_specs(cfg: V4Config, eval_set: str) -> list[Any]:
    from ..vla.closed_loop import make_test_specs

    base = SEED_BASES_V4[cfg.domain][eval_set]
    per = {"dev": cfg.dev_per_cell, "devcf": cfg.devcf_per_cell, "test": cfg.test_per_cell, "cf": cfg.cf_per_cell}[eval_set]
    return make_test_specs(cfg.domain, per, base, counterfactual=eval_set in ("devcf", "cf"))


def expert_reference(cfg: V4Config, eval_set: str) -> dict[str, Any]:
    from ..vla.closed_loop import run_closed_loop

    path = cfg.cache / f"expert_{cfg.domain}_{eval_set}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    res = run_closed_loop(None, eval_specs(cfg, eval_set), domain=cfg.domain, collision_pushback=False, decouple_initial_speed=True)
    cfg.cache.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return res


# ----------------------------------------------------------------------
# 작업자
# ----------------------------------------------------------------------
_W: dict[str, Any] = {}


def _worker_init(cfg_dict: dict[str, Any]) -> None:
    import torch

    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _W.clear()
    _W["cfg"] = cfg_dict


def _cfg_from(d: dict[str, Any], domain: str) -> V4Config:
    d = dict(d)
    for k in ("root", "v3_root", "comma_dir"):
        d[k] = Path(d[k])
    c = V4Config(**d)
    return c.for_domain(domain) if domain != c.domain else c


def _pool(cfg: V4Config, kind: str = "pool") -> dict[str, Any]:
    key = f"{kind}:{cfg.domain}"
    if key not in _W:
        from ..vla.pool import load_pool

        pool, meta = load_pool(cfg.cache / f"{kind}_{cfg.domain}")
        entry: dict[str, Any] = {"pool": pool, "meta": meta, "clip": clip_ids(pool["ep"], cfg.clip_len)}
        if kind == "pool":
            entry["scores"] = dict(np.load(cfg.cache / f"scores_{cfg.domain}.npz"))
        _W[key] = entry
    return _W[key]


def variant_params(name: str, cfg: V4Config) -> dict[str, Any]:
    if name == "v4":
        p = cfg.cache / f"v4_choice_{cfg.domain}.json"  # 도메인별 개발 세트에서 정한 조합(실영상은 주행 조합을 쓴다)
        if not p.exists():
            raise FileNotFoundError(f"v4 채택 조합({p})이 없습니다. stage_dev를 먼저 실행하세요.")
        return dict(json.loads(p.read_text(encoding="utf-8"))["params"])
    return dict(VARIANTS[name])


def policy_data(cfg: V4Config, P: dict[str, Any], params: dict[str, Any]) -> Any:
    from ..vla.policy import PolicyData, PoolImageSource

    pool = P["pool"]
    aux = None
    if params.get("aux_weight", 0.0) > 0:
        if "scores" not in P:
            raise ValueError("보조 헤드 타깃(점수기 확률)은 학습 풀에서만 쓸 수 있습니다.")
        aux = P["scores"]["probs"].astype(np.float32)
    lw = None
    if params.get("hazard_weight", 0.0) > 0 and "scores" in P:
        # P4 위험 가중 손실: 수집 시점 점수기 위험 확률 e_t로 가중치 1 + β·e_t(특권 정보 아님)
        lw = (1.0 + float(params["hazard_weight"]) * P["scores"]["event_score"]).astype(np.float32)
    return PolicyData(
        images=PoolImageSource.from_pool(pool, cfg.domain),
        group=P["clip"],
        ego_v=pool["ego_v"],
        action=pool["expert_cmd"],
        tokens=episode_tokens(pool, P["meta"]),
        domain=cfg.domain,
        features=pool["X_v1v2"].astype(np.float32),
        # 특징 프리롤: 특징 이력은 에피소드 시작에서 자른다(클립 앞 (H−1)·s 프레임의 특징을 함께 저장한다고 가정, docs/36 6절)
        feature_group=pool["ep"],
        aux_targets=aux,
        loss_weight=lw,
    )


def select_v4(cfg: V4Config, job: dict[str, Any], P: dict[str, Any], params: dict[str, Any]) -> np.ndarray:
    from ..vla.curation import select_shared

    m, b = job["method"], float(job["budget"])
    if m == "full":
        return P["clip"] >= 0
    v3c = cfg.v3()
    s = P["scores"]
    rng = np.random.default_rng(1000 + int(job["seed"]))
    score = {"random_shared": None, "care": s["event_score"], "mix_trigger": (-s["action"]).astype(np.float32), "mix_oracle": s["oracle"].astype(np.float32)}[m]
    ent = s["entropy"] if m == "care" else None
    lam = float(v3c.lam) if m == "care" else 0.0
    return select_shared(score, b, P["pool"]["ep"], cfg.clip_len, rng, reservoir=float(v3c.reservoir), entropy=ent, lam=lam, per_group_cap=params.get("per_group_cap"))


def job_key(job: dict[str, Any]) -> str:
    parts = [job["domain"], job["eval"], job["variant"], job["method"], f"b{job['budget']:.2f}", f"s{job['seed']}", f"st{job['steps']}"]
    if not job.get("use_language", True):
        parts.append("nolang")
    return "__".join(str(p) for p in parts)


def make_job(cfg: V4Config, variant: str, method: str, budget: float, seed: int, eval_set: str, **kw: Any) -> dict[str, Any]:
    job = {"domain": cfg.domain, "eval": eval_set, "variant": variant, "method": method, "budget": float(budget), "seed": int(seed), "steps": int(kw.pop("steps", cfg.steps))}
    job.update(kw)
    return job


def open_loop(model: Any, cfg: V4Config, params: dict[str, Any], stride: int = 3) -> dict[str, float]:
    from ..vla.policy import predict_open_loop

    T = _pool(cfg, "testpool")
    pool = T["pool"]
    data = policy_data(cfg, T, {k: v for k, v in params.items() if k != "aux_weight"})
    valid = np.where(T["clip"] >= 0)[0][::stride]
    pred = predict_open_loop(model, data, valid)
    target = chunk_targets(pool["expert_cmd"], T["clip"], valid, pred.shape[1])
    err = np.abs(pred - target)
    hazard = np.isin(pool["y"][valid], HAZARD_IDS)
    return {"mae": float(err.mean()), "mae_hazard": float(err[hazard].mean()) if hazard.any() else float("nan")}


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    from ..vla.closed_loop import run_closed_loop
    from ..vla.curation import selection_stats
    from ..vla.policy import PolicyConfig, make_policy_fn, train_policy

    cfg = _cfg_from(_W["cfg"], job["domain"])
    out_path = cfg.runs / f"{job_key(job)}.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    params = variant_params(job["variant"], cfg)
    P = _pool(cfg)
    mask = select_v4(cfg, job, P, params) & (P["clip"] >= 0)
    train_idx = np.where(mask)[0]
    stats = selection_stats(mask, P["pool"]["y"], P["pool"]["ep"], 6, HAZARD_IDS)
    data = policy_data(cfg, P, params)
    pkw = {k: params[k] for k in POLICY_KEYS if k in params}
    pcfg = PolicyConfig(steps=job["steps"], batch=cfg.batch, seed=int(job["seed"]), use_language=job.get("use_language", True), use_features=True, threads=1, **pkw)
    model, log = train_policy(data, train_idx, pcfg)
    fn = make_policy_fn(model, cfg.domain)
    closed = run_closed_loop(fn, eval_specs(cfg, job["eval"]), domain=cfg.domain, collision_pushback=False, decouple_initial_speed=True, feature_history=pcfg.feature_history, feature_stride=pcfg.feature_stride)
    result = {
        "job": job,
        "key": job_key(job),
        "params": params,
        "selection": stats,
        "n_train_frames": int(len(train_idx)),
        "train_log": {k: v for k, v in log.items() if k not in ("loss_curve", "config")},
        "closed_loop": closed,
        "open_loop": open_loop(model, cfg, params) if job["eval"] == "test" else {},
        "seconds": time.perf_counter() - t0,
    }
    cfg.runs.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return result


def run_jobs(cfg: V4Config, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pending = [j for j in jobs if not (cfg.runs / f"{job_key(j)}.json").exists()]
    logger.info("작업 %d개(남은 것 %d개)", len(jobs), len(pending))
    d = {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()}
    if pending:
        if cfg.workers > 1:
            with ProcessPoolExecutor(max_workers=cfg.workers, initializer=_worker_init, initargs=(d,)) as ex:
                for r in ex.map(run_job, pending):
                    logger.info("완료 %s: 충돌률 %.3f, %.0fs", r["key"], r["closed_loop"]["overall"]["collision_rate"], r["seconds"])
        else:
            _worker_init(d)
            for j in pending:
                run_job(j)
    return [json.loads((cfg.runs / f"{job_key(j)}.json").read_text(encoding="utf-8")) for j in jobs]


def success_of(cfg: V4Config, r: dict[str, Any]) -> float:
    from .vla_v3_report import episode_success

    ex = expert_reference(cfg, r["job"]["eval"])
    return float(episode_success(r["closed_loop"]["episodes"], ex, cfg.domain == "robot").mean())


# ----------------------------------------------------------------------
# 단계
# ----------------------------------------------------------------------
DEV_SEEDS = (0, 1, 2)
MAIN_SEEDS = tuple(range(10))


def stage_dev(cfg: V4Config) -> dict[str, Any]:
    """개발 세트에서 후보 설정을 비교해 도메인별 v4 조합을 정한다(docs/36 3절 채택 규칙).

    - 후보 {v3, +P2, +P3, +P4, 전부}(전부 = P2+P3+T1+P4) 중 개발 세트 CARE 2% 성공률 최대를 고른다.
      측정하지 않은 조합은 채택하지 않는다.
    - T1(주행): 반사실 개발 세트에서 CARE 2% 언어 있음 속도 오차가 v3 이하이면 더한다. AMR은 주행 결정을 따른다.
    """

    robot = cfg.domain == "robot"
    prepare(cfg)
    expert_reference(cfg, "dev")
    if not robot:
        expert_reference(cfg, "devcf")
    variants = ("v3", "p2", "p3", "p4", "p5", "all") if robot else ("v3", "p2", "p3", "t1", "p4", "p5", "all")
    jobs = [make_job(cfg, v, "care", 0.02, s, "dev") for v in variants for s in DEV_SEEDS]
    jobs += [make_job(cfg, v, "full", 1.0, 0, "dev") for v in ("v3", "all")]
    if not robot:
        jobs += [make_job(cfg, v, "care", 0.02, s, "devcf", use_language=lang) for v in ("v3", "t1", "all") for s in DEV_SEEDS[:2] for lang in (True, False)]
    res = run_jobs(cfg, jobs)
    succ: dict[str, list[float]] = {}
    full: dict[str, float] = {}
    se: dict[tuple[str, bool], list[float]] = {}
    for r in res:
        j = r["job"]
        if j["eval"] == "dev" and j["method"] == "care":
            succ.setdefault(j["variant"], []).append(success_of(cfg, r))
        if j["eval"] == "dev" and j["method"] == "full":
            full[j["variant"]] = success_of(cfg, r)
        if j["eval"] == "devcf":
            v = r["closed_loop"]["overall"].get("speed_error")
            se.setdefault((j["variant"], j.get("use_language", True)), []).append(np.nan if v is None else float(v))
    mean = {v: float(np.mean(x)) for v, x in succ.items()}
    # 측정한 후보(v3, 단일 개선, 전부) 중 개발 세트 성공률 최대를 고른다(동률이면 구성 요소가 적은 쪽, 그다음 v3 우선).
    # T1은 주행 반사실 개발 세트로 따로 정해 단일 개선 후보에 더한다("전부" 후보는 T1을 포함해 측정했다).
    cands = [v for v in ("v3", "p2", "p3", "p4", "p5", "all") if v in mean]
    order = {"v3": 0, "p2": 1, "p3": 1, "p4": 1, "p5": 1, "all": 3}
    best = max(cands, key=lambda v: (round(mean[v], 6), -order[v]))
    if robot:
        dc = cfg.cache / "v4_choice_driving.json"
        t1 = bool(json.loads(dc.read_text(encoding="utf-8"))["adopt"]["t1"]) if dc.exists() else False
    else:
        t1 = float(np.nanmean(se[("t1", True)])) <= float(np.nanmean(se[("v3", True)]))
    params = dict(VARIANTS[best])  # "전부"는 T1을 포함한 상태로 측정했으므로 그대로 쓴다
    if best != "all" and t1:
        params.update(VARIANTS["t1"])
    adopt = {"best": best, "t1": t1}
    choice = {
        "domain": cfg.domain,
        "params": params,
        "adopt": adopt,
        "dev_success": mean,
        "dev_success_by_seed": succ,
        "dev_full_success": full,
        "devcf_speed_error": {f"{v}:{'lang' if l else 'nolang'}": float(np.nanmean(x)) for (v, l), x in se.items()},
        "rule": "후보 {v3, +P2, +P3, +P4, 전부} 중 개발 세트 CARE 2% 성공률(시드 3개 평균) 최대(동률이면 단순한 쪽). T1: 반사실 개발 세트 CARE 2% 언어 있음 속도 오차 ≤ v3(AMR은 주행 결정을 따름)",
    }
    (cfg.cache / f"v4_choice_{cfg.domain}.json").write_text(json.dumps(choice, ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("v4 채택(%s): %s", cfg.domain, choice)
    return choice


def stage_main(cfg: V4Config) -> list[dict[str, Any]]:
    """새 테스트 세트(사전 등록 docs/37): v4 본 실험 + v3 설정 대조."""

    for ev in ("test", "cf"):
        expert_reference(cfg, ev)
    jobs = []
    for s in MAIN_SEEDS:
        for m in ("random_shared", "care", "mix_trigger"):
            jobs.append(make_job(cfg, "v4", m, 0.02, s, "test"))
        for m in ("random_shared", "care"):
            jobs.append(make_job(cfg, "v3", m, 0.02, s, "test"))
    for s in MAIN_SEEDS[:5]:
        jobs.append(make_job(cfg, "v4", "full", 1.0, s, "test"))
    jobs.sort(key=lambda j: (j["variant"] != "v4", j["method"] == "full", j["seed"]))
    return run_jobs(cfg, jobs)


def stage_lang(cfg: V4Config) -> list[dict[str, Any]]:
    jobs = [make_job(cfg, "v4", m, b, s, "cf", use_language=lang) for s in MAIN_SEEDS[:5] for m, b in (("care", 0.02), ("full", 1.0)) for lang in (True, False)]
    return run_jobs(cfg, jobs)


def stage_robot(cfg: V4Config) -> list[dict[str, Any]]:
    rc = cfg.for_domain("robot")
    prepare(rc)
    expert_reference(rc, "test")
    jobs = [make_job(rc, "v4", m, 0.02, s, "test") for s in MAIN_SEEDS[:5] for m in ("random_shared", "care", "mix_trigger")]
    jobs += [make_job(rc, "v4", "full", 1.0, s, "test") for s in MAIN_SEEDS[:3]]
    return run_jobs(rc, jobs)


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="v4 개선 개발·확증 실험")
    ap.add_argument("stage", choices=["prepare", "dev", "robotdev", "main", "lang", "robot", "comma"])
    ap.add_argument("--root", default="experiments/exp_130_vla_v4")
    ap.add_argument("--v3-root", default="experiments/exp_120_vla_v3")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cfg = V4Config(root=Path(a.root), v3_root=Path(a.v3_root), workers=a.workers, quick=a.quick)
    if a.quick:
        cfg = replace(cfg, dev_per_cell=1, devcf_per_cell=1, test_per_cell=1, cf_per_cell=1, steps=60, batch=32)
    if a.stage == "prepare":
        prepare(cfg)
    elif a.stage == "dev":
        stage_dev(cfg)
    elif a.stage == "robotdev":
        stage_dev(cfg.for_domain("robot"))
    elif a.stage == "main":
        stage_main(cfg)
    elif a.stage == "lang":
        stage_lang(cfg)
    elif a.stage == "robot":
        stage_robot(cfg)
    elif a.stage == "comma":
        from .comma_v4 import run_comma_v4

        run_comma_v4(cfg, **({"seeds": (0,), "budgets": (0.10,), "steps": cfg.steps} if a.quick else {}))


if __name__ == "__main__":
    main()
