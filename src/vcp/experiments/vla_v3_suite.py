from __future__ import annotations

"""v3 실험 스위트: 핵심 지표(K1~K7) 중심 재실험(docs/33).

v2(`vla_curation_suite.py`, exp_110) 대비 변경
- 벤치마크
  - 초기 속도를 지시 목표 속도와 분리해 언어 누출을 막는다(decouple_initial_speed=True).
  - 충돌 시 자차를 되밀지 않는다(collision_pushback=False).
  - 반사실 언어 평가 세트(eval="cf")를 둔다. 같은 시나리오·시드를 세 스타일 지시문으로 평가한다.
- 정책: VLA-lite + 검출 특징 토큰(use_features=True)
- 선별: 공유 저장소(select_shared). 무작위 저장소 클립을 시드별로 고정하고, 점수 몫만 방법마다 다르게 고른다.

v2 결과의 재현성을 지키기 위해 v2 스위트는 수정하지 않고, 유틸리티만 가져다 쓴다.
"""

import json
import logging
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .vla_curation_suite import (
    HAZARD_IDS,
    CurationSuiteConfig,
    _json_default,
    chunk_targets,
    clip_ids,
    episode_success,
    episode_tokens,
)

logger = logging.getLogger(__name__)

SEED_BASES_V3 = {
    "driving": {"pool": 100000, "val": 150000, "test": 800000, "cf": 850000, "testpool": 300000},
    "robot": {"pool": 400000, "val": 450000, "test": 900000, "cf": 950000, "testpool": 600000},
}
METHODS_V3 = ("full", "random", "random_shared", "care", "mix_trigger", "mix_oracle", "trigger_only")


@dataclass(slots=True)
class V3Config:
    root: Path = Path("experiments/exp_120_vla_v3")
    v2_root: Path = Path("experiments/exp_110_vla_curation")
    comma_dir: Path = Path("data/processed/comma_speedchallenge")
    domain: str = "driving"
    pool_episodes: int = 1200
    testpool_episodes: int = 150
    val_per_cell: int = 2
    test_per_cell: int = 7
    cf_per_cell: int = 3
    clip_len: int = 30
    steps: int = 6000
    batch: int = 128
    lam: float = 0.5
    reservoir: float = 0.9
    use_features: bool = True
    collision_pushback: bool = False
    decouple_initial_speed: bool = True
    workers: int = 4
    quick: bool = False

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def summary(self) -> Path:
        return self.root / "summary"

    def for_domain(self, domain: str) -> "V3Config":
        if domain == "robot":
            if self.quick:  # 스모크: 작은 크기를 유지
                return replace(self, domain="robot")
            return replace(self, domain="robot", pool_episodes=600, testpool_episodes=90, test_per_cell=4, cf_per_cell=2)
        return replace(self, domain=domain)

    def legacy(self) -> CurationSuiteConfig:
        """v2 유틸리티(score_pool 등)에 넘길 설정 객체(같은 루트·도메인)."""

        return CurationSuiteConfig(root=self.root, domain=self.domain, clip_len=self.clip_len, workers=self.workers)


# ----------------------------------------------------------------------
# 준비: 풀 생성, 점수기 복사, 풀 점수화
# ----------------------------------------------------------------------
def pool_path(cfg: V3Config, kind: str = "pool") -> Path:
    return cfg.cache / f"{kind}_{cfg.domain}"


def prepare(cfg: V3Config) -> None:
    from ..vla.pool import PoolConfig, generate_pool
    from .vla_curation_suite import score_pool

    cfg.cache.mkdir(parents=True, exist_ok=True)
    # 점수기는 v2와 같은 초기 라벨 셋으로 학습한 것을 그대로 쓴다(재학습해도 같은 데이터·시드)
    for ext in ("pt", "json"):
        src = cfg.v2_root / "cache" / f"scorer_{cfg.domain}.{ext}"
        dst = cfg.cache / f"scorer_{cfg.domain}.{ext}"
        if not dst.exists():
            if not src.exists():
                from .vla_curation_suite import train_scorer

                train_scorer(replace(cfg.legacy(), root=cfg.v2_root))
            shutil.copy2(src, dst)
    bases = SEED_BASES_V3[cfg.domain]
    for kind, n in (("pool", cfg.pool_episodes), ("testpool", cfg.testpool_episodes)):
        p = pool_path(cfg, kind)
        if (p.parent / f"{p.name}.npz").exists():
            continue
        t0 = time.perf_counter()
        generate_pool(
            p,
            PoolConfig(
                n_episodes=n,
                seed_base=bases[kind],
                domain=cfg.domain,
                workers=cfg.workers,
                collision_pushback=cfg.collision_pushback,
                decouple_initial_speed=cfg.decouple_initial_speed,
            ),
        )
        logger.info("%s 생성 %.1fs", p, time.perf_counter() - t0)
    score_pool(cfg.legacy())


def expert_reference(cfg: V3Config, eval_set: str) -> dict[str, Any]:
    from ..vla.closed_loop import run_closed_loop

    path = cfg.cache / f"expert_{cfg.domain}_{eval_set}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    res = run_closed_loop(None, eval_specs(cfg, eval_set), domain=cfg.domain, collision_pushback=cfg.collision_pushback, decouple_initial_speed=cfg.decouple_initial_speed)
    cfg.cache.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return res


def eval_specs(cfg: V3Config, eval_set: str) -> list[Any]:
    from ..vla.closed_loop import make_test_specs

    base = SEED_BASES_V3[cfg.domain][eval_set]
    if eval_set == "cf":
        return make_test_specs(cfg.domain, cfg.cf_per_cell, base, counterfactual=True)
    per_cell = cfg.val_per_cell if eval_set == "val" else cfg.test_per_cell
    return make_test_specs(cfg.domain, per_cell, base)


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


def _cfg_from(d: dict[str, Any], domain: str) -> V3Config:
    d = dict(d)
    for k in ("root", "v2_root", "comma_dir"):
        d[k] = Path(d[k])
    return V3Config(**d).for_domain(domain) if domain != d.get("domain") else V3Config(**d)


def _pool(cfg: V3Config, kind: str = "pool") -> dict[str, Any]:
    key = f"{kind}:{cfg.domain}"
    if key not in _W:
        from ..vla.pool import load_pool

        pool, meta = load_pool(pool_path(cfg, kind))
        entry: dict[str, Any] = {"pool": pool, "meta": meta, "clip": clip_ids(pool["ep"], cfg.clip_len)}
        if kind == "pool":
            entry["scores"] = dict(np.load(cfg.cache / f"scores_{cfg.domain}.npz"))
        _W[key] = entry
    return _W[key]


def policy_data(cfg: V3Config, P: dict[str, Any]) -> Any:
    from ..vla.policy import PolicyData, PoolImageSource

    pool = P["pool"]
    return PolicyData(
        images=PoolImageSource.from_pool(pool, cfg.domain),
        group=P["clip"],
        ego_v=pool["ego_v"],
        action=pool["expert_cmd"],
        tokens=episode_tokens(pool, P["meta"]),
        domain=cfg.domain,
        features=pool["X_v1v2"].astype(np.float32) if cfg.use_features else None,
    )


def select_v3(cfg: V3Config, job: dict[str, Any], P: dict[str, Any]) -> np.ndarray:
    """v3 선별. 혼합 계열은 모두 공유 저장소(같은 시드 → 같은 저장소 클립)를 쓴다."""

    from ..vla.curation import select, select_shared

    m, b = job["method"], float(job["budget"])
    group = P["pool"]["ep"]
    if m == "full":
        return P["clip"] >= 0
    s = P["scores"]
    rng = np.random.default_rng(1000 + int(job["seed"]))
    lam = float(job.get("lam", cfg.lam))
    res = float(job.get("reservoir", cfg.reservoir))
    if m == "random":
        return select("random", b, group, cfg.clip_len, rng)
    if m == "trigger_only":
        return select("action_trigger", b, group, cfg.clip_len, rng, action=s["action"])
    score = {
        "random_shared": None,
        "care": s["event_score"],
        "mix_trigger": (-s["action"]).astype(np.float32),
        "mix_oracle": s["oracle"].astype(np.float32),
    }[m]
    ent = s["entropy"] if m == "care" else None
    return select_shared(score, b, group, cfg.clip_len, rng, reservoir=res, entropy=ent, lam=lam if m == "care" else 0.0)


def job_key(job: dict[str, Any]) -> str:
    parts = [job["domain"], job["eval"], job["method"], f"b{job['budget']:.2f}", f"s{job['seed']}", f"st{job['steps']}"]
    if not job.get("use_language", True):
        parts.append("nolang")
    if not job.get("use_features", True):
        parts.append("nofeat")
    if "lam" in job or "reservoir" in job:
        parts.append(f"l{job.get('lam', 'd')}_r{job.get('reservoir', 'd')}")
    return "__".join(str(p) for p in parts)


def make_job(cfg: V3Config, method: str, budget: float, seed: int, eval_set: str = "test", **kw: Any) -> dict[str, Any]:
    job = {"domain": cfg.domain, "eval": eval_set, "method": method, "budget": float(budget), "seed": int(seed), "steps": int(kw.pop("steps", cfg.steps))}
    job.update(kw)
    return job


def open_loop(model: Any, cfg: V3Config, stride: int = 3) -> dict[str, float]:
    from ..vla.policy import predict_open_loop

    T = _pool(cfg, "testpool")
    pool = T["pool"]
    data = policy_data(cfg, T)
    valid = np.where(T["clip"] >= 0)[0][::stride]
    pred = predict_open_loop(model, data, valid)
    target = chunk_targets(pool["expert_cmd"], T["clip"], valid, pred.shape[1])
    err = np.abs(pred - target)
    hazard = np.isin(pool["y"][valid], HAZARD_IDS)
    return {"mae": float(err.mean()), "mae_hazard": float(err[hazard].mean()) if hazard.any() else float("nan"), "n_frames": int(len(valid)), "n_hazard": int(hazard.sum())}


def run_job(job: dict[str, Any]) -> dict[str, Any]:
    from ..vla.closed_loop import run_closed_loop
    from ..vla.curation import selection_stats
    from ..vla.policy import PolicyConfig, make_policy_fn, train_policy

    cfg = _cfg_from(_W["cfg"], job["domain"])
    cfg = replace(cfg, use_features=bool(job.get("use_features", cfg.use_features)))
    out_path = cfg.runs / f"{job_key(job)}.json"
    if out_path.exists():
        return json.loads(out_path.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    P = _pool(cfg)
    mask = select_v3(cfg, job, P) & (P["clip"] >= 0)
    train_idx = np.where(mask)[0]
    stats = selection_stats(mask, P["pool"]["y"], P["pool"]["ep"], 6, HAZARD_IDS)
    data = policy_data(cfg, P)
    pcfg = PolicyConfig(steps=job["steps"], batch=cfg.batch, seed=int(job["seed"]), use_language=job.get("use_language", True), use_features=cfg.use_features, threads=1)
    model, log = train_policy(data, train_idx, pcfg)
    closed = run_closed_loop(make_policy_fn(model, cfg.domain), eval_specs(cfg, job["eval"]), domain=cfg.domain, collision_pushback=cfg.collision_pushback, decouple_initial_speed=cfg.decouple_initial_speed)
    result = {
        "job": job,
        "key": job_key(job),
        "selection": stats,
        "n_train_frames": int(len(train_idx)),
        # 실제로 쓴 선별 하이퍼파라미터(튜닝 결과가 바뀌면 캐시 결과와 대조하기 위함)
        "selection_params": {"lam": float(job.get("lam", cfg.lam)), "reservoir": float(job.get("reservoir", cfg.reservoir))},
        "train_log": {k: v for k, v in log.items() if k != "loss_curve"},
        "closed_loop": closed,
        "open_loop": open_loop(model, cfg) if job["eval"] == "test" else {},
        "seconds": time.perf_counter() - t0,
    }
    cfg.runs.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return result


def run_jobs(cfg: V3Config, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pending = [j for j in jobs if not (cfg.runs / f"{job_key(j)}.json").exists()]
    logger.info("작업 %d개(남은 것 %d개)", len(jobs), len(pending))
    d = asdict(cfg)
    d = {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}
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


# ----------------------------------------------------------------------
# 단계
# ----------------------------------------------------------------------
def success_of(cfg: V3Config, r: dict[str, Any]) -> float:
    ex = expert_reference(cfg, r["job"]["eval"])
    return float(episode_success(r["closed_loop"]["episodes"], ex, cfg.domain == "robot").mean())


def stage_pilot(cfg: V3Config) -> list[dict[str, Any]]:
    """정책 v3(특징 토큰) 학습 단계 수 확인과 특징 토큰 효과의 검증 세트 점검."""

    jobs = []
    for st in (3000, 6000):
        for m, b in (("full", 1.0), ("random_shared", 0.02), ("care", 0.02)):
            jobs.append(make_job(cfg, m, b, 0, "val", steps=st))
    jobs.append(make_job(cfg, "full", 1.0, 0, "val", use_features=False))
    jobs.append(make_job(cfg, "care", 0.02, 0, "val", use_features=False))
    res = run_jobs(cfg, jobs)
    for r in res:
        logger.info("파일럿 %s: 검증 성공률 %.3f", r["key"], success_of(cfg, r))
    return res


def stage_tune(cfg: V3Config, seeds: tuple[int, ...] = (0, 1)) -> dict[str, Any]:
    grid = [(lam, r) for lam in (0.0, 0.5) for r in (0.8, 0.9, 0.95)]
    jobs = [make_job(cfg, "care", 0.02, s, "val", lam=lam, reservoir=r) for lam, r in grid for s in seeds]
    jobs += [make_job(cfg, "random_shared", 0.02, s, "val") for s in seeds]
    res = run_jobs(cfg, jobs)
    table: dict[tuple[float, float], list[float]] = {}
    for r in res:
        if r["job"]["method"] == "care":
            table.setdefault((r["job"]["lam"], r["job"]["reservoir"]), []).append(success_of(cfg, r))
    best = max(table, key=lambda k: float(np.mean(table[k])))
    rand = [success_of(cfg, r) for r in res if r["job"]["method"] == "random_shared"]
    choice = {
        "lam": best[0],
        "reservoir": best[1],
        "criterion": f"검증 성공률(시드 {len(seeds)}개 평균) 최대",
        "grid": [{"lam": k[0], "reservoir": k[1], "success": float(np.mean(v))} for k, v in sorted(table.items())],
        "random_shared_val": float(np.mean(rand)),
    }
    (cfg.cache / f"tune_{cfg.domain}.json").write_text(json.dumps(choice, ensure_ascii=False, indent=1), encoding="utf-8")
    return choice


def tuned(cfg: V3Config) -> V3Config:
    p = cfg.cache / f"tune_{cfg.domain}.json"
    if p.exists():
        t = json.loads(p.read_text(encoding="utf-8"))
        return replace(cfg, lam=float(t["lam"]), reservoir=float(t["reservoir"]))
    return cfg


MAIN_SEEDS = tuple(range(10))


def stage_main(cfg: V3Config) -> list[dict[str, Any]]:
    """주행 본 실험(사전 등록 docs/34): K1~K4."""

    cfg = tuned(cfg)
    jobs = []
    for s in MAIN_SEEDS:
        for m in ("random_shared", "care", "mix_trigger", "mix_oracle", "trigger_only"):
            jobs.append(make_job(cfg, m, 0.02, s))
    for s in MAIN_SEEDS[:5]:
        jobs.append(make_job(cfg, "full", 1.0, s))
        for b in (0.01, 0.05):
            for m in ("random_shared", "care"):
                jobs.append(make_job(cfg, m, b, s))
    jobs.sort(key=lambda j: (j["budget"] != 0.02 or j["method"] not in ("care", "random_shared", "mix_trigger"), j["seed"]))
    return run_jobs(cfg, jobs)


def stage_lang(cfg: V3Config) -> list[dict[str, Any]]:
    """반사실 언어 평가(K6): 같은 장면을 세 스타일 지시문으로."""

    cfg = tuned(cfg)
    jobs = []
    for s in MAIN_SEEDS[:5]:
        for m, b in (("full", 1.0), ("care", 0.02)):
            for lang in (True, False):
                jobs.append(make_job(cfg, m, b, s, "cf", use_language=lang))
    return run_jobs(cfg, jobs)


def stage_robot(cfg: V3Config) -> list[dict[str, Any]]:
    """AMR(K5): 준비 → AMR 검증 튜닝 → 테스트."""

    rc = cfg.for_domain("robot")
    prepare(rc)
    expert_reference(rc, "val")
    expert_reference(rc, "test")
    stage_tune(rc, seeds=(0,))
    rc = tuned(rc)
    jobs = []
    for s in MAIN_SEEDS[:5]:
        for m in ("random_shared", "care", "mix_trigger"):
            jobs.append(make_job(rc, m, 0.02, s))
    for s in MAIN_SEEDS[:3]:
        jobs.append(make_job(rc, "full", 1.0, s))
    return run_jobs(rc, jobs)


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="v3 핵심 지표 재실험")
    ap.add_argument("stage", choices=["prepare", "pilot", "tune", "main", "lang", "robot", "comma"])
    ap.add_argument("--root", default="experiments/exp_120_vla_v3")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = V3Config(root=Path(args.root), workers=args.workers, quick=args.quick)
    if args.quick:
        cfg = replace(cfg, pool_episodes=24, testpool_episodes=6, val_per_cell=1, test_per_cell=1, cf_per_cell=1, steps=60, batch=32)
    if args.stage == "prepare":
        prepare(cfg)
        for ev in ("val", "test", "cf"):
            expert_reference(cfg, ev)
    elif args.stage == "pilot":
        stage_pilot(cfg)
    elif args.stage == "tune":
        stage_tune(cfg)
    elif args.stage == "main":
        stage_main(cfg)
    elif args.stage == "lang":
        stage_lang(cfg)
    elif args.stage == "robot":
        stage_robot(cfg)
    elif args.stage == "comma":
        from .comma_v3 import run_comma_v3

        run_comma_v3(cfg)


if __name__ == "__main__":
    main()
