from __future__ import annotations

"""v3 실주행(comma) 개루프 실험(K7): 검출 특징 토큰 정책과 공유 저장소 선별.

설계(v2 comma_curation과 같은 5-fold 블록 교차검증·10fps·클립 20·청크 10)에서 바꾼 점은 두 가지다.
- 정책: VLA-lite + YOLO 검출 특징 토큰(use_features=True)
- 선별: 공유 저장소(random_shared, care, mix_trigger)와 감속 트리거 단독, 전체(100%)
점수기·fold 분할은 v2와 같으므로 fold 캐시(comma_fold*.npz)를 v2에서 복사해 쓴다.
"""

import json
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from .comma_curation import N_BLOCKS, prepare_fold, run_comma_job
from .vla_curation_suite import CurationSuiteConfig

logger = logging.getLogger(__name__)
COMMA_V3_METHODS = ("random_shared", "care", "mix_trigger", "action_trigger")


def run_comma_v3(cfg: Any, budgets: tuple[float, ...] = (0.10, 0.20), seeds: tuple[int, ...] = (0, 1), steps: int = 1500) -> list[dict[str, Any]]:
    from .vla_v3_suite import tuned

    t = tuned(replace(cfg, domain="driving"))
    legacy = CurationSuiteConfig(root=Path(cfg.root), domain="driving", comma_dir=Path(cfg.comma_dir), workers=cfg.workers, lam=t.lam, reservoir=t.reservoir)
    legacy.cache.mkdir(parents=True, exist_ok=True)
    for k in range(N_BLOCKS):
        src = Path(cfg.v2_root) / "cache" / f"comma_fold{k}.npz"
        dst = legacy.cache / f"comma_fold{k}.npz"
        if not dst.exists():
            if src.exists():
                shutil.copy2(src, dst)
            else:
                prepare_fold(legacy, k)
    jobs = []
    for k in range(N_BLOCKS):
        for s in seeds:
            for feat in (True, False):
                jobs.append({"cfg": legacy, "fold": k, "method": "full", "budget": 1.0, "seed": s, "steps": steps, "use_features": feat})
            for b in budgets:
                for m in COMMA_V3_METHODS:
                    jobs.append({"cfg": legacy, "fold": k, "method": m, "budget": b, "seed": s, "steps": steps, "use_features": True})
    for j in jobs:
        j["key"] = f"comma3__f{j['fold']}__{j['method']}__b{j['budget']:.2f}__s{j['seed']}__st{j['steps']}" + ("" if j["use_features"] else "__nofeat")
    pending = [j for j in jobs if not (legacy.runs / f"{j['key']}.json").exists()]
    logger.info("comma v3 작업 %d개(남은 것 %d개)", len(jobs), len(pending))
    if pending:
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            list(ex.map(run_comma_job, pending))
    return [json.loads((legacy.runs / f"{j['key']}.json").read_text(encoding="utf-8")) for j in jobs]
