from __future__ import annotations

"""v4 실주행(comma) 개루프 실험(K7): 시뮬레이터 개발 세트에서 고른 v4 설정을 실영상에 그대로 적용한다.

- 실영상에서는 아무것도 튜닝하지 않는다(docs/36 4절). v3 실영상 실험과 같은 fold·시드·단계 수(T=1500)·예산이다.
- fold 캐시는 점수기 확률(probs)을 담도록 v4 루트에서 새로 만든다(점수기 학습은 v2·v3와 같은 함수·시드).
- 작업 키: `comma4__f{fold}__{method}__b{budget}__s{seed}__st{steps}`
"""

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from .comma_curation import N_BLOCKS, prepare_fold, run_comma_job
from .vla_curation_suite import CurationSuiteConfig

logger = logging.getLogger(__name__)
COMMA_V4_METHODS = ("random_shared", "care", "mix_trigger")


def run_comma_v4(
    cfg: Any,
    budgets: tuple[float, ...] = (0.10, 0.20),
    seeds: tuple[int, ...] = (0, 1),
    steps: int = 1500,
    extra_policy: dict[str, Any] | None = None,
    key_suffix: str = "",
    methods: tuple[str, ...] = COMMA_V4_METHODS,
) -> list[dict[str, Any]]:
    """실영상 v4. extra_policy를 주면 정책 설정에 더한다(예: A5 P5 자차 운동 이력), key_suffix로 결과를 구분한다."""

    from .vla_v4_suite import POLICY_KEYS, variant_params

    v3c = cfg.v3()
    from dataclasses import replace as _replace

    params = variant_params("v4", _replace(cfg, domain="driving"))  # 실영상은 주행 개발 세트 조합을 쓴다
    pparams = {k: params[k] for k in POLICY_KEYS if k in params} | dict(extra_policy or {})
    cap = params.get("per_group_cap")
    legacy = CurationSuiteConfig(root=Path(cfg.root), domain="driving", comma_dir=Path(cfg.comma_dir), workers=cfg.workers, lam=v3c.lam, reservoir=v3c.reservoir, quick=cfg.quick)
    legacy.cache.mkdir(parents=True, exist_ok=True)
    folds = range(N_BLOCKS if not cfg.quick else 1)
    for k in folds:
        if not (legacy.cache / f"comma_fold{k}.npz").exists():
            prepare_fold(legacy, k)
    jobs = []
    for k in folds:
        for s in seeds:
            jobs.append({"cfg": legacy, "fold": k, "method": "full", "budget": 1.0, "seed": s, "steps": steps, "use_features": True, "policy_params": pparams, "hazard_weight": params.get("hazard_weight", 0.0)})
            for b in budgets:
                for m in methods:
                    jobs.append({"cfg": legacy, "fold": k, "method": m, "budget": b, "seed": s, "steps": steps, "use_features": True, "policy_params": pparams, "per_group_cap": cap, "hazard_weight": params.get("hazard_weight", 0.0)})
    for j in jobs:
        j["key"] = f"comma4__f{j['fold']}__{j['method']}__b{j['budget']:.2f}__s{j['seed']}__st{j['steps']}" + key_suffix
    pending = [j for j in jobs if not (legacy.runs / f"{j['key']}.json").exists()]
    logger.info("comma v4 작업 %d개(남은 것 %d개), 정책 설정 %s, 에피소드 상한 %s", len(jobs), len(pending), pparams, cap)
    if pending:
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            list(ex.map(run_comma_job, pending))
    return [json.loads((legacy.runs / f"{j['key']}.json").read_text(encoding="utf-8")) for j in jobs]
