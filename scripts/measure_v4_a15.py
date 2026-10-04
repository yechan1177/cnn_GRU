"""A15 v4 짧은 측정(docs/29 10절, docs/30 7절).

1. 특징 이력 갱신 시간: 주행 21에피소드(시나리오 7 × 스타일 3, 개발 시드 160000번대) 가짜 정책(0 명령) 폐루프를
   H=2(v3)와 H=8(s=2)로 1회씩 돌려 `timing.features_s`를 정책 호출 수로 나눈 프레임당 갱신 시간을 잰다.
   링 버퍼 쌓기만의 비용은 같은 배치 크기에서 별도로 잰다(추적기 갱신 제외).
2. S1 점수 몫 에피소드 상한: v3 주행 풀(exp_120 캐시, 읽기만)에서 예산 2%, CARE 점수
   (event_score + 0.5·entropy), ρ=0.9로 c=None(v3)과 c=1의 점수 몫 에피소드 수·위험 프레임 비중을 비교한다.
   rng는 v3 스위트 규칙(`default_rng(1000 + 시드)`)을 따른다. 보조로 시드 1·2, 트리거 혼합, 예산 10%도 기록한다.

사용: PYTHONPATH=src python scripts/measure_v4_a15.py [--out 경로] [--skip-timing]
CPU를 다른 실험과 나눠 쓰므로 무거운 반복은 하지 않는다(각 측정 1회).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from vcp.experiments.vla_curation_suite import HAZARD_IDS, clip_ids
from vcp.vla.closed_loop import make_test_specs, run_closed_loop
from vcp.vla.curation import select_shared, shared_reservoir_mask
from vcp.vla.obs import FEATURE_DIM, feature_history_offsets
from vcp.vla.pool import load_pool

logger = logging.getLogger("measure_v4_a15")

V3_CACHE = Path("/home/user/cnn_GRU/experiments/exp_120_vla_v3/cache")
DEFAULT_OUT = Path("experiments/exp_121_v4_a15_obs_curation/summary/a15_measure.json")


def measure_timing(n_per_cell: int = 1, seed_base: int = 160000) -> dict[str, Any]:
    """H=2와 H=8의 폐루프 특징 갱신 시간(프레임당, 배치 전체)."""

    specs = make_test_specs("driving", n_per_cell, seed_base)
    out: dict[str, Any] = {"n_episodes": len(specs), "seed_base": seed_base}

    def zero_policy(obs: dict[str, np.ndarray]) -> np.ndarray:
        return np.zeros(len(obs["proprio"]), dtype=np.float32)

    for h in (2, 8):
        res = run_closed_loop(zero_policy, specs, feature_history=h, feature_stride=2)
        n_calls = int(round(res["config"]["duration_s"] * res["config"]["fps"])) - 1
        tm = res["timing"]
        out[f"H{h}"] = {
            "features_s": tm["features_s"],
            "wall_s": tm["wall_s"],
            "n_policy_calls": n_calls,
            "features_ms_per_frame": 1000.0 * tm["features_s"] / n_calls,
            "features_ms_per_frame_per_episode": 1000.0 * tm["features_s"] / n_calls / len(specs),
        }
        logger.info("H=%d: features_s=%.3f, wall_s=%.1f", h, tm["features_s"], tm["wall_s"])
    # 링 버퍼 쌓기만의 비용(추적기 제외): FeatureHistory.update의 인덱싱·전치와 같은 연산
    b = len(specs)
    for h in (2, 8):
        offsets = feature_history_offsets(h, 2)
        slots = (h - 1) * 2 + 1
        buf = np.zeros((slots, b, FEATURE_DIM), dtype=np.float32)
        n_rep = 20000
        t0 = time.perf_counter()
        for t in range(n_rep):
            idx = np.maximum(t - offsets, 0) % slots
            np.ascontiguousarray(buf[idx].transpose(1, 0, 2))
        out[f"H{h}"]["stack_only_us_per_frame"] = 1e6 * (time.perf_counter() - t0) / n_rep
    return out


def _share_stats(mask: np.ndarray, res: np.ndarray, ep: np.ndarray, hazard: np.ndarray, clip: np.ndarray) -> dict[str, Any]:
    share = mask & ~res
    share_eps = np.unique(ep[share])
    share_clips = np.unique(clip[share])
    pairs = np.unique(np.stack([clip[share], ep[share]], axis=1), axis=0)  # (클립, 에피소드) 쌍
    per_ep = np.unique(pairs[:, 1], return_counts=True)[1]
    return {
        "n_selected_frames": int(mask.sum()),
        "n_score_clips": int(len(share_clips)),
        "n_score_episodes": int(len(share_eps)),
        "max_clips_per_episode": int(per_ep.max()) if len(per_ep) else 0,
        "score_hazard_frac": float(hazard[share].mean()) if share.any() else 0.0,
        "reservoir_hazard_frac": float(hazard[res].mean()) if res.any() else 0.0,
        "selected_hazard_frac": float(hazard[mask].mean()) if mask.any() else 0.0,
        "n_selected_episodes": int(len(np.unique(ep[mask]))),
    }


def measure_cap(seeds: tuple[int, ...] = (0, 1, 2), budgets: tuple[float, ...] = (0.02, 0.1), reservoir: float = 0.9,
                lam: float = 0.5, clip_len: int = 30) -> dict[str, Any]:
    """v3 주행 풀에서 c=None과 c=1의 점수 몫 구성 비교."""

    pool, _ = load_pool(V3_CACHE / "pool_driving")
    sc = dict(np.load(V3_CACHE / "scores_driving.npz"))
    ep = pool["ep"]
    clip = clip_ids(ep, clip_len)
    hazard = np.isin(pool["y"], HAZARD_IDS)
    methods = {
        "care": (sc["event_score"], sc["entropy"], lam),
        "mix_trigger": ((-sc["action"]).astype(np.float32), None, 0.0),
    }
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        for method, (score, ent, lm) in methods.items():
            for seed in seeds:
                res = shared_reservoir_mask(budget, ep, clip_len, np.random.default_rng(1000 + seed), reservoir)
                for cap in (None, 1):
                    info: dict[str, Any] = {}
                    mask = select_shared(score, budget, ep, clip_len, np.random.default_rng(1000 + seed), reservoir,
                                         entropy=ent, lam=lm, per_group_cap=cap, info=info)
                    row = {"budget": budget, "method": method, "seed": seed, "per_group_cap": cap,
                           **_share_stats(mask, res, ep, hazard, clip), "n_cap_overflow": info["n_cap_overflow"],
                           "k": info["k"], "n_reservoir": info["n_reservoir"]}
                    rows.append(row)
                    logger.info("%s", row)
    return {
        "pool": str(V3_CACHE / "pool_driving.npz"),
        "n_frames": int(len(ep)),
        "n_episodes": int(len(np.unique(ep))),
        "pool_hazard_frac": float(hazard.mean()),
        "budgets": list(budgets),
        "reservoir": reservoir,
        "lam": lam,
        "clip_len": clip_len,
        "rng": "np.random.default_rng(1000 + seed) (v3 스위트 규칙)",
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--skip-timing", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result: dict[str, Any] = {"date": time.strftime("%Y-%m-%d %H:%M:%S")}
    result["cap"] = measure_cap()
    if not args.skip_timing:
        result["timing"] = measure_timing()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("저장: %s", args.out)


if __name__ == "__main__":
    main()
