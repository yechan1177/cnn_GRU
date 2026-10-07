"""4차 심사 N3 탐색: 개발 세트에서 v4 + P5와 v4의 폐루프 성공률(CARE 2%, 시드 0~4) 요약.

입력: experiments/exp_130_vla_v4/runs/driving__dev__{all_p7_p5,all_p7}__care__b0.02__s{0..4}__st6000.json
출력: experiments/exp_130_vla_v4/summary/p5dev_explore.json
사용: PYTHONPATH=src python scripts/p5dev_explore.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vcp.experiments.vla_v3_report import episode_success
from vcp.experiments.vla_v4_suite import V4Config, expert_reference

ROOT = Path("experiments/exp_130_vla_v4")


def main() -> None:
    cfg = V4Config(root=ROOT)
    ex = expert_reference(cfg, "dev")
    out = {}
    for v in ("all_p7_p5", "all_p7"):
        files = [ROOT / "runs" / f"driving__dev__{v}__care__b0.02__s{s}__st6000.json" for s in range(5)]
        rs = [json.loads(f.read_text(encoding="utf-8")) for f in files if f.exists()]
        S = np.array([episode_success(r["closed_loop"]["episodes"], ex, False).mean() for r in rs])
        out[v] = {"mean": float(S.mean()), "sd": float(S.std(ddof=1)), "n": len(S), "by_seed": S.round(3).tolist()}
    (ROOT / "summary" / "p5dev_explore.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
