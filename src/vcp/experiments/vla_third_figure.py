"""세 확증 단계(v3 1차, v4 2차, 세 번째 테스트 3차)의 비율형 KPI 비교 그림.

결과 파일만 읽어 그린다(분석은 하지 않는다). 사전 등록으로 고정한 분석 코드
(`vla_third_report.py`, docs/39)를 건드리지 않으려고 별도 모듈로 두었다.

입력
- v3: `<v3-root>/summary/v3_kpi.json`
- v4(2차): `<root>/summary/v4_kpi.json`
- 3차: `<root>/summary/third_kpi.json`, `<root>/summary/tables/third_driving.csv`(v3 설정의 위험 시나리오 성공률)
출력: `<root>/summary/figures/fig_third_kpi.png`

사용: PYTHONPATH=src python -m vcp.experiments.vla_third_figure [--root experiments/exp_130_vla_v4] [--v3-root experiments/exp_120_vla_v3]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
GOALS = {"K1": 0.80, "K2": 0.90, "K3": 0.75, "K6": 0.30, "K7": 0.65}
SHORT = {"K1": "CARE 2% 성공률", "K2": "데이터 효율", "K3": "위험 시나리오", "K6": "언어 오차 감소율", "K7": "실영상 AUROC"}


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        logger.warning("결과 파일 없음: %s", path)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _get(d: dict[str, Any], *keys: str) -> float:
    """중첩 키를 따라가 숫자를 돌려준다. 없으면 NaN."""

    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return float("nan")
        cur = cur[k]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return float("nan")


def _v3_hazard_third(root: Path) -> float:
    """세 번째 테스트에서 v3 설정 CARE 2%의 위험 시나리오 성공률(표 third_driving)."""

    p = root / "summary" / "tables" / "third_driving.csv"
    if not p.exists():
        return float("nan")
    with p.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("설정") == "v3 정책" and row.get("방법") == "CARE":
                try:
                    return float(row["위험 시나리오 성공률"])
                except (KeyError, ValueError):
                    return float("nan")
    return float("nan")


def collect(root: Path, v3_root: Path) -> dict[str, dict[str, float]]:
    """그림에 쓰는 값: 단계별 KPI와 같은 세트의 v3 설정 값."""

    v3k = _load(v3_root / "summary" / "v3_kpi.json")
    v4k = _load(root / "summary" / "v4_kpi.json")
    t3 = _load(root / "summary" / "third_kpi.json")
    vals: dict[str, dict[str, float]] = {
        "v3": {k: _get(v3k, k, "speed_error_reduction" if k == "K6" else "value") for k in GOALS},
        "v4": {k: _get(v4k, k, "speed_error_reduction" if k == "K6" else "value") for k in GOALS},
        "third": {
            "K1": _get(t3, "kpi", "K1", "value"),
            "K2": _get(t3, "kpi", "K2", "value"),
            "K3": _get(t3, "kpi", "K3", "value"),
            "K6": _get(t3, "k6", "K6", "speed_error_reduction"),
            "K7": float("nan"),  # 실영상 새 데이터 없음(3단계 K7 판정 없음)
        },
        # 같은 세트의 v3 설정(K1·K3만)
        "v3_on_v4test": {"K1": _get(v4k, "driving", "v3", "care", "success"), "K3": _get(v4k, "driving", "v3", "care", "hazard")},
        "v3_on_third": {"K1": _get(t3, "tests", "H-v3", "b"), "K3": _v3_hazard_third(root)},
        "p5_explore": {"K7": _get(t3, "comma_p5", "auroc_p5")},
    }
    return vals


def figure(root: Path, v3_root: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import koreanize_matplotlib  # noqa: F401
    except ModuleNotFoundError:  # pragma: no cover
        logger.warning("koreanize_matplotlib 없음: 한글 글꼴이 깨질 수 있다")
    vals = collect(root, v3_root)
    ks = list(GOALS)
    x = np.arange(len(ks))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.bar(x - w, [vals["v3"][k] for k in ks], w, label="1차: v3 설정, v3 테스트(800000번대)", color="#cbd2d9", hatch="//", edgecolor="#7b8794")
    ax.bar(x, [vals["v4"][k] for k in ks], w, label="2차: v4 설정, 새 테스트(1000000번대)", color="#63a4e8")
    ax.bar(x + w, [vals["third"][k] for k in ks], w, label="3차: v4 설정, 세 번째 테스트(1200000번대)", color="#1a4f8b")
    # 같은 세트의 v3 설정 값(K1·K3): 2차·3차 막대 위에 표시
    first = True
    for off, key in ((0.0, "v3_on_v4test"), (w, "v3_on_third")):
        for k, v in vals[key].items():
            if math.isfinite(v):
                i = ks.index(k)
                ax.plot(i + off, v, marker="x", color="k", ms=7, mew=1.8, ls="none", label="같은 세트의 v3 설정(K1·K3)" if first else None)
                first = False
    p5 = vals["p5_explore"]["K7"]
    if math.isfinite(p5):
        ax.plot(ks.index("K7") + w, p5, marker="D", mfc="none", mec="#1a4f8b", ms=6, ls="none", label="3차 v4+P5(실영상, 탐색)")
    for i, k in enumerate(ks):
        ax.plot([i - 0.45, i + 0.45], [GOALS[k]] * 2, color="#c53030", ls="--", lw=1.5, label="목표" if i == 0 else None)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, [f"{k}\n{SHORT[k]}" for k in ks], fontsize=8)
    ax.set_ylabel("값(K6는 감소율)")
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    out = root / "summary" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "fig_third_kpi.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    logger.info("그림 저장: %s", path)
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/exp_130_vla_v4")
    ap.add_argument("--v3-root", default="experiments/exp_120_vla_v3")
    a = ap.parse_args()
    figure(Path(a.root), Path(a.v3_root))


if __name__ == "__main__":
    main()
