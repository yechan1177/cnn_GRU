from __future__ import annotations

"""v4 핵심 지표 집계(docs/36, 사전 등록 docs/37). 처음 보는 새 테스트 세트(1000000번대)에서 계산한다.

- K1~K7 정의·판정은 v3(docs/34)와 같다. K4는 v4 설정에서 CARE − 저장소+감속 트리거다.
- 추가: 같은 새 테스트 세트에서 v4 설정 − v3 설정(CARE 2%, 무작위 2%)의 개선 효과.
- 개발 세트 결과(cache/v4_choice.json)도 표로 남긴다.
결과: `<root>/summary/v4_kpi.json`, `tables/v4_*.md`, `figures/fig_v4_*.png`
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .vla_report import _fmt, _write_table, hierarchical_bootstrap, holm, seed_t
from .vla_v3_report import TARGETS, episode_success
from .vla_v4_suite import V4Config, expert_reference

logger = logging.getLogger(__name__)
V3_KPI_PATH = Path("experiments/exp_120_vla_v3/summary/v3_kpi.json")
LABELS = {"full": "전체(100%)", "random_shared": "무작위(공유 저장소)", "care": "CARE", "mix_trigger": "저장소 + 감속 트리거"}


def _runs(root: Path, prefix: str) -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "runs").glob(f"{prefix}*.json"))]


def _group(runs: list[dict[str, Any]]) -> dict[tuple, list[dict[str, Any]]]:
    g: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        j = r["job"]
        g[(j["eval"], j["variant"], j["method"], round(j["budget"], 2), j.get("use_language", True))].append(r)
    for v in g.values():
        v.sort(key=lambda r: r["job"]["seed"])
    return g


def _S(rs: list[dict[str, Any]], ex: dict[str, Any], moving: bool) -> tuple[np.ndarray, list[int]]:
    return np.stack([episode_success(r["closed_loop"]["episodes"], ex, moving) for r in rs]).astype(float), [r["job"]["seed"] for r in rs]


def _pair(a: tuple[np.ndarray, list[int]], b: tuple[np.ndarray, list[int]]) -> tuple[np.ndarray, np.ndarray]:
    common = sorted(set(a[1]) & set(b[1]))
    return a[0][[a[1].index(s) for s in common]], b[0][[b[1].index(s) for s in common]]


def _ms(x: np.ndarray) -> tuple[float, float]:
    m = x.mean(1)
    return float(m.mean()), float(m.std(ddof=1)) if len(m) > 1 else 0.0


def _hazard(rs: list[dict[str, Any]], ex: dict[str, Any], moving: bool) -> np.ndarray:
    out = []
    for r in rs:
        eps = r["closed_loop"]["episodes"]
        s = episode_success(eps, ex, moving)
        out.append(float(s[np.array([bool(e["hazard"]) for e in eps])].mean()))
    return np.asarray(out)


def build_v4_report(root: Path, quick: bool = False) -> dict[str, Any]:
    cfg = V4Config(root=root, quick=quick)
    if quick:
        from dataclasses import replace

        cfg = replace(cfg, dev_per_cell=1, devcf_per_cell=1, test_per_cell=1, cf_per_cell=1, steps=60, batch=32)
    tables = root / "summary" / "tables"
    kpi: dict[str, Any] = {}

    # ---------- 개발 세트(선택 근거) ----------
    names = {"v3": "v3 기준", "p2": "+P2 시간 특징 인코더", "p3": "+P3 위험 보조 헤드", "t1": "+T1 지시문 드롭아웃", "p4": "+P4 위험 가중 손실", "p5": "+P5 자차 운동 이력", "all": "전부(P2+P3+T1+P4)", "all_p6": "전부+P6 수치 목표", "all_p7": "전부+P7 목표 속도 잔차"}
    kpi["dev_choice"] = {}
    rows = []
    for dom, dname in (("driving", "주행"), ("robot", "AMR")):
        ch = root / "cache" / f"v4_choice_{dom}.json"
        if not ch.exists():
            continue
        c = json.loads(ch.read_text(encoding="utf-8"))
        kpi["dev_choice"][dom] = c
        # 원고 자리표시자용 중첩 형태: dev_choice:<도메인>:devcf:<설정>:<lang|nolang> (원래 키 "설정:언어"는 ':' 때문에 경로로 못 읽는다)
        if c.get("devcf_speed_error"):
            nest: dict[str, dict[str, float]] = defaultdict(dict)
            for key, val in c["devcf_speed_error"].items():
                var, _, lang = key.partition(":")
                nest[var][lang] = val
            c["devcf"] = dict(nest)
        for v, x in c["dev_success_by_seed"].items():
            rows.append([dname, names.get(v, v), _fmt(float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0), "채택" if c["adopt"].get("best") == v else ("T1 채택" if v == "t1" and c["adopt"].get("t1") else "-")])
        if dom == "driving" and c.get("devcf_speed_error"):
            _write_table(tables / "v4_dev_lang", ["설정:언어", "반사실 개발 세트 속도 오차(m/s)"], [[k, _fmt(v, digits=2)] for k, v in sorted(c["devcf_speed_error"].items())])
    if rows:
        _write_table(tables / "v4_dev", ["도메인", "설정", "개발 세트 성공률(CARE 2%, 시드 3)", "채택"], rows)

    # ---------- 주행 새 테스트 ----------
    G = _group([r for r in _runs(root, "driving__")])
    if (root / "cache" / "expert_driving_test.json").exists():
        ex = expert_reference(cfg, "test")
        cond = {}
        for (ev, var, m, b, lang), rs in G.items():
            if ev == "test" and lang:
                S = _S(rs, ex, False)
                cond[(var, m, b)] = {"S": S, "success": _ms(S[0]), "hazard": (float(_hazard(rs, ex, False).mean()), 0.0), "collision": float(np.mean([r["closed_loop"]["overall"]["collision_rate"] for r in rs])), "n": len(rs)}
        rows = [[var, LABELS.get(m, m), f"{int(round(100 * b))}%", str(c["n"]), _fmt(*c["success"]), _fmt(c["hazard"][0]), _fmt(c["collision"])] for (var, m, b), c in sorted(cond.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1]))]
        _write_table(tables / "v4_driving", ["설정", "방법", "예산", "시드", "성공률", "위험 시나리오 성공률", "충돌률"], rows)
        care, full = cond.get(("v4", "care", 0.02)), cond.get(("v4", "full", 1.0))
        if care:
            kpi["K1"] = {"value": care["success"][0], "sd": care["success"][1], "n_seeds": care["n"]}
            kpi["K3"] = {"value": care["hazard"][0]}
        if care and full:
            kpi["K2"] = {"value": care["success"][0] / full["success"][0], "care": care["success"][0], "full": full["success"][0]}
        comps = {}
        pairs = {
            "care_vs_mix_trigger": (("v4", "care", 0.02), ("v4", "mix_trigger", 0.02)),
            "care_vs_random_shared": (("v4", "care", 0.02), ("v4", "random_shared", 0.02)),
            "v4_vs_v3_care": (("v4", "care", 0.02), ("v3", "care", 0.02)),
            "v4_vs_v3_random_shared": (("v4", "random_shared", 0.02), ("v3", "random_shared", 0.02)),
            "v3_care_vs_random_shared": (("v3", "care", 0.02), ("v3", "random_shared", 0.02)),
        }
        for name, (a, b) in pairs.items():
            if a in cond and b in cond:
                A, B = _pair(cond[a]["S"], cond[b]["S"])
                comps[name] = hierarchical_bootstrap(A, B) | {"seed_t": seed_t(A, B), "n_seeds": len(A)}
        fam = {k: comps[k]["p_le0"] for k in ("care_vs_mix_trigger", "care_vs_random_shared", "v4_vs_v3_care") if k in comps}
        for k, p in holm(fam).items():
            comps[k]["p_holm"] = p
        kpi["comparisons"] = comps
        # 원고 자리표시자용 조건별 값: driving:<설정>:<방법>:<success|success_sd|hazard|collision|n>
        drv: dict[str, dict[str, Any]] = defaultdict(dict)
        for (var, m, b), c in cond.items():
            drv[var][m] = {"success": c["success"][0], "success_sd": c["success"][1], "hazard": c["hazard"][0], "collision": c["collision"], "n": c["n"], "budget": b}
        kpi["driving"] = dict(drv)
        # 원고 자리표시자용 학습 비용(새 테스트 2% 실행의 학습 기록): cost:<v3|v4>:<n_params|samples_per_s|train_time_s>
        # 여러 작업자가 CPU를 공유한 상태의 1스레드 측정이므로 절대값은 참고치다(docs/31 11.4).
        cost: dict[str, dict[str, float]] = {}
        for var in ("v3", "v4"):
            logs = [r.get("train_log") or {} for (ev, v, m, b, lang), rs in G.items() if ev == "test" and v == var and lang and b < 1.0 for r in rs]
            logs = [t for t in logs if t.get("n_params")]
            if logs:
                cost[var] = {
                    "n_params": float(logs[0]["n_params"]),
                    "samples_per_s": float(np.median([t["samples_per_s"] for t in logs])),
                    "train_time_s": float(np.median([t["train_time_s"] for t in logs])),
                    "n_runs": float(len(logs)),
                }
        kpi["cost"] = cost
        if "care_vs_mix_trigger" in comps:
            kpi["K4"] = comps["care_vs_mix_trigger"]

    # ---------- 반사실 언어(K6) ----------
    if (root / "cache" / "expert_driving_cf.json").exists():
        k6 = {}
        rows = []
        for m, b in (("care", 0.02), ("full", 1.0)):
            vals = {}
            for lang in (True, False):
                rs = G.get(("cf", "v4", m, b, lang))
                if not rs:
                    continue
                se = [r["closed_loop"]["overall"].get("speed_error") or np.nan for r in rs]
                sep = []
                for r in rs:
                    hw: dict[tuple, dict[str, float]] = defaultdict(dict)
                    for e in r["closed_loop"]["episodes"]:
                        if e.get("headway_mean") is not None:
                            hw[(e["scenario"], e["seed"])][e["style"]] = e["headway_mean"]
                    d = [v["cautious"] - v["brisk"] for v in hw.values() if "cautious" in v and "brisk" in v]
                    sep.append(float(np.mean(d)) if d else np.nan)
                vals[lang] = {"speed_error": float(np.nanmean(se)), "style_sep": float(np.nanmean(sep)), "n": len(rs)}
                rows.append([LABELS[m], "있음" if lang else "없음", str(len(rs)), _fmt(vals[lang]["speed_error"], digits=2), _fmt(vals[lang]["style_sep"], digits=2)])
            if True in vals and False in vals:
                k6[m] = {"lang": vals[True], "nolang": vals[False], "speed_error_reduction": 1.0 - vals[True]["speed_error"] / vals[False]["speed_error"]}
        _write_table(tables / "v4_language_cf", ["데이터", "언어", "시드", "속도 오차(m/s)", "신중−민첩 추종 간격 차(s)"], rows)
        if "care" in k6:
            kpi["K6"] = k6["care"] | {"by_data": k6}

    # ---------- AMR(K5) ----------
    rc = cfg.for_domain("robot")
    if (root / "cache" / "expert_robot_test.json").exists():
        exr = expert_reference(rc, "test")
        GR = _group(_runs(root, "robot__"))
        rcond = {(var, m, b): _S(rs, exr, True) for (ev, var, m, b, lang), rs in GR.items() if ev == "test"}
        _write_table(tables / "v4_robot", ["설정", "방법", "예산", "시드", "성공률"], [[v, LABELS.get(m, m), f"{int(round(100 * b))}%", str(len(S[1])), _fmt(*_ms(S[0]))] for (v, m, b), S in sorted(rcond.items())])
        if ("v4", "care", 0.02) in rcond and ("v4", "random_shared", 0.02) in rcond:
            A, B = _pair(rcond[("v4", "care", 0.02)], rcond[("v4", "random_shared", 0.02)])
            kpi["K5"] = hierarchical_bootstrap(A, B) | {"care": float(A.mean()), "random_shared": float(B.mean()), "n_seeds": len(A)}
        if ("v4", "full", 1.0) in rcond:
            kpi["robot_full_success"] = float(rcond[("v4", "full", 1.0)][0].mean())
        # 원고 자리표시자용: robot:<설정>:<방법>:<success|success_sd|n>
        rob: dict[str, dict[str, Any]] = defaultdict(dict)
        for (v, m, b), S in rcond.items():
            mu, sd = _ms(S[0])
            rob[v][m] = {"success": mu, "success_sd": sd, "n": len(S[1]), "budget": b}
        kpi["robot"] = dict(rob)

    # ---------- 실영상(K7) ----------
    cr = _runs(root, "comma4__")
    if cr:
        cg: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for r in cr:
            cg[(r["job"]["method"], round(r["job"]["budget"], 2))].append(r)
        _write_table(tables / "v4_comma", ["방법", "예산", "실행", "제동 시작 AUROC", "MAE", "제동 MAE"], [[LABELS.get(m, m), f"{int(round(100 * b))}%", str(len(rs)), _fmt(float(np.mean([r["open_loop"]["brake_onset_auroc"] for r in rs])), float(np.std([r["open_loop"]["brake_onset_auroc"] for r in rs], ddof=1)) if len(rs) > 1 else 0.0), _fmt(float(np.mean([r["open_loop"]["mae"] for r in rs]))), _fmt(float(np.nanmean([r["open_loop"]["mae_braking"] for r in rs])))] for (m, b), rs in sorted(cg.items())])
        f = cg.get(("full", 1.0))
        if f:
            au = [r["open_loop"]["brake_onset_auroc"] for r in f]
            kpi["K7"] = {"value": float(np.mean(au)), "sd": float(np.std(au, ddof=1)) if len(au) > 1 else 0.0, "min": float(np.min(au)), "max": float(np.max(au)), "n_runs": len(au)}
            # 정책 없이 현재 자차 가속도(−a_t)만 쓴 기준선(같은 시험 프레임). K7 판정에는 쓰지 않는 참고값이다(docs/36 2.0b)
            base = [r["open_loop"].get("baseline_neg_accel_auroc") for r in f]
            base = [float(x) for x in base if x is not None]
            if base:
                kpi["K7"]["baseline_neg_accel"] = float(np.mean(base))

    # ---------- KPI 요약(v2·v3·v4) ----------
    v3k = json.loads(V3_KPI_PATH.read_text(encoding="utf-8")) if V3_KPI_PATH.exists() else {}

    def judge(k: str, v: dict[str, Any] | None) -> str:
        if not v:
            return "미측정"
        rule = {"K1": lambda x: x["value"] >= 0.80, "K2": lambda x: x["value"] >= 0.90, "K3": lambda x: x["value"] >= 0.75, "K4": lambda x: x["lo"] > 0, "K5": lambda x: x["lo"] > -0.03, "K6": lambda x: x["speed_error_reduction"] >= 0.30, "K7": lambda x: x["value"] >= 0.65}[k]
        return "달성" if rule(v) else "미달"

    def show(k: str, v: dict[str, Any] | None) -> str:
        if not v:
            return "-"
        if k in ("K4", "K5"):
            return f"{v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]".replace("-", "−")
        if k == "K6":
            return f"{100 * v['speed_error_reduction']:.1f}%".replace("-", "−")
        return f"{v['value']:.3f}"

    rows = [[k, TARGETS[k][0], TARGETS[k][1], show(k, v3k.get(k)) + f" ({judge(k, v3k.get(k))})", show(k, kpi.get(k)), judge(k, kpi.get(k))] for k in TARGETS]
    _write_table(tables / "v4_kpi", ["ID", "지표", "목표", "v3(800000번대 테스트)", "v4(새 테스트 1000000번대)", "판정"], rows)
    kpi["judgement"] = {k: judge(k, kpi.get(k)) for k in TARGETS}
    try:
        _figure(root, kpi, v3k)
    except Exception as exc:  # 그림 실패가 집계를 막지 않게 한다
        logger.warning("v4 그림 생성 실패: %s", exc)
    (root / "summary").mkdir(parents=True, exist_ok=True)
    (root / "summary" / "v4_kpi.json").write_text(json.dumps(kpi, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    logger.info("v4 KPI: %s", kpi["judgement"])
    return kpi


def _figure(root: Path, kpi: dict[str, Any], v3k: dict[str, Any]) -> None:
    """비율형 KPI(K1·K2·K3·K7)와 K6 감소율의 v3·v4 비교 막대 + 목표선."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import koreanize_matplotlib  # noqa: F401
    except ModuleNotFoundError:  # pragma: no cover
        pass
    goals = {"K1": 0.80, "K2": 0.90, "K3": 0.75, "K6": 0.30, "K7": 0.65}

    def val(d: dict[str, Any], k: str) -> float:
        v = d.get(k)
        if not v:
            return float("nan")
        return float(v["speed_error_reduction"] if k == "K6" else v["value"])

    ks = list(goals)
    x = np.arange(len(ks))
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.bar(x - 0.2, [val(v3k, k) for k in ks], 0.38, label="v3", color="#9aa5b1")
    ax.bar(x + 0.2, [val(kpi, k) for k in ks], 0.38, label="v4(새 테스트)", color="#2b6cb0")
    for i, k in enumerate(ks):
        ax.plot([i - 0.42, i + 0.42], [goals[k]] * 2, color="#c53030", ls="--", lw=1.5, label="목표" if i == 0 else None)
    ax.axhline(0, color="k", lw=0.6)
    short = {"K1": "CARE 2% 성공률", "K2": "데이터 효율", "K3": "위험 시나리오", "K6": "언어 오차 감소율", "K7": "실영상 AUROC"}
    ax.set_xticks(x, [f"{k}\n{short[k]}" for k in ks], fontsize=8)
    ax.set_ylabel("값(K6는 감소율)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = root / "summary" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_v4_kpi.png", dpi=160)
    plt.close(fig)


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/exp_130_vla_v4")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    build_v4_report(Path(a.root), a.quick)


if __name__ == "__main__":
    main()
