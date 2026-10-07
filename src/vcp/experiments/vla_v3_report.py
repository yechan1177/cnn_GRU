from __future__ import annotations

"""v3 핵심 지표(K1~K7) 집계(docs/33, 사전 등록 docs/34).

모든 차이 비교는 시드·에피소드 2단계 계층 부트스트랩(같은 시드 번호 대응)으로 한다.
결과: `<root>/summary/v3_kpi.json`, `tables/v3_kpi.md`, `tables/v3_*.md`, `figures/fig_v3_*.png`
"""

import json
import logging
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .vla_curation_suite import MIN_REF_DISTANCE, SUCCESS_PROGRESS
from .vla_report import _fmt, _write_table, hierarchical_bootstrap, holm, seed_t
from .vla_v3_suite import V3Config, expert_reference

logger = logging.getLogger(__name__)
# 표에 쓰는 방법 이름(원고와 같은 한국어 표기)
V3_LABELS = {"full": "전체(100%)", "random_shared": "무작위(공유 저장소)", "care": "CARE", "mix_trigger": "저장소 + 감속 트리거", "mix_oracle": "저장소 + 오라클*", "trigger_only": "감속 트리거 단독", "action_trigger": "감속 트리거 단독"}

TARGETS = {
    "K1": ("CARE 2% 폐루프 성공률", ">= 0.80"),
    "K2": ("데이터 효율(CARE 2% / 전체)", ">= 0.90"),
    "K3": ("위험 시나리오 성공률(CARE 2%)", ">= 0.75"),
    "K4": ("점수기 고유 기여(CARE − 트리거 혼합)", "CI 하한 > 0"),
    "K5": ("AMR CARE − 무작위(공유 저장소)", "CI 하한 > −0.03"),
    "K6": ("언어가 줄이는 속도 추종 오차(반사실)", ">= 30%"),
    "K7": ("실영상 제동 시작 예측 AUROC(전체 데이터)", ">= 0.65"),
}


def episode_success(episodes: list[dict[str, Any]], expert: dict[str, Any], moving_only: bool = False) -> np.ndarray:
    """v2 `episode_success`와 같은 정의이되, 전문가 기준 거리를 (시나리오, 시드, 스타일)로 찾는다.

    반사실 세트는 같은 (시나리오, 시드)를 세 스타일로 평가하므로 (시나리오, 시드)만으로 찾으면
    마지막 스타일(민첩)의 거리만 남는다. 스타일까지 키에 넣으면 일반 세트에서는 v2와 같은 결과다.
    """

    ref = {(e["scenario"], e["seed"], e.get("style")): e["distance"] for e in expert["episodes"]}
    key = "collision_moving" if moving_only else "collision"
    out = []
    for e in episodes:
        d_ref = ref[(e["scenario"], e["seed"], e.get("style"))]  # 평가 사양이 다르면 KeyError
        progress_ok = True if d_ref < MIN_REF_DISTANCE else e["distance"] >= SUCCESS_PROGRESS * d_ref
        out.append((not e[key]) and progress_ok)
    return np.asarray(out, dtype=bool)


def _load_runs(root: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "runs").glob("*.json")) if not p.name.startswith("comma")]


def _group(runs: list[dict[str, Any]]) -> dict[tuple, list[dict[str, Any]]]:
    g: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        j = r["job"]
        k = (j["domain"], j["eval"], j["method"], round(j["budget"], 2), j["steps"], j.get("use_language", True), j.get("use_features", True), j.get("lam"), j.get("reservoir"))
        g[k].append(r)
    for v in g.values():
        v.sort(key=lambda r: r["job"]["seed"])
    return g


def _succ(rs: list[dict[str, Any]], ex: dict[str, Any], moving: bool) -> tuple[np.ndarray, list[int]]:
    return np.stack([episode_success(r["closed_loop"]["episodes"], ex, moving) for r in rs]).astype(float), [r["job"]["seed"] for r in rs]


def _hazard_succ(rs: list[dict[str, Any]], ex: dict[str, Any], moving: bool) -> np.ndarray:
    out = []
    for r in rs:
        eps = r["closed_loop"]["episodes"]
        s = episode_success(eps, ex, moving)
        m = np.array([bool(e["hazard"]) for e in eps])
        out.append(float(s[m].mean()))
    return np.asarray(out)


def _pair(a: tuple[np.ndarray, list[int]], b: tuple[np.ndarray, list[int]]) -> tuple[np.ndarray, np.ndarray]:
    common = sorted(set(a[1]) & set(b[1]))
    return a[0][[a[1].index(s) for s in common]], b[0][[b[1].index(s) for s in common]]


def build_v3_report(root: Path, steps: int | None = None, quick: bool = False) -> dict[str, Any]:
    cfg = V3Config(root=root, quick=quick)
    if quick:
        cfg = replace(cfg, pool_episodes=24, testpool_episodes=6, val_per_cell=1, test_per_cell=1, cf_per_cell=1, steps=60, batch=32)
    if steps is not None:  # 스모크(--quick) 결과 집계용
        cfg.steps = int(steps)
    runs = _load_runs(root)
    G = _group(runs)
    steps = cfg.steps
    kpi: dict[str, Any] = {}
    tables = root / "summary" / "tables"

    def key(dom: str, ev: str, m: str, b: float, lang: bool = True, feat: bool = True) -> tuple:
        return (dom, ev, m, round(b, 2), steps, lang, feat, None, None)

    # ---------- 주행 테스트 ----------
    ex = expert_reference(cfg, "test") if (root / "cache" / "expert_driving_test.json").exists() else None
    cond: dict[tuple[str, float], dict[str, Any]] = {}
    if ex is not None:
        for (dom, ev, m, b, st, lang, feat, lam, res), rs in G.items():
            if dom == "driving" and ev == "test" and st == steps and lang and feat and lam is None:
                S = _succ(rs, ex, False)
                hz = _hazard_succ(rs, ex, False)
                cl = [r["closed_loop"]["overall"] for r in rs]
                cond[(m, b)] = {
                    "S": S,
                    "success": (float(S[0].mean(1).mean()), float(S[0].mean(1).std(ddof=1)) if len(rs) > 1 else 0.0),
                    "hazard": (float(hz.mean()), float(hz.std(ddof=1)) if len(rs) > 1 else 0.0),
                    "collision": (float(np.mean([c["collision_rate"] for c in cl])), 0.0),
                    "speed_error": (float(np.nanmean([c.get("speed_error") or np.nan for c in cl])), 0.0),
                    "n": len(rs),
                }
        rows = []
        labels = V3_LABELS
        for (m, b), c in sorted(cond.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            rows.append([labels.get(m, m), f"{int(round(b * 100))}%", str(c["n"]), _fmt(*c["success"]), _fmt(*c["hazard"]), _fmt(c["collision"][0]), _fmt(c["speed_error"][0], digits=2)])
        _write_table(tables / "v3_driving", ["방법", "예산", "시드", "성공률", "위험 시나리오 성공률", "충돌률", "속도 오차(m/s)"], rows)
        # 보조(탐색적): 시나리오별 성공률(2%·100%)과 선택 데이터 구성
        scen = sorted({e["scenario"] for e in ex["episodes"]})
        srows, crows = [], []
        for m, b in [("full", 1.0)] + [(m, 0.02) for m in ("care", "mix_trigger", "mix_oracle", "random_shared", "trigger_only")]:
            rs = G.get(("driving", "test", m, b, steps, True, True, None, None))
            if not rs:
                continue
            per: dict[str, list[float]] = defaultdict(list)
            for r in rs:
                eps = r["closed_loop"]["episodes"]
                sc = episode_success(eps, ex, False)
                for e, ok in zip(eps, sc):
                    per[e["scenario"]].append(float(ok))
            srows.append([labels.get(m, m), f"{int(round(b * 100))}%"] + [_fmt(float(np.mean(per[c]))) for c in scen])
            sel = [r["selection"] for r in rs]
            hist = np.mean([np.asarray(x["class_hist"], float) / max(1, sum(x["class_hist"])) for x in sel], axis=0)
            crows.append([labels.get(m, m), f"{int(round(b * 100))}%", _fmt(float(hist[2] + hist[3])), _fmt(float(np.mean([x["hazard_event_recall"] for x in sel]))), _fmt(float(np.mean([x["label_entropy_norm"] for x in sel]))), _fmt(float(np.mean([x["group_coverage"] for x in sel])))])
        _write_table(tables / "v3_driving_scenario", ["방법", "예산"] + scen, srows)
        _write_table(tables / "v3_selection", ["방법", "예산", "위험 프레임 비중", "위험 이벤트 회수율", "라벨 엔트로피(정규화)", "에피소드 포괄률"], crows)
        care, full = cond.get(("care", 0.02)), cond.get(("full", 1.0))
        if care:
            kpi["K1"] = {"value": care["success"][0], "sd": care["success"][1], "n_seeds": care["n"]}
            kpi["K3"] = {"value": care["hazard"][0], "sd": care["hazard"][1]}
        if care and full:
            kpi["K2"] = {"value": care["success"][0] / full["success"][0], "care": care["success"][0], "full": full["success"][0]}
        comps: dict[str, Any] = {}
        for other in ("mix_trigger", "random_shared", "mix_oracle", "trigger_only"):
            o = cond.get((other, 0.02))
            if care and o:
                A, B = _pair(care["S"], o["S"])
                comps[f"care_vs_{other}_b0.02"] = hierarchical_bootstrap(A, B) | {"seed_t": seed_t(A, B), "n_seeds": len(A)}
        for b in (0.01, 0.05):
            c_, r_ = cond.get(("care", b)), cond.get(("random_shared", b))
            if c_ and r_:
                A, B = _pair(c_["S"], r_["S"])
                comps[f"care_vs_random_shared_b{b:.2f}"] = hierarchical_bootstrap(A, B) | {"n_seeds": len(A)}
        mt, rs_ = cond.get(("mix_trigger", 0.02)), cond.get(("random_shared", 0.02))
        if mt and rs_:
            A, B = _pair(mt["S"], rs_["S"])
            comps["mix_trigger_vs_random_shared_b0.02"] = hierarchical_bootstrap(A, B) | {"n_seeds": len(A)}
        fam = {k: v["p_le0"] for k, v in comps.items() if k in ("care_vs_mix_trigger_b0.02", "care_vs_random_shared_b0.02", "care_vs_mix_oracle_b0.02")}
        for k, p in holm(fam).items():
            comps[k]["p_holm"] = p
        kpi["comparisons"] = comps
        if "care_vs_mix_trigger_b0.02" in comps:
            kpi["K4"] = comps["care_vs_mix_trigger_b0.02"]

    # ---------- AMR ----------
    rc = cfg.for_domain("robot")
    if (rc.cache / "expert_robot_test.json").exists():
        exr = expert_reference(rc, "test")
        rcond = {}
        for (dom, ev, m, b, st, lang, feat, lam, res), rs in G.items():
            if dom == "robot" and ev == "test" and lam is None and feat and lang:
                rcond[(m, b)] = _succ(rs, exr, True)
        rows = []
        for (m, b), S in sorted(rcond.items()):
            rows.append([V3_LABELS.get(m, m), f"{int(round(b * 100))}%", str(len(S[1])), _fmt(float(S[0].mean(1).mean()), float(S[0].mean(1).std(ddof=1)) if len(S[1]) > 1 else 0.0)])
        _write_table(tables / "v3_robot", ["방법", "예산", "시드", "성공률"], rows)
        if ("care", 0.02) in rcond and ("random_shared", 0.02) in rcond:
            A, B = _pair(rcond[("care", 0.02)], rcond[("random_shared", 0.02)])
            kpi["K5"] = hierarchical_bootstrap(A, B) | {"care": float(A.mean()), "random_shared": float(B.mean()), "n_seeds": len(A)}
        if ("full", 1.0) in rcond:
            kpi["robot_full_success"] = float(rcond[("full", 1.0)][0].mean())
        kpi["robot_expert_success"] = float(episode_success(exr["episodes"], exr, True).mean())

    # ---------- 반사실 언어(K6) ----------
    if (root / "cache" / "expert_driving_cf.json").exists():
        excf = expert_reference(cfg, "cf")
        lang_rows = []
        k6 = {}
        for m, b in (("full", 1.0), ("care", 0.02)):
            vals = {}
            for lang in (True, False):
                rs = G.get(("driving", "cf", m, b, steps, lang, True, None, None))
                if not rs:
                    continue
                se = [r["closed_loop"]["overall"].get("speed_error") or np.nan for r in rs]
                S = _succ(rs, excf, False)
                # 스타일 구분도: 같은 (시나리오, 시드)에서 신중 − 민첩 평균 추종 간격 차이
                sep = []
                for r in rs:
                    hw: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
                    for e in r["closed_loop"]["episodes"]:
                        if e.get("headway_mean") is not None:
                            hw[(e["scenario"], e["seed"])][e["style"]] = e["headway_mean"]
                    d = [v["cautious"] - v["brisk"] for v in hw.values() if "cautious" in v and "brisk" in v]
                    sep.append(float(np.mean(d)) if d else np.nan)
                vals[lang] = {"speed_error": float(np.nanmean(se)), "success": float(S[0].mean()), "style_sep": float(np.nanmean(sep)), "n": len(rs)}
                lang_rows.append([V3_LABELS.get(m, m), "있음" if lang else "없음", str(len(rs)), _fmt(vals[lang]["success"]), _fmt(vals[lang]["speed_error"], digits=2), _fmt(vals[lang]["style_sep"], digits=2)])
            if True in vals and False in vals:
                k6[m] = {"lang": vals[True], "nolang": vals[False], "speed_error_reduction": 1.0 - vals[True]["speed_error"] / vals[False]["speed_error"]}
        ex_sep = []
        hw = defaultdict(dict)
        for e in excf["episodes"]:
            if e.get("headway_mean") is not None:
                hw[(e["scenario"], e["seed"])][e["style"]] = e["headway_mean"]
        ex_sep = [v["cautious"] - v["brisk"] for v in hw.values() if "cautious" in v and "brisk" in v]
        lang_rows.append(["전문가", "-", "-", "1.000", _fmt(excf["overall"].get("speed_error") or np.nan, digits=2), _fmt(float(np.mean(ex_sep)) if ex_sep else np.nan, digits=2)])
        _write_table(tables / "v3_language_cf", ["데이터", "언어", "시드", "성공률", "속도 오차(m/s)", "신중−민첩 추종 간격 차(s)"], lang_rows)
        if k6:
            kpi["K6"] = k6.get("care", k6.get("full")) | {"by_data": k6}

    # ---------- 실영상(K7) ----------
    cruns = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "runs").glob("comma3__*.json"))]
    if cruns:
        cg: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for r in cruns:
            cg[(r["job"]["method"], round(r["job"]["budget"], 2), bool(r["job"].get("use_features", False)))].append(r)
        rows = []
        for (m, b, f), rs in sorted(cg.items()):
            au = [r["open_loop"]["brake_onset_auroc"] for r in rs]
            mae = [r["open_loop"]["mae"] for r in rs]
            mb = [r["open_loop"]["mae_braking"] for r in rs]
            rows.append([V3_LABELS.get(m, m), f"{int(round(b * 100))}%", "O" if f else "X", str(len(rs)), _fmt(float(np.mean(au)), float(np.std(au, ddof=1))), _fmt(float(np.mean(mae))), _fmt(float(np.mean(mb)))])
        _write_table(tables / "v3_comma", ["방법", "예산", "특징 토큰", "실행", "제동 시작 AUROC", "MAE", "제동 MAE"], rows)
        full_f = cg.get(("full", 1.0, True))
        full_n = cg.get(("full", 1.0, False))
        if full_f:
            kpi["K7"] = {"value": float(np.mean([r["open_loop"]["brake_onset_auroc"] for r in full_f])), "sd": float(np.std([r["open_loop"]["brake_onset_auroc"] for r in full_f], ddof=1)), "nofeat": float(np.mean([r["open_loop"]["brake_onset_auroc"] for r in full_n])) if full_n else None}

    # ---------- KPI 요약표 ----------
    def judge(k: str) -> str:
        v = kpi.get(k)
        if v is None:
            return "미측정"
        if k == "K1":
            return "달성" if v["value"] >= 0.80 else "미달"
        if k == "K2":
            return "달성" if v["value"] >= 0.90 else "미달"
        if k == "K3":
            return "달성" if v["value"] >= 0.75 else "미달"
        if k == "K4":
            return "달성" if v["lo"] > 0 else "미달"
        if k == "K5":
            return "달성" if v["lo"] > -0.03 else "미달"
        if k == "K6":
            return "달성" if v["speed_error_reduction"] >= 0.30 else "미달"
        if k == "K7":
            return "달성" if v["value"] >= 0.65 else "미달"
        return "-"

    def show(k: str) -> str:
        v = kpi.get(k)
        if v is None:
            return "-"
        if k in ("K4", "K5"):
            return f"{v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]".replace("-", "−")
        if k == "K6":
            return f"{100 * v['speed_error_reduction']:.1f}% (오차 {v['nolang']['speed_error']:.2f}→{v['lang']['speed_error']:.2f} m/s)"
        if k == "K2":
            return f"{v['value']:.3f} ({v['care']:.3f}/{v['full']:.3f})"
        return f"{v['value']:.3f}" + (f" ± {v['sd']:.3f}" if "sd" in v and v.get("sd") is not None else "")

    v2 = {"K1": "0.751(시드 3) / 0.736(확증)", "K2": "0.84", "K3": "0.671", "K4": "+0.066 [−0.010, +0.140]", "K5": "−0.032 [−0.194, +0.125]", "K6": "측정 불가(누출)", "K7": "0.523"}
    rows = [[k, TARGETS[k][0], TARGETS[k][1], v2[k], show(k), judge(k)] for k in TARGETS]
    _write_table(tables / "v3_kpi", ["ID", "지표", "목표", "v2", "v3", "판정"], rows)
    kpi["judgement"] = {k: judge(k) for k in TARGETS}
    try:
        _figures(root, kpi, cond if ex is not None else {})
    except Exception as exc:  # 그림 실패가 집계를 막지 않게 한다
        logger.warning("v3 그림 생성 실패: %s", exc)
    (root / "summary").mkdir(parents=True, exist_ok=True)
    (root / "summary" / "v3_kpi.json").write_text(json.dumps(kpi, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    logger.info("v3 KPI: %s", kpi["judgement"])
    return kpi


V2_POINTS = {"K1": 0.751, "K2": 0.84, "K3": 0.671, "K7": 0.523}
GOALS = {"K1": 0.80, "K2": 0.90, "K3": 0.75, "K7": 0.65}


def _figures(root: Path, kpi: dict[str, Any], cond: dict[tuple[str, float], dict[str, Any]]) -> None:
    """그림 2종: (a) 비율형 KPI의 v2·v3·목표, (b) 예산별 주행 성공률."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import koreanize_matplotlib  # noqa: F401  (한글 글꼴)
    except ModuleNotFoundError:  # pragma: no cover
        pass
    out = root / "summary" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    ks = [k for k in ("K1", "K2", "K3", "K7") if k in kpi]
    if ks:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        x = np.arange(len(ks))
        ax.bar(x - 0.2, [V2_POINTS[k] for k in ks], 0.38, label="v2", color="#9aa5b1")
        ax.bar(x + 0.2, [kpi[k]["value"] for k in ks], 0.38, label="v3", color="#2b6cb0")
        for i, k in enumerate(ks):
            ax.plot([i - 0.42, i + 0.42], [GOALS[k]] * 2, color="#c53030", lw=1.6, ls="--", label="목표" if i == 0 else None)
        ax.set_xticks(x, [f"{k}\n{TARGETS[k][0][:12]}" for k in ks], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.set_ylabel("값")
        fig.tight_layout()
        fig.savefig(out / "fig_v3_kpi.png", dpi=160)
        plt.close(fig)
    if cond:
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        for m, lab, c in (("care", "CARE", "#2b6cb0"), ("random_shared", "무작위(공유 저장소)", "#718096")):
            pts = sorted((b, v["success"][0], v["success"][1]) for (mm, b), v in cond.items() if mm == m)
            if pts:
                b_, mu, sd = map(np.asarray, zip(*pts))
                ax.errorbar(100 * b_, mu, yerr=sd, marker="o", capsize=3, label=lab, color=c)
        if ("full", 1.0) in cond:
            ax.axhline(cond[("full", 1.0)]["success"][0], color="k", ls=":", label="전체(100%)")
        ax.axhline(0.80, color="#c53030", ls="--", lw=1, label="K1 목표")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5], ["1%", "2%", "5%"])
        ax.minorticks_off()  # 로그 축 보조 눈금 라벨(예: 3×10^0)이 찍히지 않게 한다
        ax.set_xlabel("저장 예산")
        ax.set_ylabel("폐루프 성공률")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "fig_v3_budget.png", dpi=160)
        plt.close(fig)


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/exp_120_vla_v3")
    ap.add_argument("--steps", type=int, default=None, help="집계할 학습 단계 수(기본: V3Config.steps)")
    ap.add_argument("--quick", action="store_true", help="스모크 설정(크기·단계 수)으로 집계")
    a = ap.parse_args()
    build_v3_report(Path(a.root), a.steps, a.quick)


if __name__ == "__main__":
    main()
