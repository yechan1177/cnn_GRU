from __future__ import annotations

"""세 번째 테스트 세트 확증 집계(사전 등록 docs/39, 3차 심사 A2~A7 대응).

평가 세트: 주행 test3(1200000번대)·cf3(1250000번대), AMR test3(1300000번대). 정책·선별 설정은 v4(docs/37)와 같다.
분석
- 재현: K1~K6(판정 규칙 docs/34·37과 같음). K1·K2·K3·K6는 부트스트랩 CI와 목표 이상 비율을 함께 보고한다.
- 확증 가설(Holm 군 A, 단측 α=0.05): H-v3, H-a3, H-a3n, H-P7, H-L
- 동등성(H-K4e): 점수기 출력을 학습에 쓰지 않는 v4(v4_noaux)에서 CARE − 트리거 혼합의 90% CI가 (−0.03, +0.03) 안(TOST)
- 블록별 제거(A7): Δ = v4 − 블록 제거, 시드 5개, Holm 군 B
- AMR(K5, A4): 시드 10개, v3 대조
- 실영상 P5(A5): 같은 fold 재사용이므로 탐색적
결과: `<root>/summary/third_kpi.json`, `tables/third_*.md`
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .vla_report import _fmt, _write_table, hierarchical_bootstrap, holm
from .vla_v3_report import episode_success
from .vla_v4_suite import V4Config, expert_reference

logger = logging.getLogger(__name__)
LABELS = {"full": "전체(100%)", "random_shared": "무작위(공유 저장소)", "care": "CARE", "mix_trigger": "저장소 + 감속 트리거"}
VAR_LABELS = {
    "v4": "v4(전부+P7)",
    "v4_noaux": "v4, P3·P4 제외(점수기 출력 학습 미사용)",
    "v3": "v3 정책",
    "all_p6": "전부+P6(P7 제외)",
    "loo_p2": "P2 제외",
    "loo_p3": "P3 제외",
    "loo_t1": "T1 제외",
    "loo_p4": "P4 제외",
    "loo_p67": "P6·P7 제외",
    "p67_nolangemb": "P6·P7만(학습 언어 경로 없음)",
}


def _runs(root: Path, prefix: str, ev: str) -> dict[tuple, list[dict[str, Any]]]:
    g: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for p in sorted((root / "runs").glob(f"{prefix}__{ev}__*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        j = r["job"]
        g[(j["variant"], j["method"], round(j["budget"], 2), j.get("use_language", True))].append(r)
    for v in g.values():
        v.sort(key=lambda r: r["job"]["seed"])
    return g


def _S(rs: list[dict[str, Any]], ex: dict[str, Any], moving: bool) -> tuple[np.ndarray, list[int]]:
    return np.stack([episode_success(r["closed_loop"]["episodes"], ex, moving) for r in rs]).astype(float), [r["job"]["seed"] for r in rs]


def _pair(a: tuple[np.ndarray, list[int]], b: tuple[np.ndarray, list[int]]) -> tuple[np.ndarray, np.ndarray]:
    common = sorted(set(a[1]) & set(b[1]))
    return a[0][[a[1].index(s) for s in common]], b[0][[b[1].index(s) for s in common]]


def _boot(X: np.ndarray, n_boot: int = 10000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_s, n_e = X.shape
    out = np.empty(n_boot)
    for i in range(n_boot):
        out[i] = X[np.ix_(rng.integers(0, n_s, n_s), rng.integers(0, n_e, n_e))].mean()
    return out


def _paired_boot(A: np.ndarray, B: np.ndarray, n_boot: int = 10000, seed: int = 0) -> np.ndarray:
    """같은 시드·에피소드 대응 차이의 2단계 부트스트랩 분포."""

    rng = np.random.default_rng(seed)
    D = A - B
    n_s, n_e = D.shape
    out = np.empty(n_boot)
    for i in range(n_boot):
        out[i] = D[np.ix_(rng.integers(0, n_s, n_s), rng.integers(0, n_e, n_e))].mean()
    return out


def _seed_boot(x: np.ndarray, n_boot: int = 10000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])


def _ci(bs: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))


def _se_by_seed(rs: list[dict[str, Any]]) -> dict[int, float]:
    return {r["job"]["seed"]: float(r["closed_loop"]["overall"].get("speed_error") or np.nan) for r in rs}


def _ep_se(rs: list[dict[str, Any]]) -> tuple[np.ndarray, list[int]]:
    """실행별 에피소드 속도 오차 [시드, 에피소드](값 없는 에피소드는 NaN)."""

    return np.array([[np.nan if e.get("speed_error") is None else float(e["speed_error"]) for e in r["closed_loop"]["episodes"]] for r in rs]), [r["job"]["seed"] for r in rs]


def _hier_ratio(L: np.ndarray, N: np.ndarray, n_boot: int = 10000, seed: int = 0) -> np.ndarray:
    """1 − mean(L)/mean(N)의 시드·에피소드 2단계 부트스트랩(같은 시드·에피소드 대응). 4차 심사 N2 보조 분석."""

    rng = np.random.default_rng(seed)
    n_s, n_e = L.shape
    out = np.empty(n_boot)
    for i in range(n_boot):
        si, ei = rng.integers(0, n_s, n_s), rng.integers(0, n_e, n_e)
        out[i] = 1 - np.nanmean(L[np.ix_(si, ei)]) / np.nanmean(N[np.ix_(si, ei)])
    return out


def _hier_diff(A: np.ndarray, B: np.ndarray, n_boot: int = 10000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_s, n_e = A.shape
    out = np.empty(n_boot)
    for i in range(n_boot):
        si, ei = rng.integers(0, n_s, n_s), rng.integers(0, n_e, n_e)
        out[i] = np.nanmean(A[np.ix_(si, ei)]) - np.nanmean(B[np.ix_(si, ei)])
    return out


def build_third_report(root: Path, quick: bool = False, test_ev: str = "test3", cf_ev: str = "cf3", tag: str = "third") -> dict[str, Any]:
    """기본 인자(test3·cf3·third)는 사전 등록(docs/39)으로 고정한 분석과 같다. 네 번째 평가 세트(docs/40)는 인자만 바꿔 같은 분석을 쓴다."""

    from dataclasses import replace

    cfg = V4Config(root=root, quick=quick)
    if quick:
        cfg = replace(cfg, dev_per_cell=1, devcf_per_cell=1, test_per_cell=1, cf_per_cell=1, steps=60, batch=32)
    tables = root / "summary" / "tables"
    out: dict[str, Any] = {"kpi": {}, "tests": {}, "loo": {}, "k6": {}, "robot": {}, "comma_p5": {}}

    # ---------------- 주행 test3 ----------------
    if (root / "cache" / f"expert_driving_{test_ev}.json").exists():
        ex = expert_reference(cfg, test_ev)
        G = _runs(root, "driving", test_ev)
        C = {(v, m, b): _S(rs, ex, False) for (v, m, b, lang), rs in G.items() if lang}
        hz = np.array([bool(e["hazard"]) for e in ex["episodes"]])
        rows = []
        for (v, m, b), S in sorted(C.items(), key=lambda kv: (list(VAR_LABELS).index(kv[0][0]) if kv[0][0] in VAR_LABELS else 99, kv[0][2], kv[0][1])):
            sm = S[0].mean(1)
            rows.append([VAR_LABELS.get(v, v), LABELS.get(m, m), f"{int(round(100 * b))}%", str(len(sm)), _fmt(float(sm.mean()), float(sm.std(ddof=1)) if len(sm) > 1 else 0.0), _fmt(float(S[0][:, hz].mean()))])
        _write_table(tables / f"{tag}_driving", ["설정", "방법", "예산", "시드", "성공률", "위험 시나리오 성공률"], rows)
        out["expert_success"] = float(np.mean([not e["collision"] for e in ex["episodes"]]))

        care, full = C.get(("v4", "care", 0.02)), C.get(("v4", "full", 1.0))
        if care:
            b1, b3 = _boot(care[0]), _boot(care[0][:, hz])
            out["kpi"]["K1"] = {"value": float(care[0].mean()), "lo": _ci(b1)[0], "hi": _ci(b1)[1], "p_ge_goal": float((b1 >= 0.80).mean()), "n_seeds": len(care[1])}
            out["kpi"]["K3"] = {"value": float(care[0][:, hz].mean()), "sd": float(care[0][:, hz].mean(1).std(ddof=1)), "lo": _ci(b3)[0], "hi": _ci(b3)[1], "p_ge_goal": float((b3 >= 0.75).mean())}
            if full:
                bf = _boot(full[0], seed=1)
                r = b1 / bf
                out["kpi"]["K2"] = {"value": float(care[0].mean() / full[0].mean()), "care": float(care[0].mean()), "full": float(full[0].mean()), "lo": _ci(r)[0], "hi": _ci(r)[1], "p_ge_goal": float((r >= 0.90).mean())}

        def test(name: str, a: tuple, b: tuple) -> None:
            if a in C and b in C:
                A, B = _pair(C[a], C[b])
                hb = hierarchical_bootstrap(A, B)
                out["tests"][name] = hb | {"n_seeds": len(A), "a": float(A.mean()), "b": float(B.mean())}

        test("K4", ("v4", "care", 0.02), ("v4", "mix_trigger", 0.02))
        test("H-v3", ("v4", "care", 0.02), ("v3", "care", 0.02))
        test("H-a3", ("v4", "care", 0.02), ("v4", "random_shared", 0.02))
        test("H-a3n", ("v4_noaux", "care", 0.02), ("v4_noaux", "random_shared", 0.02))
        test("H-P7", ("v4", "care", 0.02), ("all_p6", "care", 0.02))
        test("mix_vs_random_v4", ("v4", "mix_trigger", 0.02), ("v4", "random_shared", 0.02))
        test("mix_vs_random_noaux", ("v4_noaux", "mix_trigger", 0.02), ("v4_noaux", "random_shared", 0.02))
        test("v4_vs_noaux_care", ("v4", "care", 0.02), ("v4_noaux", "care", 0.02))
        # 동등성(TOST): v4_noaux CARE − 트리거 혼합의 90% CI가 (−0.03, +0.03) 안
        a, b = ("v4_noaux", "care", 0.02), ("v4_noaux", "mix_trigger", 0.02)
        if a in C and b in C:
            A, B = _pair(C[a], C[b])
            bs = _paired_boot(A, B)
            lo90, hi90 = _ci(bs, 0.90)
            out["tests"]["H-K4e"] = {"diff": float((A - B).mean()), "lo90": lo90, "hi90": hi90, "margin": 0.03, "equivalent": bool(lo90 > -0.03 and hi90 < 0.03), "p_lower": float((bs <= -0.03).mean()), "p_upper": float((bs >= 0.03).mean()), "n_seeds": len(A)}
        # 블록별 제거(A7): Δ = v4 − 제거, Holm 군 B
        for v in ("loo_p2", "loo_p3", "loo_t1", "loo_p4", "loo_p67"):
            if ("v4", "care", 0.02) in C and (v, "care", 0.02) in C:
                A, B = _pair(C[("v4", "care", 0.02)], C[(v, "care", 0.02)])
                out["loo"][v] = hierarchical_bootstrap(A, B) | {"n_seeds": len(A), "v4": float(A.mean()), "removed": float(B.mean())}
        for k, p in holm({k: v["p_le0"] for k, v in out["loo"].items()}).items():
            out["loo"][k]["p_holm"] = p
        _write_table(tables / f"{tag}_loo", ["제거한 블록", "시드", "v4 성공률", "제거 시 성공률", "v4 − 제거 [95% CI]", "Holm p"], [[VAR_LABELS[k], str(v["n_seeds"]), _fmt(v["v4"]), _fmt(v["removed"]), f"{v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]".replace("-", "−"), f"{v['p_holm']:.3f}"] for k, v in out["loo"].items()])

    # ---------------- 반사실 cf3(K6 재현·분해) ----------------
    if (root / "cache" / f"expert_driving_{cf_ev}.json").exists():
        H = _runs(root, "driving", cf_ev)
        se = {}
        for key, name in ((("v4", "care", 0.02, True), "v4_lang"), (("v4", "care", 0.02, False), "v4_nolang"), (("all_p6", "care", 0.02, True), "p6_lang"), (("all_p6", "care", 0.02, False), "p6_nolang"), (("p67_nolangemb", "care", 0.02, True), "p67_only"), (("ctrl", "none", 0.0, True), "ctrl")):
            if key in H:
                se[name] = _se_by_seed(H[key])
        k6: dict[str, Any] = {}
        if "v4_lang" in se and "v4_nolang" in se:
            common = sorted(set(se["v4_lang"]) & set(se["v4_nolang"]))
            L, N = np.array([se["v4_lang"][c] for c in common]), np.array([se["v4_nolang"][c] for c in common])
            rng = np.random.default_rng(0)
            bs = np.array([1 - L[ii].mean() / N[ii].mean() for ii in (rng.integers(0, len(common), len(common)) for _ in range(10000))])
            k6["K6"] = {"speed_error_reduction": float(1 - L.mean() / N.mean()), "lo": _ci(bs)[0], "hi": _ci(bs)[1], "p_ge_goal": float((bs >= 0.30).mean()), "by_seed": {str(c): float(1 - a / b) for c, a, b in zip(common, L, N)}, "n_seeds_reduced": int((L < N).sum()), "lang": float(L.mean()), "nolang": float(N.mean()), "n_seeds": len(common)}
        if "v4_lang" in se and "p67_only" in se:
            common = sorted(set(se["v4_lang"]) & set(se["p67_only"]))
            d = np.array([se["v4_lang"][c] - se["p67_only"][c] for c in common])
            bs = _seed_boot(d)
            # H-L: 학습 언어 경로가 오차를 줄인다(d < 0). 단측 p = P(부트스트랩 평균 ≥ 0)
            k6["H-L"] = {"diff": float(d.mean()), "lo": _ci(bs)[0], "hi": _ci(bs)[1], "p_le0": float((bs >= 0).mean()), "n_seeds": len(common)}
        if "p6_lang" in se and "p6_nolang" in se:
            common = sorted(set(se["p6_lang"]) & set(se["p6_nolang"]))
            L, N = np.array([se["p6_lang"][c] for c in common]), np.array([se["p6_nolang"][c] for c in common])
            k6["p6_reduction"] = float(1 - L.mean() / N.mean())
        k6["mean_speed_error"] = {k: float(np.nanmean(list(v.values()))) for k, v in se.items()}
        # 보조(4차 심사 N2): 시드·에피소드 계층 부트스트랩 구간. 사전 등록 판정에는 쓰지 않는다
        for a_key, b_key, name in ((("v4", "care", 0.02, True), ("v4", "care", 0.02, False), "K6_hier"), (("v4", "care", 0.02, True), ("p67_nolangemb", "care", 0.02, True), "H-L_hier")):
            if a_key in H and b_key in H:
                A, B = _pair(_ep_se(H[a_key]), _ep_se(H[b_key]))
                bs = _hier_ratio(A, B) if name == "K6_hier" else _hier_diff(A, B)
                k6[name] = {"lo": _ci(bs)[0], "hi": _ci(bs)[1], "n_seeds": int(A.shape[0])} | ({"p_ge_goal": float((bs >= 0.30).mean())} if name == "K6_hier" else {"p_le0": float((bs >= 0).mean())})
        out["k6"] = k6
        lab = {"v4_lang": "v4, 언어 있음", "v4_nolang": "v4, 언어 없음(P6·P7도 꺼짐)", "p6_lang": "전부+P6(P7 없음), 언어 있음", "p6_nolang": "전부+P6, 언어 없음", "p67_only": "P6·P7만(학습 언어 경로 없음)", "ctrl": "제어기 단독(학습 없음)"}
        _write_table(tables / f"{tag}_k6", [f"조건(CARE 2%, {cf_ev})", "실행", "속도 오차(m/s)"], [[lab[k], str(len(v)), _fmt(float(np.nanmean(list(v.values()))), digits=2)] for k, v in se.items()])

    # Holm 군 A
    fam = {k: out["tests"][k]["p_le0"] for k in ("H-v3", "H-a3", "H-a3n", "H-P7") if k in out["tests"]}
    if "H-L" in out["k6"]:
        fam["H-L"] = out["k6"]["H-L"]["p_le0"]
    for k, p in holm(fam).items():
        (out["k6"]["H-L"] if k == "H-L" else out["tests"][k])["p_holm"] = p
    trows = []
    for k in ("H-v3", "H-a3", "H-a3n", "H-P7", "K4", "mix_vs_random_v4", "mix_vs_random_noaux", "v4_vs_noaux_care"):
        if k in out["tests"]:
            t = out["tests"][k]
            trows.append([k, str(t["n_seeds"]), _fmt(t["a"]), _fmt(t["b"]), f"{t['diff']:+.3f} [{t['lo']:+.3f}, {t['hi']:+.3f}]".replace("-", "−"), f"{t['p_le0']:.4f}", f"{t['p_holm']:.4f}" if "p_holm" in t else "-"])
    if "H-L" in out["k6"]:
        t = out["k6"]["H-L"]
        trows.append(["H-L(속도 오차, m/s)", str(t["n_seeds"]), "-", "-", f"{t['diff']:+.2f} [{t['lo']:+.2f}, {t['hi']:+.2f}]".replace("-", "−"), f"{t['p_le0']:.4f}", f"{t['p_holm']:.4f}"])
    if "H-K4e" in out["tests"]:
        t = out["tests"]["H-K4e"]
        trows.append(["H-K4e(동등성 ±0.03, 90% CI)", str(t["n_seeds"]), "-", "-", f"{t['diff']:+.3f} [{t['lo90']:+.3f}, {t['hi90']:+.3f}]".replace("-", "−"), "동등" if t["equivalent"] else "동등 아님", "-"])
    _write_table(tables / f"{tag}_tests", ["가설·비교", "시드", "A", "B", "A − B [95% CI]", "단측 p", "Holm p"], trows)

    # ---------------- AMR test3(K5, A4) ----------------
    rc = cfg.for_domain("robot")
    if (rc.cache / f"expert_robot_{test_ev}.json").exists():
        exr = expert_reference(rc, test_ev)
        R = {(v, m, b): _S(rs, exr, True) for (v, m, b, lang), rs in _runs(root, "robot", test_ev).items()}
        rows = [[VAR_LABELS.get(v, v), LABELS.get(m, m), f"{int(round(100 * b))}%", str(len(S[1])), _fmt(float(S[0].mean(1).mean()), float(S[0].mean(1).std(ddof=1)) if len(S[1]) > 1 else 0.0)] for (v, m, b), S in sorted(R.items())]
        _write_table(tables / f"{tag}_robot", ["설정", "방법", "예산", "시드", "성공률"], rows)
        for name, a, b in (("K5", ("v4", "care", 0.02), ("v4", "random_shared", 0.02)), ("K5_v3", ("v3", "care", 0.02), ("v3", "random_shared", 0.02)), ("v4_vs_v3_care", ("v4", "care", 0.02), ("v3", "care", 0.02))):
            if a in R and b in R:
                A, B = _pair(R[a], R[b])
                out["robot"][name] = hierarchical_bootstrap(A, B) | {"n_seeds": len(A), "a": float(A.mean()), "b": float(B.mean())}
        out["robot"]["expert_success"] = float(np.mean([not e["collision_moving"] for e in exr["episodes"]]))

    # ---------------- 실영상 P5(A5, 탐색적) ----------------
    p5 = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "runs").glob("comma4__*full*__p5.json"))]
    base = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "runs").glob("comma4__*full*st1500.json"))]
    if p5:
        au5 = [r["open_loop"]["brake_onset_auroc"] for r in p5]
        au4 = [r["open_loop"]["brake_onset_auroc"] for r in base]
        out["comma_p5"] = {"auroc_p5": float(np.mean(au5)), "sd_p5": float(np.std(au5, ddof=1)) if len(au5) > 1 else 0.0, "auroc_v4": float(np.mean(au4)) if au4 else None, "baseline_neg_accel": float(np.mean([r["open_loop"]["baseline_neg_accel_auroc"] for r in p5])), "n_runs": len(p5)}

    # ---------------- KPI 판정 ----------------
    j = {}
    k = out["kpi"]
    if "K1" in k:
        j["K1"] = "달성" if k["K1"]["value"] >= 0.80 else "미달"
    if "K2" in k:
        j["K2"] = "달성" if k["K2"]["value"] >= 0.90 else "미달"
    if "K3" in k:
        j["K3"] = "달성" if k["K3"]["value"] >= 0.75 else "미달"
    if "K4" in out["tests"]:
        t = out["tests"]["K4"]
        j["K4"] = "달성" if t["lo"] > 0 else "미달"
    if "K5" in out["robot"]:
        j["K5"] = "달성" if out["robot"]["K5"]["lo"] > -0.03 else "미달"
    if "K6" in out["k6"]:
        j["K6"] = "달성" if out["k6"]["K6"]["speed_error_reduction"] >= 0.30 else "미달"
    out["judgement"] = j
    (root / "summary").mkdir(parents=True, exist_ok=True)
    (root / "summary" / f"{tag}_kpi.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    logger.info("세 번째 테스트 판정: %s", j)
    return out


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/exp_130_vla_v4")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--test-ev", default="test3")
    ap.add_argument("--cf-ev", default="cf3")
    ap.add_argument("--tag", default="third")
    a = ap.parse_args()
    build_third_report(Path(a.root), a.quick, a.test_ev, a.cf_ev, a.tag)


if __name__ == "__main__":
    main()
