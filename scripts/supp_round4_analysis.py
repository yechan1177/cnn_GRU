"""4차 심사(A22) 대응 보조 분석(재분석만, 새 실행 없음).

사전 등록으로 고정한 분석 코드(`vla_third_report.py`, `vla_v4_report.py`의 판정 로직)는 고치지 않고,
같은 원자료(`experiments/exp_130_vla_v4/runs/*.json`, `experiments/exp_120_vla_v3/runs/*.json`)와
같은 성공 정의(`vla_v3_report.episode_success`)로 다음을 계산한다. 모든 값은 판정에 쓰지 않는 보조(기술) 통계다.

- N1  : 3단계 선별 효과의 차이의 차이 (H-a3 − H-a3n) = (v4 CARE − v4 무작위) − (P3·P4 제외 CARE − 무작위)
- N2  : 2단계 K6의 시드·에피소드 계층 구간, 2·3단계 정책의 동일성 확인(학습 기록 대조), 2·3단계 전문가 성공률
- n2  : 3단계 정책 효과/선별 효과 비(H-v3 / H-a3)와 2·3단계 범위
- n4  : 2·3단계 시나리오별 성공률 표
- n8  : K2 비율의 에피소드 대응 재표집 구간(2·3단계)
- n12 : AMR 시드별 차이의 SD와 반폭 0.03에 필요한 시드 수, 3단계 분산 성분과 시드·에피소드 배분
- n13 : 같은 정책의 2·3단계 에피소드를 합친 K1·K2·K3(기술 통계)
- n9  : 표 42(3단계 가설 검정)의 한국어 행 이름 판과 차이의 차이 행

결과: `experiments/exp_130_vla_v4/summary/supp_round4.json`, 표 `summary/tables/supp4_*.md`
사용: PYTHONPATH=src nice -n 10 python scripts/supp_round4_analysis.py [--n-boot 10000]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from vcp.experiments.vla_report import _write_table, hierarchical_bootstrap
from vcp.experiments.vla_v3_report import episode_success
from vcp.experiments.vla_v4_report import _seeds_for_halfwidth

logger = logging.getLogger("supp_round4")

SCEN = ["cut_in", "lead_brake", "stop_and_go", "vru_crossing", "dense_traffic", "follow", "free_drive"]
SCEN_KO = {"cut_in": "끼어들기ʰ", "lead_brake": "선행차 급제동ʰ", "stop_and_go": "정체ʰ", "vru_crossing": "보행자 횡단ʰ", "dense_traffic": "밀집", "follow": "추종", "free_drive": "정상"}
METHOD_KO = {"care": "CARE 2%", "mix_trigger": "저장소 + 감속 트리거 2%", "random_shared": "무작위(공유 저장소) 2%", "full": "전체 100%"}


# ----------------------------------------------------------------------
# 입력
# ----------------------------------------------------------------------
def load_runs(root: Path, pattern: str) -> dict[tuple, list[dict[str, Any]]]:
    """runs/<pattern>을 (variant, method, budget, use_language)별로 묶고 시드 순으로 정렬한다."""

    g: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for p in sorted((root / "runs").glob(pattern)):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:  # 실행 중 쓰기 등
            logger.warning("읽기 실패 %s: %s", p.name, exc)
            continue
        j = r["job"]
        g[(j.get("variant", "v3"), j["method"], round(j["budget"], 2), j.get("use_language", True))].append(r)
    for v in g.values():
        v.sort(key=lambda r: r["job"]["seed"])
    return g


def load_expert(root: Path, name: str) -> dict[str, Any]:
    return json.loads((root / "cache" / name).read_text(encoding="utf-8"))


def success_matrix(rs: list[dict[str, Any]], ex: dict[str, Any], moving: bool = False) -> tuple[np.ndarray, list[int]]:
    return np.stack([episode_success(r["closed_loop"]["episodes"], ex, moving) for r in rs]).astype(float), [r["job"]["seed"] for r in rs]


def pair(a: tuple[np.ndarray, list[int]], b: tuple[np.ndarray, list[int]]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    common = sorted(set(a[1]) & set(b[1]))
    return a[0][[a[1].index(s) for s in common]], b[0][[b[1].index(s) for s in common]], common


def ci(bs: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    q = (1 - level) / 2
    return float(np.quantile(bs, q)), float(np.quantile(bs, 1 - q))


# ----------------------------------------------------------------------
# N1: 차이의 차이
# ----------------------------------------------------------------------
def diff_of_diffs(G: dict[tuple, list], ex: dict[str, Any], n_boot: int) -> dict[str, Any]:
    keys = [("v4", "care", 0.02, True), ("v4", "random_shared", 0.02, True), ("v4_noaux", "care", 0.02, True), ("v4_noaux", "random_shared", 0.02, True)]
    if not all(k in G for k in keys):
        return {}
    M = [success_matrix(G[k], ex) for k in keys]
    seeds = sorted(set.intersection(*[set(m[1]) for m in M]))
    X = [m[0][[m[1].index(s) for s in seeds]] for m in M]
    d1, d2 = X[0] - X[1], X[2] - X[3]
    hb = hierarchical_bootstrap(d1, d2, n_boot=n_boot)
    return hb | {"h_a3": float(d1.mean()), "h_a3n": float(d2.mean()), "n_seeds": len(seeds)}


# ----------------------------------------------------------------------
# N2: K6 계층 구간(2단계)
# ----------------------------------------------------------------------
def ep_speed_error(rs: list[dict[str, Any]]) -> tuple[np.ndarray, list[int]]:
    return np.array([[np.nan if e.get("speed_error") is None else float(e["speed_error"]) for e in r["closed_loop"]["episodes"]] for r in rs]), [r["job"]["seed"] for r in rs]


def k6_hier(H: dict[tuple, list], n_boot: int) -> dict[str, Any]:
    a, b = ("v4", "care", 0.02, True), ("v4", "care", 0.02, False)
    if a not in H or b not in H:
        return {}
    L, N, seeds = pair(ep_speed_error(H[a]), ep_speed_error(H[b]))
    rng = np.random.default_rng(0)
    n_s, n_e = L.shape
    bs = np.empty(n_boot)
    for i in range(n_boot):
        si, ei = rng.integers(0, n_s, n_s), rng.integers(0, n_e, n_e)
        bs[i] = 1 - np.nanmean(L[np.ix_(si, ei)]) / np.nanmean(N[np.ix_(si, ei)])
    lo, hi = ci(bs)
    run_l = {r["job"]["seed"]: r["closed_loop"]["overall"].get("speed_error") for r in H[a]}
    run_n = {r["job"]["seed"]: r["closed_loop"]["overall"].get("speed_error") for r in H[b]}
    point = 1 - np.mean([run_l[s] for s in seeds]) / np.mean([run_n[s] for s in seeds])
    return {"point": float(point), "lo": lo, "hi": hi, "p_ge_goal": float((bs >= 0.30).mean()), "n_seeds": len(seeds), "n_episodes": int(n_e)}


# ----------------------------------------------------------------------
# N2: 2·3단계 정책의 동일성(학습 기록 대조)
# ----------------------------------------------------------------------
def _fingerprint(r: dict[str, Any]) -> tuple:
    t = r.get("train_log") or {}
    s = r.get("selection") or {}
    return (
        t.get("final_loss"),
        t.get("final_aux_loss"),
        tuple(t.get("aux_loss_curve") or ()),
        tuple(t.get("loss_curve") or ()),
        t.get("lang_dropped_frac"),
        t.get("n_train"),
        s.get("n_selected"),
        s.get("hazard_frame_recall"),
        r.get("n_train_frames"),
    )


def same_policy(root: Path) -> dict[str, Any]:
    """3단계 실행마다, 2단계(본 실험·탐색·개발)에 학습 설정이 같은 실행이 있으면 학습 기록이 같은지 대조한다.

    학습 설정 키 = (도메인, 설정, 방법, 예산, 학습 시드, 학습 단계 수, 언어 사용). 평가 세트는 키에 넣지 않는다
    (같은 설정·시드의 정책을 다른 평가 세트에서 평가했는지 보려는 것이므로).
    """

    def key(r: dict[str, Any]) -> tuple:
        j = r["job"]
        return (j["domain"], j.get("variant"), j["method"], round(j["budget"], 2), j["seed"], j["steps"], j.get("use_language", True))

    stage2: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    stage3: list[dict[str, Any]] = []
    for p in sorted((root / "runs").glob("*.json")):
        if p.name.startswith("comma"):
            continue
        parts = p.name.split("__")
        dom, ev = parts[0], parts[1]
        if ev in ("test3", "cf3"):
            stage3.append(json.loads(p.read_text(encoding="utf-8")))
        elif ev in ("test", "cf", "dev", "devcf"):
            stage2[(dom, ev)].append(json.loads(p.read_text(encoding="utf-8")))
    s2 = defaultdict(list)
    for rs in stage2.values():
        for r in rs:
            s2[key(r)].append(r)
    out: dict[str, Any] = {"n_stage3_runs": 0, "n_trained_stage3": 0, "n_matched": 0, "n_identical": 0, "n_unmatched": 0, "by_domain": {}, "by_condition": {}, "mismatch_examples": []}
    for r in stage3:
        out["n_stage3_runs"] += 1
        if r["job"]["steps"] == 0:  # 제어기 단독(학습 없음)
            continue
        out["n_trained_stage3"] += 1
        j = r["job"]
        dom = j["domain"]
        bd = out["by_domain"].setdefault(dom, {"trained": 0, "matched": 0, "identical": 0})
        bc = out["by_condition"].setdefault(f"{dom}:{j['eval']}:{j.get('variant')}:{j['method']}:{round(j['budget'], 2)}:{'lang' if j.get('use_language', True) else 'nolang'}", {"seeds": [], "identical_seeds": [], "sources": []})
        bd["trained"] += 1
        bc["seeds"].append(j["seed"])
        cands = s2.get(key(r))
        if not cands:
            out["n_unmatched"] += 1
            continue
        out["n_matched"] += 1
        bd["matched"] += 1
        same = [c for c in cands if _fingerprint(c) == _fingerprint(r)]
        if same:
            out["n_identical"] += 1
            bd["identical"] += 1
            bc["identical_seeds"].append(j["seed"])
            bc["sources"] = sorted(set(bc["sources"]) | {c["job"]["eval"] for c in same})
        elif len(out["mismatch_examples"]) < 5:
            out["mismatch_examples"].append(list(map(str, key(r))))
    for v in out["by_condition"].values():
        v["seeds"].sort()
        v["identical_seeds"].sort()
    return out


# ----------------------------------------------------------------------
# n4·N2: 시나리오별 성공률과 전문가 성공률
# ----------------------------------------------------------------------
def scenario_rates(G: dict[tuple, list], ex: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for var, m, b in (("v4", "care", 0.02), ("v4", "mix_trigger", 0.02), ("v4", "random_shared", 0.02), ("v4", "full", 1.0), ("v3", "care", 0.02)):
        rs = G.get((var, m, b, True))
        if not rs:
            continue
        per: dict[str, list[float]] = defaultdict(list)
        for r in rs:
            eps = r["closed_loop"]["episodes"]
            for e, ok in zip(eps, episode_success(eps, ex, False)):
                per[e["scenario"]].append(float(ok))
        out.setdefault(var, {})[m] = {c: float(np.mean(per[c])) for c in SCEN if c in per} | {"n_seeds": len(rs)}
    eps = ex["episodes"]
    out["expert"] = {c: float(np.mean([not e["collision"] for e in eps if e["scenario"] == c])) for c in SCEN}
    out["expert_summary"] = {
        "driving": float(np.mean([not e["collision"] for e in eps])),
        "hazard": float(np.mean([not e["collision"] for e in eps if e["hazard"]])),
        "vru_crossing": out["expert"]["vru_crossing"],
    }
    return out


def scenario_table(s2: dict[str, Any], s3: dict[str, Any], path: Path) -> None:
    rows = []
    for label, var, m in (("v4", "v4", "care"), ("v4", "v4", "mix_trigger"), ("v4", "v4", "random_shared"), ("v4", "v4", "full"), ("v3", "v3", "care")):
        for stage, s in (("2차", s2), ("3차", s3)):
            d = s.get(var, {}).get(m)
            if d:
                rows.append([stage, label, METHOD_KO[m], str(d["n_seeds"])] + [f"{d[c]:.3f}" for c in SCEN])
    for stage, s in (("2차", s2), ("3차", s3)):
        rows.append([stage, "전문가", "-", "-"] + [f"{s['expert'][c]:.3f}" for c in SCEN])
    _write_table(path, ["단계", "설정", "방법", "시드"] + [SCEN_KO[c] for c in SCEN], rows)


# ----------------------------------------------------------------------
# n8·n13: K2 대응 재표집, 2·3단계 종합
# ----------------------------------------------------------------------
def pooled_kpis(sets: list[tuple[np.ndarray, np.ndarray, np.ndarray]], n_boot: int) -> dict[str, Any]:
    """sets: [(CARE [시드×에피소드], 전체 [시드×에피소드], 위험 마스크)]. 같은 정책이므로 시드 표집은 세트 사이에 공유하고,
    에피소드는 세트 안에서 층화 재표집하며 CARE와 전체에 같은 에피소드를 쓴다(에피소드 대응)."""

    rng = np.random.default_rng(0)
    n_c, n_f = sets[0][0].shape[0], sets[0][1].shape[0]
    k1, k2, k3 = np.empty(n_boot), np.empty(n_boot), np.empty(n_boot)
    for i in range(n_boot):
        sc, sf = rng.integers(0, n_c, n_c), rng.integers(0, n_f, n_f)
        c_all, f_all, c_hz = [], [], []
        for C, F, hz in sets:
            ei = rng.integers(0, C.shape[1], C.shape[1])
            c_all.append(C[np.ix_(sc, ei)])
            f_all.append(F[np.ix_(sf, ei)])
            c_hz.append(C[np.ix_(sc, ei[hz[ei]])])
        c = np.concatenate(c_all, axis=1).mean()
        k1[i], k2[i] = c, c / np.concatenate(f_all, axis=1).mean()
        k3[i] = np.concatenate(c_hz, axis=1).mean()
    Cc = np.concatenate([s[0] for s in sets], axis=1)
    Fc = np.concatenate([s[1] for s in sets], axis=1)
    Hc = np.concatenate([s[0][:, s[2]] for s in sets], axis=1)
    res = {}
    for name, val, bs, goal in (("K1", Cc.mean(), k1, 0.80), ("K2", Cc.mean() / Fc.mean(), k2, 0.90), ("K3", Hc.mean(), k3, 0.75)):
        lo, hi = ci(bs)
        res[name] = {"value": float(val), "lo": lo, "hi": hi, "p_ge_goal": float((bs >= goal).mean())}
    res["n_seeds_care"], res["n_seeds_full"] = int(n_c), int(n_f)
    res["n_episodes"] = int(sum(s[0].shape[1] for s in sets))
    return res


def stage_kpi_inputs(G: dict[tuple, list], ex: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    c, f = G.get(("v4", "care", 0.02, True)), G.get(("v4", "full", 1.0, True))
    if not c or not f:
        return None
    C, sc = success_matrix(c, ex)
    F, sf = success_matrix(f, ex)
    hz = np.array([bool(e["hazard"]) for e in c[0]["closed_loop"]["episodes"]])
    return C, F, hz


# ----------------------------------------------------------------------
# n12: AMR 검정력
# ----------------------------------------------------------------------
def amr_seed_sd(G: dict[tuple, list], ex: dict[str, Any], var: str) -> dict[str, Any]:
    a, b = G.get((var, "care", 0.02, True)), G.get((var, "random_shared", 0.02, True))
    if not a or not b:
        return {}
    A, B, seeds = pair(success_matrix(a, ex, True), success_matrix(b, ex, True))
    d = A.mean(1) - B.mean(1)
    sd = float(d.std(ddof=1))
    return {"n_seeds": len(seeds), "n_episodes": int(A.shape[1]), "mean_diff": float(d.mean()), "sd_seed_diff": sd, "seeds_for_halfwidth_0.03": int(_seeds_for_halfwidth(sd, 0.03)), "D": A - B}


def variance_components(D: np.ndarray, observed_halfwidth: float | None = None) -> dict[str, Any]:
    """대응 차이 행렬 D[시드, 에피소드]의 2원 분산 분석(반복 없음)으로 시드·에피소드·잔차 성분을 추정하고,
    사전 등록 판정에 쓰는 시드·에피소드 2단계 부트스트랩 구간의 반폭이 0.03 미만이 되는 시드·에피소드 배분을 근사한다.

    2단계 재표집에서 평균 차이의 분산은 대략 σ²_s/S + σ²_e/E + 2σ²_r/(S·E)이다(시드 재표집과 에피소드 재표집이
    잔차 성분을 각각 한 번씩 반영한다). 관측된 3단계 반폭에 맞도록 상수 배율로 보정한 뒤 배분을 구한다(근사).
    """

    S, E = D.shape
    gm = D.mean()
    rm, cm = D.mean(1), D.mean(0)
    ms_s = E * ((rm - gm) ** 2).sum() / (S - 1)
    ms_e = S * ((cm - gm) ** 2).sum() / (E - 1)
    resid = D - rm[:, None] - cm[None, :] + gm
    ms_r = (resid ** 2).sum() / ((S - 1) * (E - 1))
    vs, ve, vr = max((ms_s - ms_r) / E, 0.0), max((ms_e - ms_r) / S, 0.0), ms_r

    def var_boot(s: int, e: int) -> float:
        return vs / s + ve / e + 2 * vr / (s * e)

    scale = (observed_halfwidth / 1.96) ** 2 / var_boot(S, E) if observed_halfwidth else 1.0

    def halfwidth(s: int, e: int) -> float:
        return 1.96 * math.sqrt(scale * var_boot(s, e))

    plan = {}
    for e in (E, 2 * E, 4 * E):
        plan[str(e)] = next((s for s in range(2, 401) if halfwidth(s, e) < 0.03), None)
    return {
        "var_seed": float(vs), "var_episode": float(ve), "var_resid": float(vr),
        "var_row_means": float(rm.var(ddof=1)), "var_col_means": float(cm.var(ddof=1)),
        "model_halfwidth_current": float(1.96 * math.sqrt(var_boot(S, E))),
        "calibration_scale": float(scale),
        "seeds_needed_by_episodes": plan,
    }


# ----------------------------------------------------------------------
# n9: 표 42 한국어 판
# ----------------------------------------------------------------------
def tests_table(t3: dict[str, Any], dd: dict[str, Any], path: Path) -> None:
    T, K6 = t3.get("tests", {}), t3.get("k6", {})

    def f3(x: float) -> str:
        return f"{x:.3f}"

    def ci3(t: dict[str, Any], lo: str = "lo", hi: str = "hi") -> str:
        return f"{t['diff']:+.3f} [{t[lo]:+.3f}, {t[hi]:+.3f}]".replace("-", "−")

    rows = []
    names = [
        ("H-v3", "H-v3: v4 CARE − v3 CARE", "확증(군 A)"),
        ("H-a3", "H-a3: v4 CARE − v4 무작위", "확증(군 A)"),
        ("H-a3n", "H-a3n: P3·P4 제외 v4의 CARE − 무작위", "확증(군 A)"),
        ("H-P7", "H-P7: v4 CARE − 전부+P6 CARE", "확증(군 A)"),
        ("K4", "K4: v4 CARE − v4 저장소 + 감속 트리거", "KPI(CI 하한 > 0)"),
        ("mix_vs_random_v4", "트리거 혼합 − 무작위(v4)", "보조"),
        ("mix_vs_random_noaux", "트리거 혼합 − 무작위(P3·P4 제외 v4)", "보조"),
        ("v4_vs_noaux_care", "v4 CARE − P3·P4 제외 v4 CARE", "보조"),
    ]
    for k, label, kind in names:
        if k in T:
            t = T[k]
            rows.append([label, kind, str(t["n_seeds"]), f3(t["a"]), f3(t["b"]), ci3(t), f"{t['p_le0']:.4f}", f"{t['p_holm']:.4f}" if "p_holm" in t else "-"])
    if dd:
        rows.append(["(H-a3 − H-a3n): 선별 효과의 차이", "보조(4차 심사)", str(dd["n_seeds"]), f3(dd["h_a3"]), f3(dd["h_a3n"]), ci3(dd), f"{dd['p_le0']:.4f}", "-"])
    if "H-L" in K6:
        t = K6["H-L"]
        rows.append(["H-L: v4(언어) − P6·P7만, 속도 오차(m/s)", "확증(군 A)", str(t["n_seeds"]), f"{K6['mean_speed_error']['v4_lang']:.2f}", f"{K6['mean_speed_error']['p67_only']:.2f}", f"{t['diff']:+.2f} [{t['lo']:+.2f}, {t['hi']:+.2f}]".replace("-", "−"), f"{t['p_le0']:.4f}", f"{t['p_holm']:.4f}"])
        if "H-L_hier" in K6:
            h = K6["H-L_hier"]
            rows.append(["H-L의 시드·에피소드 계층 구간(m/s)", "보조(4차 심사)", str(h["n_seeds"]), "-", "-", f"{t['diff']:+.2f} [{h['lo']:+.2f}, {h['hi']:+.2f}]".replace("-", "−"), f"{h['p_le0']:.4f}", "-"])
    if "H-K4e" in T:
        t = T["H-K4e"]
        rows.append(["H-K4e: P3·P4 제외 v4의 CARE − 트리거 혼합(동등 ±0.03, 90% CI)", "동등성", str(t["n_seeds"]), "-", "-", ci3(t, "lo90", "hi90"), "동등" if t["equivalent"] else "동등 아님", "-"])
    _write_table(path, ["가설·비교", "구분", "시드", "A", "B", "A − B [95% CI]", "단측 p", "Holm p"], rows)


# ----------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/exp_130_vla_v4")
    ap.add_argument("--v3-root", default="experiments/exp_120_vla_v3")
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    root, v3root, nb = Path(a.root), Path(a.v3_root), min(a.n_boot, 10000)
    tables = root / "summary" / "tables"
    t3 = json.loads((root / "summary" / "third_kpi.json").read_text(encoding="utf-8"))
    v4k = json.loads((root / "summary" / "v4_kpi.json").read_text(encoding="utf-8"))
    out: dict[str, Any] = {"n_boot": nb}

    ex2, ex3 = load_expert(root, "expert_driving_test.json"), load_expert(root, "expert_driving_test3.json")
    G2, G3 = load_runs(root, "driving__test__*.json"), load_runs(root, "driving__test3__*.json")

    # N1
    out["ha_diff"] = diff_of_diffs(G3, ex3, nb)
    out["ha_diff_stage2_explore"] = diff_of_diffs(G2, ex2, nb)
    logger.info("N1 차이의 차이: %s", {k: out["ha_diff"].get(k) for k in ("diff", "lo", "hi", "p_le0")})

    # N2: K6 계층 구간(2단계), 3단계는 third_kpi의 K6_hier와 같은 함수 형태로 재확인
    out["k6_hier_stage2"] = k6_hier(load_runs(root, "driving__cf__*.json"), nb)
    out["k6_hier_stage3_check"] = k6_hier(load_runs(root, "driving__cf3__*.json"), nb)
    logger.info("N2 K6 계층(2단계): %s", out["k6_hier_stage2"])

    # N2: 같은 정책 확인
    out["same_policy"] = same_policy(root)
    logger.info("N2 같은 정책: %s", {k: v for k, v in out["same_policy"].items() if k != "mismatch_examples"})

    # n4·N2: 시나리오별, 전문가
    sc2, sc3 = scenario_rates(G2, ex2), scenario_rates(G3, ex3)
    out["scenario"] = {"stage2": sc2, "stage3": sc3}
    out["expert"] = {"stage2": sc2["expert_summary"], "stage3": sc3["expert_summary"]}
    scenario_table(sc2, sc3, tables / "supp4_scenario_23")

    # n8·n13: K2 대응 재표집과 2·3단계 종합
    i2, i3 = stage_kpi_inputs(G2, ex2), stage_kpi_inputs(G3, ex3)
    if i2 and i3:
        out["k2_paired"] = {"stage2": pooled_kpis([i2], nb)["K2"], "stage3": pooled_kpis([i3], nb)["K2"]}
        out["pooled_23"] = pooled_kpis([i2, i3], nb)
        logger.info("n13 종합: %s", out["pooled_23"])

    # n3: 학습 없는 제어기 단독의 오차 감소율(언어 없음 v4 대비, 3단계 cf3)
    mse = t3.get("k6", {}).get("mean_speed_error", {})
    if mse.get("ctrl") and mse.get("v4_nolang"):
        out["ctrl_reduction_stage3"] = 1 - mse["ctrl"] / mse["v4_nolang"]

    # n2: 정책 효과/선별 효과 비
    hv3, ha3 = t3["tests"]["H-v3"]["diff"], t3["tests"]["H-a3"]["diff"]
    r2 = v4k.get("ratios", {})
    out["ratio"] = {"stage3": hv3 / ha3, "stage2_min": r2.get("policy_over_selection_min"), "stage2_max": r2.get("policy_over_selection_max")}
    out["ratio"]["min_23"] = min(x for x in (out["ratio"]["stage3"], out["ratio"]["stage2_min"]) if x is not None)
    out["ratio"]["max_23"] = max(x for x in (out["ratio"]["stage3"], out["ratio"]["stage2_max"]) if x is not None)

    # n12: AMR 검정력
    rex1 = load_expert(v3root, "expert_robot_test.json")
    rex2, rex3 = load_expert(root, "expert_robot_test.json"), load_expert(root, "expert_robot_test3.json")
    R1 = load_runs(v3root, "robot__test__*.json")
    R2, R3 = load_runs(root, "robot__test__*.json"), load_runs(root, "robot__test3__*.json")
    amr = {"v3_stage1": amr_seed_sd(R1, rex1, "v3"), "v4_stage2": amr_seed_sd(R2, rex2, "v4"), "v4_stage3": amr_seed_sd(R3, rex3, "v4"), "v3_stage3": amr_seed_sd(R3, rex3, "v3")}
    D3 = amr["v4_stage3"].pop("D", None)
    for v in amr.values():
        v.pop("D", None)
    sds = [v["sd_seed_diff"] for v in amr.values() if v]
    amr["sd_max"] = float(max(sds))
    amr["sd_min"] = float(min(sds))
    amr["seeds_for_halfwidth_0.03_min"] = int(_seeds_for_halfwidth(min(sds), 0.03))
    amr["seeds_for_halfwidth_0.03_max"] = int(_seeds_for_halfwidth(max(sds), 0.03))
    if D3 is not None:
        hw = (t3["robot"]["K5"]["hi"] - t3["robot"]["K5"]["lo"]) / 2
        amr["components_stage3"] = variance_components(D3, hw)
        amr["stage3_hier_halfwidth"] = float(hw)
        amr["info_ratio_for_0.03"] = float((hw / 0.03) ** 2)
    out["amr_power"] = amr
    logger.info("n12 AMR: %s", amr)

    # n9: 표 42 한국어 판
    tests_table(t3, out["ha_diff"], tables / "supp4_third_tests")

    path = root / "summary" / "supp_round4.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    logger.info("저장: %s", path)


if __name__ == "__main__":
    main()
