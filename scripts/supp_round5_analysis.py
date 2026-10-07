"""5차 심사(A25) 대응 보조 분석(재집계만, 새 실행 없음).

사전 등록으로 고정한 분석 코드(`vla_third_report.py`, `vla_v4_report.py`의 판정 로직, `vla_v4_suite.py`)는 고치지 않고,
같은 원자료(`experiments/exp_130_vla_v4/runs/*.json`)와 같은 성공 정의(`vla_v3_report.episode_success`)로 다음을 계산한다.
4차 대응 보조 분석(`scripts/supp_round4_analysis.py`)의 함수를 그대로 재사용해 4단계(test4·cf4)로 범위를 넓힌다.
모든 값은 판정에 쓰지 않는 보조 통계이며, 아래 '사후'로 표시한 항목은 사전 등록 가설이 아닌 사후 분석이다.

- r7·n4 : 4단계 시나리오별 성공률(v4 CARE·트리거 혼합·무작위·전체, v3 CARE, 전문가), 표 `supp5_scenario_4`,
          2·3·4단계 각각에서 v4 CARE 2%의 최저 시나리오
- r8·n8 : 4단계 K2 비율의 에피소드 대응 재표집 구간과 목표 이상 비율(사전 등록 K2는 독립 재표집)
- (사후) 4단계 선별 효과의 차이의 차이 H-a4 − H-a4n
- r4·n2 : (사후) 정책 효과 − 선별 효과 = H-v − H-a = v4 무작위 − v3 CARE(3·4단계), 4단계 정책/선별 효과 비와 2·3·4단계 범위
- r6    : v4 + P5 탐색(개발 세트)에서 2단계 개발 선택에 쓰이지 않은 시드(3·4)만의 비교, 새로 실행한 실행 수(로그 근거)
- r3    : 4단계 실행 수와 고유 정책 수(학습 설정·학습 기록 대조), 이전 단계와의 중복 여부
- r2·n9 : 표 48(4단계 가설 검정)의 한국어 행 이름·'구분' 열 판, 표 `supp5_fourth_tests`

결과: `experiments/exp_130_vla_v4/summary/supp_round5.json`, 표 `summary/tables/supp5_*.md`
사용: PYTHONPATH=src nice -n 10 python scripts/supp_round5_analysis.py [--n-boot 10000]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 같은 폴더의 4차 보조 분석 모듈 재사용

from supp_round4_analysis import (  # noqa: E402
    METHOD_KO,
    SCEN,
    SCEN_KO,
    _fingerprint,
    diff_of_diffs,
    load_expert,
    load_runs,
    pair,
    pooled_kpis,
    scenario_rates,
    stage_kpi_inputs,
    success_matrix,
)
from vcp.experiments.vla_report import _write_table, hierarchical_bootstrap  # noqa: E402
from vcp.experiments.vla_v3_report import episode_success  # noqa: E402

logger = logging.getLogger("supp_round5")

STAGE4_EVALS = ("test4", "cf4")


def _ci_txt(d: float, lo: float, hi: float, digits: int = 3) -> str:
    return f"{d:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]".replace("-", "−")


# ----------------------------------------------------------------------
# r7·n4: 4단계 시나리오
# ----------------------------------------------------------------------
def scenario_table_4(s4: dict[str, Any], path: Path) -> None:
    rows = []
    for label, var, m in (("v4", "v4", "care"), ("v4", "v4", "mix_trigger"), ("v4", "v4", "random_shared"), ("v4", "v4", "full"), ("v3", "v3", "care")):
        d = s4.get(var, {}).get(m)
        if d:
            rows.append([label, METHOD_KO[m], str(d["n_seeds"])] + [f"{d[c]:.3f}" for c in SCEN])
    rows.append(["전문가", "-", "-"] + [f"{s4['expert'][c]:.3f}" for c in SCEN])
    _write_table(path, ["설정", "방법", "시드"] + [SCEN_KO[c] for c in SCEN], rows)


def lowest_scenario(sc: dict[str, Any], var: str = "v4", method: str = "care") -> dict[str, Any]:
    d = sc.get(var, {}).get(method, {})
    vals = {c: d[c] for c in SCEN if c in d}
    if not vals:
        return {}
    c = min(vals, key=vals.get)
    return {"scenario": c, "value": float(vals[c])}


# ----------------------------------------------------------------------
# r4: 정책 효과 − 선별 효과(사후)
# ----------------------------------------------------------------------
def policy_minus_selection(G: dict[tuple, list], ex: dict[str, Any], n_boot: int) -> dict[str, Any]:
    """H-v − H-a = (v4 CARE − v3 CARE) − (v4 CARE − v4 무작위) = v4 무작위 − v3 CARE. 같은 시드·에피소드 대응 계층 부트스트랩."""

    a, b = ("v4", "random_shared", 0.02, True), ("v3", "care", 0.02, True)
    if a not in G or b not in G:
        return {}
    A, B, seeds = pair(success_matrix(G[a], ex), success_matrix(G[b], ex))
    return hierarchical_bootstrap(A, B, n_boot=n_boot) | {"a_v4_random": float(A.mean()), "b_v3_care": float(B.mean()), "seeds": seeds}


# ----------------------------------------------------------------------
# r6: p5dev 탐색의 시드 3·4와 새 실행 수
# ----------------------------------------------------------------------
def p5dev_breakdown(root: Path, p5_saved: dict[str, Any]) -> dict[str, Any]:
    ex = load_expert(root, "expert_driving_dev.json")
    out: dict[str, Any] = {}
    for v in ("all_p7_p5", "all_p7"):
        by_seed = {}
        for s in range(5):
            p = root / "runs" / f"driving__dev__{v}__care__b0.02__s{s}__st6000.json"
            if p.exists():
                r = json.loads(p.read_text(encoding="utf-8"))
                by_seed[s] = float(episode_success(r["closed_loop"]["episodes"], ex, False).mean())
        new = [by_seed[s] for s in (3, 4) if s in by_seed]
        out[v] = {"by_seed": {str(k): v_ for k, v_ in by_seed.items()}, "mean_all": float(np.mean(list(by_seed.values()))), "mean_s34": float(np.mean(new)) if new else None}
        saved = p5_saved.get(v, {}).get("by_seed")
        if saved:  # p5dev_explore.json(소수 셋째 자리 반올림)과 대조
            out[v]["matches_p5dev_explore"] = bool(np.allclose(np.round([by_seed[s] for s in range(len(saved))], 3), saved))
    a, b = out.get("all_p7", {}).get("by_seed", {}), out.get("all_p7_p5", {}).get("by_seed", {})
    out["s34_p5_lower_both"] = bool(all(b[str(s)] < a[str(s)] for s in (3, 4) if str(s) in a and str(s) in b))
    out["all_p5_lower"] = bool(all(b[k] < a[k] for k in a if k in b))

    # 새로 실행한 실행: p5dev 단계 로그의 '완료 <키>' 행(실행을 끝낸 작업만 기록됨)과 계획 작업 수
    log = root / "logs" / "p5dev.log"
    done, planned, remaining = [], None, None
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            m = re.search(r"작업 (\d+)개\(남은 것 (\d+)개\)", line)
            if m:
                planned, remaining = int(m.group(1)), int(m.group(2))
            m = re.search(r"완료 (driving__dev__\S+?):", line)
            if m:
                done.append(m.group(1))
    planned_keys = [f"driving__dev__{v}__care__b0.02__s{s}__st6000" for v in ("all_p7_p5", "all_p7") for s in range(5)]
    reused = sorted(k for k in planned_keys if k not in done)
    out["runs"] = {"planned": planned, "remaining_at_start": remaining, "executed": sorted(done), "n_executed": len(done), "reused": reused, "n_reused": len(reused)}
    # 파일 수정 시각(보조 근거; 저장소 체크아웃 방식에 따라 바뀔 수 있다)
    stamps = {}
    for k in reused:
        p = root / "runs" / f"{k}.json"
        if p.exists():
            import datetime as _dt

            stamps[k] = _dt.datetime.fromtimestamp(p.stat().st_mtime, tz=_dt.timezone.utc).isoformat(timespec="seconds")
    out["runs"]["reused_mtime_utc"] = stamps
    return out


# ----------------------------------------------------------------------
# r3: 4단계 실행 수와 고유 정책 수
# ----------------------------------------------------------------------
def _train_key(r: dict[str, Any]) -> tuple:
    j = r["job"]
    return (j["domain"], j.get("variant"), j["method"], round(j["budget"], 2), j["seed"], j["steps"], j.get("use_language", True))


def unique_policies(root: Path) -> dict[str, Any]:
    """4단계(test4·cf4) 실행을 학습 설정 키로 묶고, 같은 키의 실행끼리 학습 기록(최종 손실·보조 손실 곡선 등)이 같은지 대조한다.

    평가 세트는 키에 넣지 않는다(같은 정책을 다른 세트에서 평가했는지 보려는 것). 이전 단계 실행과 키·학습 기록이 겹치는지도 본다.
    """

    stage4: list[dict[str, Any]] = []
    other_keys: set[tuple] = set()
    other_fp: set[tuple] = set()
    for p in sorted((root / "runs").glob("*.json")):
        if p.name.startswith("comma"):
            continue
        ev = p.name.split("__")[1]
        r = json.loads(p.read_text(encoding="utf-8"))
        if ev in STAGE4_EVALS:
            stage4.append(r)
        else:
            other_keys.add(_train_key(r))
            other_fp.add(_fingerprint(r))
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in stage4:
        groups[_train_key(r)].append(r)
    shared = []
    for k, rs in groups.items():
        if len(rs) > 1:
            fps = {_fingerprint(r) for r in rs}
            shared.append({"key": list(map(str, k)), "evals": sorted(r["job"]["eval"] for r in rs), "identical_train_log": len(fps) == 1})
    by_cond: dict[str, int] = defaultdict(int)
    for r in stage4:
        j = r["job"]
        by_cond[f"{j['eval']}:{j.get('variant')}:{j['method']}:{round(j['budget'], 2)}:{'lang' if j.get('use_language', True) else 'nolang'}"] += 1
    n_unique_fp = len({_fingerprint(r) for r in stage4})
    shared_conds = defaultdict(int)
    for s in shared:
        shared_conds[f"{s['key'][1]}:{s['key'][2]}:{s['key'][3]}:{'lang' if s['key'][6] == 'True' else 'nolang'}:{'+'.join(s['evals'])}"] += 1
    return {
        "n_runs": len(stage4),
        "n_unique_train_keys": len(groups),
        "n_unique_train_logs": n_unique_fp,
        "n_shared_keys": len(shared),
        "all_shared_identical": all(s["identical_train_log"] for s in shared),
        "shared_conditions": dict(shared_conds),
        "by_condition": dict(sorted(by_cond.items())),
        "seeds": sorted({r["job"]["seed"] for r in stage4}),
        "overlap_keys_with_other_stages": len(set(groups) & other_keys),
        "overlap_train_logs_with_other_stages": len({_fingerprint(r) for r in stage4} & other_fp),
    }


# ----------------------------------------------------------------------
# r2·n9: 표 48 한국어 판
# ----------------------------------------------------------------------
def fourth_tests_table(t4: dict[str, Any], dd: dict[str, Any], pms: dict[str, Any], path: Path) -> None:
    T, K6 = t4.get("tests", {}), t4.get("k6", {})

    def f3(x: float) -> str:
        return f"{x:.3f}"

    rows = []
    names = [
        ("H-v3", "H-v4: v4 CARE − v3 CARE", "확증(군 A)"),
        ("H-a3", "H-a4: v4 CARE − v4 무작위", "확증(군 A)"),
        ("H-a3n", "H-a4n: P3·P4 제외 v4의 CARE − 무작위", "확증(군 A)"),
        ("H-P7", "H-P7b: v4 CARE − 전부+P6 CARE", "확증(군 A)"),
        ("K4", "K4: v4 CARE − v4 저장소 + 감속 트리거", "KPI(CI 하한 > 0)"),
        ("mix_vs_random_v4", "트리거 혼합 − 무작위(v4)", "보조"),
        ("mix_vs_random_noaux", "트리거 혼합 − 무작위(P3·P4 제외 v4)", "보조"),
        ("v4_vs_noaux_care", "v4 CARE − P3·P4 제외 v4 CARE", "보조"),
    ]
    for k, label, kind in names:
        if k in T:
            t = T[k]
            rows.append([label, kind, str(t["n_seeds"]), f3(t["a"]), f3(t["b"]), _ci_txt(t["diff"], t["lo"], t["hi"]), f"{t['p_le0']:.4f}", f"{t['p_holm']:.4f}" if "p_holm" in t else "-"])
    if dd:
        rows.append(["(H-a4 − H-a4n): 선별 효과의 차이", "보조(사후, 5차 심사)", str(dd["n_seeds"]), f3(dd["h_a3"]), f3(dd["h_a3n"]), _ci_txt(dd["diff"], dd["lo"], dd["hi"]), f"{dd['p_le0']:.4f}", "-"])
    if pms:
        rows.append(["(H-v4 − H-a4) = v4 무작위 − v3 CARE: 정책 효과 − 선별 효과", "보조(사후, 5차 심사)", str(pms["n_seeds"]), f3(pms["a_v4_random"]), f3(pms["b_v3_care"]), _ci_txt(pms["diff"], pms["lo"], pms["hi"]), f"{pms['p_le0']:.4f}", "-"])
    if "H-L" in K6:
        t = K6["H-L"]
        mse = K6["mean_speed_error"]
        rows.append(["H-L4: v4(언어) − P6·P7만, 속도 오차(m/s)", "확증(군 A)", str(t["n_seeds"]), f"{mse['v4_lang']:.2f}", f"{mse['p67_only']:.2f}", _ci_txt(t["diff"], t["lo"], t["hi"], 2), f"{t['p_le0']:.4f}", f"{t['p_holm']:.4f}"])
        if "H-L_hier" in K6:
            h = K6["H-L_hier"]
            rows.append(["H-L4의 시드·에피소드 계층 구간(m/s)", "보조(사전 등록 보조 분석)", str(h["n_seeds"]), "-", "-", _ci_txt(t["diff"], h["lo"], h["hi"], 2), f"{h['p_le0']:.4f}", "-"])
    if "H-K4e" in T:
        t = T["H-K4e"]
        rows.append(["H-K4e: P3·P4 제외 v4의 CARE − 트리거 혼합(동등 ±0.03, 90% CI)", "동등성", str(t["n_seeds"]), "-", "-", _ci_txt(t["diff"], t["lo90"], t["hi90"]), "동등" if t["equivalent"] else "동등 아님", "-"])
    _write_table(path, ["가설·비교", "구분", "시드", "A", "B", "A − B [95% CI]", "단측 p", "Holm p"], rows)


# ----------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/exp_130_vla_v4")
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    root, nb = Path(a.root), min(a.n_boot, 10000)
    summ, tables = root / "summary", root / "summary" / "tables"
    t3 = json.loads((summ / "third_kpi.json").read_text(encoding="utf-8"))
    t4 = json.loads((summ / "fourth_kpi.json").read_text(encoding="utf-8"))
    s4 = json.loads((summ / "supp_round4.json").read_text(encoding="utf-8"))
    p5 = json.loads((summ / "p5dev_explore.json").read_text(encoding="utf-8")) if (summ / "p5dev_explore.json").exists() else {}
    out: dict[str, Any] = {"n_boot": nb, "note": "보조 통계. 'posthoc' 표시는 사전 등록 가설이 아닌 사후 분석이며 판정에 쓰지 않는다."}

    ex3, ex4 = load_expert(root, "expert_driving_test3.json"), load_expert(root, "expert_driving_test4.json")
    G3, G4 = load_runs(root, "driving__test3__*.json"), load_runs(root, "driving__test4__*.json")

    # r7·n4: 4단계 시나리오
    sc4 = scenario_rates(G4, ex4)
    out["scenario_stage4"] = sc4
    out["expert_stage4"] = sc4["expert_summary"]
    out["lowest_scenario_care"] = {
        "stage2": lowest_scenario(s4["scenario"]["stage2"]),
        "stage3": lowest_scenario(s4["scenario"]["stage3"]),
        "stage4": lowest_scenario(sc4),
    }
    out["vru_lowest_all_three"] = all(v.get("scenario") == "vru_crossing" for v in out["lowest_scenario_care"].values())
    scenario_table_4(sc4, tables / "supp5_scenario_4")
    logger.info("r7 시나리오(4단계 v4 CARE): %s", sc4.get("v4", {}).get("care"))

    # r8·n8: 4단계 K2 에피소드 대응 구간
    i4 = stage_kpi_inputs(G4, ex4)
    if i4:
        pk = pooled_kpis([i4], nb)
        out["k2_paired_stage4"] = pk["K2"]
        out["k2_independent_stage4"] = {k: t4["kpi"]["K2"][k] for k in ("value", "lo", "hi", "p_ge_goal")}
        logger.info("r8 K2 대응(4단계): %s", pk["K2"])

    # (사후) 4단계 차이의 차이
    out["ha_diff_stage4"] = diff_of_diffs(G4, ex4, nb) | {"posthoc": True}
    logger.info("4단계 차이의 차이: %s", out["ha_diff_stage4"])

    # r4·n2: 정책 효과 − 선별 효과(사후), 비
    out["policy_minus_selection"] = {"stage3": policy_minus_selection(G3, ex3, nb) | {"posthoc": True}, "stage4": policy_minus_selection(G4, ex4, nb) | {"posthoc": True}}
    for st, tk in (("stage3", t3), ("stage4", t4)):  # 사전 등록 분석의 H-v − H-a 점추정과 같은지 확인
        pm = out["policy_minus_selection"][st]
        pm["check_hv_minus_ha"] = float(tk["tests"]["H-v3"]["diff"] - tk["tests"]["H-a3"]["diff"])
    logger.info("정책 − 선별: %s", {k: (v["diff"], v["lo"], v["hi"]) for k, v in out["policy_minus_selection"].items()})
    hv4, ha4 = t4["tests"]["H-v3"]["diff"], t4["tests"]["H-a3"]["diff"]
    r4 = s4.get("ratio", {})
    ratio = {"stage4": hv4 / ha4, "stage3": r4.get("stage3"), "stage2_min": r4.get("stage2_min"), "stage2_max": r4.get("stage2_max")}
    vals = [x for x in (ratio["stage4"], ratio["stage3"], ratio["stage2_min"], ratio["stage2_max"]) if x is not None]
    ratio["min_234"], ratio["max_234"] = float(min(vals)), float(max(vals))
    out["ratio"] = ratio

    # r6: p5dev
    out["p5dev"] = p5dev_breakdown(root, p5)
    logger.info("r6 p5dev: %s", {k: out["p5dev"][k] for k in ("all_p7", "all_p7_p5")})

    # r3: 고유 정책
    out["unique_policies_stage4"] = unique_policies(root)
    logger.info("r3 정책 수: %s", {k: v for k, v in out["unique_policies_stage4"].items() if k != "by_condition"})

    # r2·n9: 표 48 한국어 판
    fourth_tests_table(t4, out["ha_diff_stage4"], out["policy_minus_selection"]["stage4"], tables / "supp5_fourth_tests")

    path = summ / "supp_round5.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    logger.info("저장: %s", path)


if __name__ == "__main__":
    main()
