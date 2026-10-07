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

from .vla_report import _T975, _fmt, _write_table, hierarchical_bootstrap, holm, seed_t
from .vla_v3_report import TARGETS, episode_success
from .vla_v4_suite import V4Config, expert_reference

logger = logging.getLogger(__name__)
V3_KPI_PATH = Path("experiments/exp_120_vla_v3/summary/v3_kpi.json")
LABELS = {"full": "전체(100%)", "random_shared": "무작위(공유 저장소)", "care": "CARE", "mix_trigger": "저장소 + 감속 트리거"}
DEV_NAMES = {"v3": "v3 기준", "t1": "+T1 지시문 드롭아웃", "all": "전부(P2+P3+T1+P4)", "all_p6": "전부+P6 수치 목표", "all_p7": "전부+P7 목표 속도 잔차"}
# 3차 심사 탐색적 절제(새 테스트 재사용, 확증 아님). 표 이름(한국어)과 원고 자리표시자용 영문 별칭
K6_CASES = [
    ("full_lang", "v4 전부+P7, 언어 있음", ("cf", "v4", "care", 0.02, True)),
    ("full_nolang", "v4 전부+P7, 언어 없음(P6·P7도 꺼짐)", ("cf", "v4", "care", 0.02, False)),
    ("p6_lang", "전부+P6(P7 없음), 언어 있음", ("cf", "all_p6", "care", 0.02, True)),
    ("p6_nolang", "전부+P6(P7 없음), 언어 없음", ("cf", "all_p6", "care", 0.02, False)),
    ("p67_only", "P6·P7만(학습 언어 경로 없음)", ("cf", "p67_nolangemb", "care", 0.02, True)),
    ("ctrl", "제어기 단독(학습 없음)", ("cf", "ctrl", "none", 0.0, True)),
]
K4_EXPLORE_LABELS = {
    "noaux_care_vs_mix_trigger": "P3·P4 제외 v4: CARE − 트리거 혼합",
    "noaux_care_vs_random_shared": "P3·P4 제외 v4: CARE − 무작위",
    "noaux_mix_trigger_vs_random_shared": "P3·P4 제외 v4: 트리거 혼합 − 무작위",
    "v4_mix_trigger_vs_random_shared": "v4: 트리거 혼합 − 무작위",
    "v3_mix_trigger_vs_random_shared": "v3 설정: 트리거 혼합 − 무작위",
    "v3_care_vs_mix_trigger": "v3 설정: CARE − 트리거 혼합",
}


def _t975(df: int) -> float:
    """t 분포 97.5% 분위수(표에 없는 자유도는 그보다 작은 가장 가까운 자유도의 값, 보수적)."""

    if df in _T975:
        return _T975[df]
    keys = [k for k in _T975 if k <= df]
    return _T975[max(keys)] if keys else float("nan")


def _seeds_for_halfwidth(sd: float, hw: float, n_max: int = 200) -> int:
    """시드 수준 대응 t 구간 반폭이 hw 미만이 되는 최소 시드 수(시드 변동만 고려한 하한)."""

    for n in range(2, n_max + 1):
        if _t975(n - 1) * sd / np.sqrt(n) < hw:
            return n
    return n_max


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


def _boot_mean(S: np.ndarray, mask: np.ndarray | None = None, n_boot: int = 10000, seed: int = 0) -> dict[str, float]:
    """단일 조건 성공률의 시드·에피소드 2단계 부트스트랩 95% CI(mask가 있으면 해당 에피소드만)."""

    rng = np.random.default_rng(seed)
    X = S if mask is None else S[:, mask]
    n_s, n_e = X.shape
    bs = np.empty(n_boot)
    for i in range(n_boot):
        si = rng.integers(0, n_s, n_s)
        ei = rng.integers(0, n_e, n_e)
        bs[i] = X[np.ix_(si, ei)].mean()
    return {"mean": float(X.mean()), "sd_seed": float(X.mean(1).std(ddof=1)) if n_s > 1 else 0.0, "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)), "boot": bs}


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
            # 3차 심사 m6·M2(3): 한국어 설정 이름, 감소율, 시드별 값(개발 반사실 실행 파일)
            dev_runs = _group(_runs(root, "driving__devcf__"))
            dl_rows = []
            for var in ("v3", "t1", "all", "all_p6", "all_p7"):
                if var not in c["devcf"]:
                    continue
                lv, nv = c["devcf"][var].get("lang", np.nan), c["devcf"][var].get("nolang", np.nan)
                red = 1.0 - lv / nv if nv else np.nan
                c["devcf"][var]["reduction"] = float(red)
                sl = {r["job"]["seed"]: r["closed_loop"]["overall"].get("speed_error") for r in dev_runs.get(("devcf", var, "care", 0.02, True), [])}
                sn = {r["job"]["seed"]: r["closed_loop"]["overall"].get("speed_error") for r in dev_runs.get(("devcf", var, "care", 0.02, False), [])}
                c["devcf"][var]["by_seed_lang"] = {str(k): float(v) for k, v in sorted(sl.items())}
                c["devcf"][var]["by_seed_nolang"] = {str(k): float(v) for k, v in sorted(sn.items())}
                seed_txt = " / ".join(f"{sl[k]:.2f}·{sn[k]:.2f}" for k in sorted(set(sl) & set(sn)))
                dl_rows.append([DEV_NAMES.get(var, var), _fmt(lv, digits=2), _fmt(nv, digits=2), f"{100 * red:.1f}%".replace("-", "−"), seed_txt or "-"])
            _write_table(tables / "v4_dev_lang", ["설정", "언어 있음(m/s)", "언어 없음(m/s)", "감소율", "시드별 언어 있음·없음(m/s)"], dl_rows)
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
                hz = _hazard(rs, ex, False)
                cond[(var, m, b)] = {"S": S, "success": _ms(S[0]), "hazard": (float(hz.mean()), float(hz.std(ddof=1)) if len(hz) > 1 else 0.0), "collision": float(np.mean([r["closed_loop"]["overall"]["collision_rate"] for r in rs])), "n": len(rs)}
        # 3차 심사 뒤 탐색적 절제(새 테스트 재사용) 행은 '(탐색)'으로 표시한다
        var_label = {"v4_noaux": "v4 P3·P4 제외(탐색)"}
        explore_rows = {("v3", "mix_trigger")}
        rows = [[var_label.get(var, var), LABELS.get(m, m) + (" (탐색)" if (var, m) in explore_rows else ""), f"{int(round(100 * b))}%", str(c["n"]), _fmt(*c["success"]), _fmt(*c["hazard"]), _fmt(c["collision"])] for (var, m, b), c in sorted(cond.items(), key=lambda kv: (kv[0][0], kv[0][2], kv[0][1]))]
        _write_table(tables / "v4_driving", ["설정", "방법", "예산", "시드", "성공률", "위험 시나리오 성공률", "충돌률"], rows)
        care, full = cond.get(("v4", "care", 0.02)), cond.get(("v4", "full", 1.0))
        hz_mask = np.array([bool(e["hazard"]) for e in G[("test", "v4", "care", 0.02, True)][0]["closed_loop"]["episodes"]]) if care else None
        if care:
            b1 = _boot_mean(care["S"][0])
            b3 = _boot_mean(care["S"][0], hz_mask)
            kpi["K1"] = {"value": care["success"][0], "sd": care["success"][1], "n_seeds": care["n"], "lo": b1["lo"], "hi": b1["hi"], "p_ge_goal": float((b1["boot"] >= 0.80).mean())}
            kpi["K3"] = {"value": care["hazard"][0], "sd": b3["sd_seed"], "lo": b3["lo"], "hi": b3["hi"], "p_ge_goal": float((b3["boot"] >= 0.75).mean()), "n_episodes": int(hz_mask.sum() * care["n"])}
            # 목표와의 차이(값, 에피소드 수 환산): 3차 심사 M5
            kpi["K3"]["margin"] = kpi["K3"]["value"] - 0.75
            kpi["K3"]["margin_episodes"] = int(round(kpi["K3"]["margin"] * kpi["K3"]["n_episodes"]))
        if care and full:
            bf = _boot_mean(full["S"][0], seed=1)
            ratio = b1["boot"] / bf["boot"]
            kpi["K2"] = {"value": care["success"][0] / full["success"][0], "care": care["success"][0], "full": full["success"][0], "lo": float(np.quantile(ratio, 0.025)), "hi": float(np.quantile(ratio, 0.975)), "p_ge_goal": float((ratio >= 0.90).mean())}
        # 탐색적: v4 시나리오별 성공률과 새 테스트 전문가 성공률(3차 심사 A1)
        scen = sorted({e["scenario"] for e in ex["episodes"]})
        srows = []
        for var, m, b in (("v4", "full", 1.0), ("v4", "care", 0.02), ("v4", "mix_trigger", 0.02), ("v4", "random_shared", 0.02), ("v3", "care", 0.02)):
            rs = G.get(("test", var, m, b, True))
            if not rs:
                continue
            per: dict[str, list[float]] = defaultdict(list)
            for r in rs:
                for e, ok in zip(r["closed_loop"]["episodes"], episode_success(r["closed_loop"]["episodes"], ex, False)):
                    per[e["scenario"]].append(float(ok))
            srows.append([var, LABELS.get(m, m), f"{int(round(100 * b))}%"] + [_fmt(float(np.mean(per[c]))) for c in scen])
            kpi.setdefault("driving_scenario", {}).setdefault(var, {})[m] = {c: float(np.mean(per[c])) for c in scen}
        ex_ok = {c: float(np.mean([not e["collision"] for e in ex["episodes"] if e["scenario"] == c])) for c in scen}
        srows.append(["전문가", "-", "-"] + [_fmt(ex_ok[c]) for c in scen])
        kpi.setdefault("driving_scenario", {})["expert"] = dict(ex_ok)
        scen_ko = {"cut_in": "끼어들기ʰ", "dense_traffic": "밀집", "follow": "추종", "free_drive": "정상", "lead_brake": "선행차 급제동ʰ", "stop_and_go": "정체ʰ", "vru_crossing": "보행자 횡단ʰ"}
        _write_table(tables / "v4_driving_scenario", ["설정", "방법", "예산"] + [scen_ko.get(c, c) for c in scen], srows)
        kpi["expert_test_success"] = {
            "driving": float(np.mean([not e["collision"] for e in ex["episodes"]])),
            "hazard": float(np.mean([not e["collision"] for e in ex["episodes"] if e["hazard"]])),
            "vru_crossing": ex_ok.get("vru_crossing", float("nan")),
        }
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
        # 정책 개선 효과(H-v, v4 무작위 − v3 무작위) / 선별 효과(같은 정책 안 CARE − 무작위)의 배율 범위(3차 심사 m1)
        pol = [comps[k]["diff"] for k in ("v4_vs_v3_care", "v4_vs_v3_random_shared") if k in comps]
        sel = [comps[k]["diff"] for k in ("v3_care_vs_random_shared", "care_vs_random_shared") if k in comps]
        if pol and sel and min(sel) > 0:
            ratios = [a / b for a in pol for b in sel]
            kpi["ratios"] = {"policy_over_selection_min": float(min(ratios)), "policy_over_selection_max": float(max(ratios))}
        # 원고 자리표시자용 조건별 값: driving:<설정>:<방법>:<success|success_sd|hazard|collision|n>
        drv: dict[str, dict[str, Any]] = defaultdict(dict)
        for (var, m, b), c in cond.items():
            drv[var][m] = {"success": c["success"][0], "success_sd": c["success"][1], "hazard": c["hazard"][0], "hazard_sd": c["hazard"][1], "collision": c["collision"], "n": c["n"], "budget": b}
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
    excf = expert_reference(cfg, "cf") if (root / "cache" / "expert_driving_cf.json").exists() else None

    def cf_stats(rs: list[dict[str, Any]]) -> dict[str, Any]:
        """반사실 실행들의 속도 오차·성공률·자유주행 프레임 수·시드별 오차(3차 심사 m3·m4)."""

        se = {r["job"]["seed"]: (r["closed_loop"]["overall"].get("speed_error") or np.nan) for r in rs}
        free = [float(np.mean([e.get("n_free_frames") or 0 for e in r["closed_loop"]["episodes"]])) for r in rs]
        out = {"speed_error": float(np.nanmean(list(se.values()))), "by_seed": {str(k): float(v) for k, v in sorted(se.items())}, "n": len(rs), "free_frames": float(np.mean(free))}
        if excf is not None:
            out["success"] = float(_S(rs, excf, False)[0].mean())
        return out

    if excf is not None:
        k6 = {}
        rows = []
        for m, b in (("care", 0.02), ("full", 1.0)):
            vals = {}
            for lang in (True, False):
                rs = G.get(("cf", "v4", m, b, lang))
                if not rs:
                    continue
                sep = []
                for r in rs:
                    hw: dict[tuple, dict[str, float]] = defaultdict(dict)
                    for e in r["closed_loop"]["episodes"]:
                        if e.get("headway_mean") is not None:
                            hw[(e["scenario"], e["seed"])][e["style"]] = e["headway_mean"]
                    d = [v["cautious"] - v["brisk"] for v in hw.values() if "cautious" in v and "brisk" in v]
                    sep.append(float(np.mean(d)) if d else np.nan)
                st = cf_stats(rs)
                vals[lang] = {"speed_error": st["speed_error"], "style_sep": float(np.nanmean(sep)), "n": len(rs), "success": st.get("success", float("nan")), "free_frames": st["free_frames"]}
                rows.append([LABELS[m], "있음" if lang else "없음", str(len(rs)), _fmt(vals[lang]["success"]), _fmt(vals[lang]["speed_error"], digits=2), _fmt(vals[lang]["free_frames"], digits=0), _fmt(vals[lang]["style_sep"], digits=2)])
            if True in vals and False in vals:
                k6[m] = {"lang": vals[True], "nolang": vals[False], "speed_error_reduction": 1.0 - vals[True]["speed_error"] / vals[False]["speed_error"]}
                # 시드별 감소율과 시드 부트스트랩 CI(3차 심사 M2). K6 자체는 '평균의 비'(3.7절)이며, 시드별 값은 보조다
                sl = {r["job"]["seed"]: r["closed_loop"]["overall"].get("speed_error") for r in G[("cf", "v4", m, b, True)]}
                sn = {r["job"]["seed"]: r["closed_loop"]["overall"].get("speed_error") for r in G[("cf", "v4", m, b, False)]}
                common = sorted(set(sl) & set(sn))
                L, N = np.array([sl[c] for c in common], float), np.array([sn[c] for c in common], float)
                k6[m]["by_seed"] = {str(c): float(1 - sl[c] / sn[c]) for c in common}
                rng = np.random.default_rng(0)
                bs = []
                for _ in range(10000):
                    ii = rng.integers(0, len(common), len(common))
                    bs.append(1 - L[ii].mean() / N[ii].mean())
                k6[m]["lo"], k6[m]["hi"] = float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
                k6[m]["n_seeds_reduced"] = int((L < N).sum())
                k6[m]["sd_seed_reduction"] = float(np.std([1 - a / c for a, c in zip(L, N)], ddof=1)) if len(common) > 1 else 0.0
                k6[m]["mean_of_seed_reductions"] = float(np.mean([1 - a / c for a, c in zip(L, N)]))
        hw_ex = defaultdict(dict)
        for e in excf["episodes"]:
            if e.get("headway_mean") is not None:
                hw_ex[(e["scenario"], e["seed"])][e["style"]] = e["headway_mean"]
        ex_sep = [v["cautious"] - v["brisk"] for v in hw_ex.values() if "cautious" in v and "brisk" in v]
        rows.append(["전문가", "-", "-", "1.000", _fmt(excf["overall"].get("speed_error") or np.nan, digits=2), _fmt(float(np.mean([e.get("n_free_frames") or 0 for e in excf["episodes"]])), digits=0), _fmt(float(np.mean(ex_sep)) if ex_sep else np.nan, digits=2)])
        _write_table(tables / "v4_language_cf", ["데이터", "언어", "시드", "성공률", "속도 오차(m/s)", "자유주행 프레임(에피소드 평균)", "신중−민첩 추종 간격 차(s)"], rows)
        if "care" in k6:
            kpi["K6"] = k6["care"] | {"by_data": k6}
        kpi["expert_cf"] = {"speed_error": float(excf["overall"].get("speed_error") or np.nan)}

    # ---------- 탐색적 절제(3차 심사 M2·M3, 새 테스트는 이미 보았으므로 확증 아님) ----------
    explore: dict[str, Any] = {}
    if excf is not None:
        erows = []
        ref = G.get(("cf", "v4", "care", 0.02, False))
        ref_seed = {r["job"]["seed"]: r["closed_loop"]["overall"].get("speed_error") for r in ref} if ref else {}
        for alias, name, key in K6_CASES:
            rs = G.get(key)
            if not rs:
                continue
            st = cf_stats(rs)
            se_seed = st["by_seed"]
            common = sorted(set(int(k) for k in se_seed) & set(ref_seed))
            red = {str(c): float(1 - se_seed[str(c)] / ref_seed[c]) for c in common}
            if ref_seed:
                st["reduction_vs_nolang"] = float(1 - st["speed_error"] / float(np.mean(list(ref_seed.values()))))
            if len(common) > 1:
                Lr = np.array([se_seed[str(c)] for c in common])
                Nr = np.array([ref_seed[c] for c in common])
                rng = np.random.default_rng(0)
                bs = [1 - Lr[ii].mean() / Nr[ii].mean() for ii in (rng.integers(0, len(common), len(common)) for _ in range(10000))]
                st["reduction_lo"], st["reduction_hi"] = float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))
                st["reduction_by_seed"] = red
                st["n_seeds_reduced"] = int((Lr < Nr).sum())
            vals = list(se_seed.values())
            st["min"], st["max"] = float(np.min(vals)), float(np.max(vals))
            st["name"] = name
            explore.setdefault("k6", {})[name] = {"speed_error": st["speed_error"], "n": st["n"]}
            explore.setdefault("k6_alias", {})[alias] = st
            red_txt = f"{100 * st['reduction_vs_nolang']:.1f}%".replace("-", "−") if "reduction_vs_nolang" in st else "-"
            if alias == "full_nolang":
                red_txt = "(기준)"
            elif alias == "p6_nolang":
                red_txt = "(기준과 같은 정책: 언어가 없으면 P6·P7이 꺼진다)"
            elif "reduction_lo" in st:
                red_txt += f" [{100 * st['reduction_lo']:.1f}, {100 * st['reduction_hi']:.1f}]".replace("-", "−") + f", {st['n_seeds_reduced']}/{len(common)}"
            rng_txt = f"{st['min']:.2f}~{st['max']:.2f}" if st["n"] > 1 else "-"
            erows.append([name, str(st["n"]), _fmt(st["speed_error"], digits=2), rng_txt, red_txt, _fmt(st.get("success", float("nan"))), _fmt(st["free_frames"], digits=0)])
        # 학습 언어 경로의 몫: v4(전부+P7, 언어 있음) − P6·P7만(같은 시드 대응, 속도 오차 차이)
        a, b = explore.get("k6_alias", {}).get("full_lang"), explore.get("k6_alias", {}).get("p67_only")
        if a and b:
            cs = sorted(set(a["by_seed"]) & set(b["by_seed"]))
            dd = np.array([a["by_seed"][c] - b["by_seed"][c] for c in cs])
            rng = np.random.default_rng(0)
            bs = [dd[rng.integers(0, len(dd), len(dd))].mean() for _ in range(10000)]
            explore["k6_learned_lang_path"] = {"diff": float(dd.mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)), "n_seeds": len(cs), "n_seeds_worse": int((dd > 0).sum())}
        if erows:
            _write_table(tables / "v4_explore_k6", ["조건(CARE 2%, 반사실 새 테스트, 탐색적)", "실행", "속도 오차(m/s)", "시드별 범위(m/s)", "v4 언어 없음 대비 감소율 [시드 부트스트랩 95% CI], 감소 시드", "성공률", "자유주행 프레임(에피소드 평균)"], erows)
    if (root / "cache" / "expert_driving_test.json").exists() and any(k[1] == "v4_noaux" for k in G):
        ex = expert_reference(cfg, "test")
        ec = {(var, m): _S(rs, ex, False) for (ev, var, m, b, lang), rs in G.items() if ev == "test" and lang and b == 0.02}
        pairs = {
            "noaux_care_vs_mix_trigger": (("v4_noaux", "care"), ("v4_noaux", "mix_trigger")),
            "noaux_care_vs_random_shared": (("v4_noaux", "care"), ("v4_noaux", "random_shared")),
            "noaux_mix_trigger_vs_random_shared": (("v4_noaux", "mix_trigger"), ("v4_noaux", "random_shared")),
            "v4_mix_trigger_vs_random_shared": (("v4", "mix_trigger"), ("v4", "random_shared")),
            "v3_mix_trigger_vs_random_shared": (("v3", "mix_trigger"), ("v3", "random_shared")),
            "v3_care_vs_mix_trigger": (("v3", "care"), ("v3", "mix_trigger")),
        }
        erows = []
        for name, (a, b) in pairs.items():
            if a in ec and b in ec:
                A, B = _pair(ec[a], ec[b])
                hb = hierarchical_bootstrap(A, B)
                explore.setdefault("k4", {})[name] = hb | {"n_seeds": len(A), "a": float(A.mean()), "b": float(B.mean())}
                erows.append([K4_EXPLORE_LABELS.get(name, name), str(len(A)), _fmt(float(A.mean())), _fmt(float(B.mean())), f"{hb['diff']:+.3f} [{hb['lo']:+.3f}, {hb['hi']:+.3f}]".replace("-", "−"), f"{hb['p_le0']:.3f}"])
        if erows:
            _write_table(tables / "v4_explore_k4", ["비교(새 테스트 재사용, 2%, 탐색적)", "시드", "A 성공률", "B 성공률", "A − B [계층 부트스트랩 95% CI]", "단측 p(보정 없음)"], erows)
    kpi["explore"] = explore

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
            # 3차 심사 M7: 시드별 차이와 검정력 점검(시드 변동만 고려한 필요 시드 수는 하한이다)
            dsd = A.mean(1) - B.mean(1)
            sd = float(dsd.std(ddof=1)) if len(dsd) > 1 else float("nan")
            kpi["K5"] |= {"by_seed": {str(sd_): float(x) for sd_, x in zip(sorted(set(rcond[("v4", "care", 0.02)][1]) & set(rcond[("v4", "random_shared", 0.02)][1])), dsd)}, "sd_seed_diff": sd, "halfwidth": float((kpi["K5"]["hi"] - kpi["K5"]["lo"]) / 2), "seeds_for_halfwidth_0.03": _seeds_for_halfwidth(sd, 0.03) if np.isfinite(sd) else None}
            v3k5 = (json.loads(V3_KPI_PATH.read_text(encoding="utf-8")) if V3_KPI_PATH.exists() else {}).get("K5")
            if v3k5:
                kpi["K5"]["v3_halfwidth"] = float((v3k5["hi"] - v3k5["lo"]) / 2)
        if ("v4", "full", 1.0) in rcond:
            kpi["robot_full_success"] = float(rcond[("v4", "full", 1.0)][0].mean())
        kpi.setdefault("expert_test_success", {})["robot"] = float(episode_success(exr["episodes"], exr, True).mean())
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

    # ---------- v3 원자료 진단 수치(3차 심사 M12: 원고의 손 수치를 결과 파일 키로) ----------
    try:
        kpi["v3_diag"] = _v3_diagnosis(cfg)
    except Exception as exc:  # 진단 재계산 실패가 집계를 막지 않게 한다
        logger.warning("v3 진단 재계산 실패: %s", exc)

    # ---------- KPI 요약(v2·v3·v4) ----------
    v3k = json.loads(V3_KPI_PATH.read_text(encoding="utf-8")) if V3_KPI_PATH.exists() else {}

    def judge(k: str, v: dict[str, Any] | None) -> str:
        if not v:
            return "미측정"
        rule = {"K1": lambda x: x["value"] >= 0.80, "K2": lambda x: x["value"] >= 0.90, "K3": lambda x: x["value"] >= 0.75, "K4": lambda x: x["lo"] > 0 and x.get("p_holm", x.get("p_le0", 1.0)) < 0.05, "K5": lambda x: x["lo"] > -0.03, "K6": lambda x: x["speed_error_reduction"] >= 0.30, "K7": lambda x: x["value"] >= 0.65}[k]
        return "달성" if rule(v) else "미달"

    def show(k: str, v: dict[str, Any] | None) -> str:
        if not v:
            return "-"
        if k in ("K4", "K5"):
            return f"{v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]".replace("-", "−")
        if k == "K6":
            txt = f"{100 * v['speed_error_reduction']:.1f}%"
            if "lo" in v:  # 시드 부트스트랩 95% CI(3차 심사 M2, 보조)
                txt += f" [{100 * v['lo']:.1f}, {100 * v['hi']:.1f}]"
            return txt.replace("-", "−")
        if "lo" in v:  # 시드·에피소드 부트스트랩 95% CI(3차 심사 M5·m2, 보조)
            return f"{v['value']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]"
        return f"{v['value']:.3f}"

    rows = [[k, TARGETS[k][0], TARGETS[k][1], show(k, v3k.get(k)) + f" ({judge(k, v3k.get(k))})", show(k, kpi.get(k)), judge(k, kpi.get(k))] for k in TARGETS]
    _write_table(tables / "v4_kpi", ["ID", "지표", "목표", "v3(800000번대 테스트)", "v4(새 테스트 1000000번대) [95% CI]", "판정"], rows)
    kpi["judgement"] = {k: judge(k, kpi.get(k)) for k in TARGETS}
    try:
        _figure(root, kpi, v3k)
    except Exception as exc:  # 그림 실패가 집계를 막지 않게 한다
        logger.warning("v4 그림 생성 실패: %s", exc)
    (root / "summary").mkdir(parents=True, exist_ok=True)
    (root / "summary" / "v4_kpi.json").write_text(json.dumps(kpi, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    logger.info("v4 KPI: %s", kpi["judgement"])
    return kpi


def _v3_diagnosis(cfg: V4Config) -> dict[str, Any]:
    """1단계(v3) 원자료에서 v4 진단에 쓴 수치를 다시 계산한다(3차 심사 M12, docs/36 2.0~2.1절의 값).

    - AMR 전체 데이터 정책(시드 0~2)의 시나리오별 주행 중 충돌·진행 미달 횟수, 시드 0의 평균 속도, 개루프 MAE
    - v3 반사실 CARE 2%의 스타일별 속도 오차·에피소드 평균 속도·목표 속도
    - 실영상 CARE 점수기의 제동 시작 AUROC(fold별)와 −a_t 기준선(정책 실행과 같은 시험 프레임·onset 정의)
    - S1 측정(점수 몫 에피소드 수·위험 비중): `experiments/exp_121_v4_a15_obs_curation/summary/a15_measure.json`
    """

    from .metrics import auroc

    v3r = cfg.v3_root
    out: dict[str, Any] = {}
    exr = json.loads((v3r / "cache" / "expert_robot_test.json").read_text(encoding="utf-8"))
    exd = {(e["scenario"], e["seed"], e["style"]): e for e in exr["episodes"]}
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((v3r / "runs").glob("robot__test__full__b1.00__s*.json"))]
    amr: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "collision_moving": 0, "progress_fail": 0})
    speed0: dict[str, list[float]] = defaultdict(list)
    for r in runs:
        for e in r["closed_loop"]["episodes"]:
            x = exd[(e["scenario"], e["seed"], e["style"])]
            a = amr[e["scenario"]]
            a["n"] += 1
            a["collision_moving"] += int(bool(e["collision_moving"]))
            a["progress_fail"] += int((not e["collision_moving"]) and x["distance"] >= 1.0 and e["distance"] < 0.8 * x["distance"])
            if r["job"]["seed"] == 0:
                speed0[e["scenario"]].append(float(e["mean_speed"]))
    for sc, a in amr.items():
        a["policy_speed_seed0"] = float(np.mean(speed0[sc])) if speed0[sc] else float("nan")
        a["expert_speed"] = float(np.mean([x["mean_speed"] for k, x in exd.items() if k[0] == sc]))
    out["amr_full"] = dict(amr)
    out["amr_full"]["n_seeds"] = len(runs)
    r0 = next((r for r in runs if r["job"]["seed"] == 0), None)
    if r0:
        ol = r0["open_loop"]
        out["amr_open_loop_seed0"] = {"mae": ol["mae"], "mae_hazard": ol["mae_hazard"], "n_hazard": ol["n_hazard"], "n_frames": ol["n_frames"], "hazard_frac": ol["n_hazard"] / ol["n_frames"], "ratio": ol["mae_hazard"] / ol["mae"]}
    sty: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for p in sorted((v3r / "runs").glob("driving__cf__care__b0.02__s*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        lang = "lang" if r["job"].get("use_language", True) else "nolang"
        for e in r["closed_loop"]["episodes"]:
            d = sty[f"{lang}_{e['style']}"]
            d["mean_speed"].append(float(e["mean_speed"]))
            d["v_target"].append(float(e["v_target"]))
            if e.get("speed_error") is not None:
                d["speed_error"].append(float(e["speed_error"]))
    out["cf_care_by_style"] = {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in sty.items()}
    # 실영상 점수기 AUROC(comma_curation.run_comma_job과 같은 onset 정의)
    tbl = Path(cfg.comma_dir) / "comma_table_10fps.npz"
    if tbl.exists():
        from .comma_curation import BRAKING, CHUNK, _segments

        d = np.load(tbl)
        y, acc = d["y"], d["accel"]
        n = len(y)
        oracle = y == BRAKING
        sc_au, base_au = [], []
        for k in range(5):
            fp = v3r / "cache" / f"comma_fold{k}.npz"
            if not fp.exists():
                continue
            F = np.load(fp)
            seg = _segments(F["test_mask"])
            ti = np.where(seg >= 0)[0]
            g = np.where(seg >= 0, seg, -1 - np.arange(n))
            fb = np.zeros(len(ti), dtype=bool)
            for j in range(1, CHUNK + 1):
                nx = np.minimum(ti + j, n - 1)
                fb |= oracle[nx] & (g[nx] == g[ti])
            om = ~oracle[ti]
            sc_au.append(float(auroc(F["event_score"][ti][om], fb[om])))
            base_au.append(float(auroc(-acc[ti][om], fb[om])))
        if sc_au:
            out["comma_scorer_auroc"] = {"mean": float(np.mean(sc_au)), "min": float(np.min(sc_au)), "max": float(np.max(sc_au)), "by_fold": sc_au}
            out["comma_neg_accel_auroc"] = {"mean": float(np.mean(base_au)), "by_fold": base_au}
    a15 = Path("experiments/exp_121_v4_a15_obs_curation/summary/a15_measure.json")
    if a15.exists():
        rows = [r for r in json.loads(a15.read_text(encoding="utf-8"))["cap"]["rows"] if abs(r["budget"] - 0.02) < 1e-9]
        s1: dict[str, Any] = {}
        for m in ("care", "mix_trigger"):
            for cap, tag in ((None, "nocap"), (1, "cap1")):
                rr = [r for r in rows if r["method"] == m and r["per_group_cap"] == cap]
                if rr:
                    s1[f"{m}_{tag}"] = {
                        "score_hazard_frac_min": float(min(r["score_hazard_frac"] for r in rr)),
                        "score_hazard_frac_max": float(max(r["score_hazard_frac"] for r in rr)),
                        "score_hazard_frac_mean": float(np.mean([r["score_hazard_frac"] for r in rr])),
                        "n_score_episodes_min": int(min(r["n_score_episodes"] for r in rr)),
                        "n_score_episodes_max": int(max(r["n_score_episodes"] for r in rr)),
                        "n_seeds": len(rr),
                    }
        out["s1"] = s1
    return out


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
    # 3차 심사 M9: 같은 새 테스트의 v3 설정 값(K1·K3)을 함께 그려, 같은 세트 비교와 교차 세트 비교를 구분한다
    v3new = {k: float("nan") for k in ks}
    c3 = (kpi.get("driving") or {}).get("v3", {}).get("care")
    if c3:
        v3new["K1"], v3new["K3"] = c3["success"], c3["hazard"]
    w = 0.27
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x - w, [val(v3k, k) for k in ks], w, label="v3 설정, v3 테스트(800000번대, 교차 세트)", color="#cbd2d9", hatch="//", edgecolor="#7b8794")
    ax.bar(x, [v3new[k] for k in ks], w, label="v3 설정, 새 테스트(같은 세트, K1·K3만)", color="#7b8794")
    ax.bar(x + w, [val(kpi, k) for k in ks], w, label="v4 설정, 새 테스트", color="#2b6cb0")
    for i, k in enumerate(ks):
        ax.plot([i - 0.45, i + 0.45], [goals[k]] * 2, color="#c53030", ls="--", lw=1.5, label="목표" if i == 0 else None)
    ax.axhline(0, color="k", lw=0.6)
    short = {"K1": "CARE 2% 성공률", "K2": "데이터 효율", "K3": "위험 시나리오", "K6": "언어 오차 감소율", "K7": "실영상 AUROC"}
    ax.set_xticks(x, [f"{k}\n{short[k]}" for k in ks], fontsize=8)
    ax.set_ylabel("값(K6는 감소율)")
    ax.legend(fontsize=7, loc="lower right")
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
