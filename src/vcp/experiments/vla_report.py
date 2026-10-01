from __future__ import annotations

"""VLA 연계 큐레이션 실험 결과 집계·통계·그림.

입력: `<root>/summary/curation_runs.json`(write_summary 결과), `<root>/cache/expert_*.json`, comma 실행 JSON.
출력(`<root>/summary/`)
- `vla_stats.json`: 원고 자리표시자에 쓰는 이름 붙은 수치
- `tables/*.csv`, `tables/*.md`: 표
- `figures/*.png`: 그림

통계 원칙
- 같은 평가 시나리오(시드)를 모든 조건이 공유하므로 에피소드 단위 **대응(paired) 부트스트랩**으로
  방법 간 성공률 차이의 95% 신뢰구간을 구한다(시드 평균 후 에피소드 재표집, 4000회).
- 표의 ± 는 시드 간 표준편차다.
"""

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .vla_curation_suite import CurationSuiteConfig, episode_success, expert_reference

logger = logging.getLogger(__name__)

METHOD_LABELS = {
    "full": "전체(100%)",
    "random": "무작위",
    "uniform": "등간격",
    "action_trigger": "감속 트리거",
    "rule_ittc": "역 TTC 규칙",
    "uncertainty": "불확실성",
    "event": "맥락 이벤트",
    "event_sim": "맥락 이벤트(시뮬 점수기)",
    "coreset": "Coreset",
    "offline_loss": "오프라인 손실*",
    "oracle": "오라클*",
    "ours": "제안(CARE)",
}
EDGE_OK = {"random", "uniform", "action_trigger", "rule_ittc", "uncertainty", "event", "event_sim", "ours"}
METHOD_ORDER = ["random", "uniform", "action_trigger", "rule_ittc", "uncertainty", "coreset", "event", "ours", "offline_loss", "oracle"]


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def _fmt(m: float, s: float | None = None, digits: int = 3) -> str:
    if not np.isfinite(m):
        return "-"
    if s is None or not np.isfinite(s):
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def load_runs(root: Path) -> list[dict[str, Any]]:
    return json.loads((root / "summary" / "curation_runs.json").read_text(encoding="utf-8"))


def condition_key(job: dict[str, Any]) -> tuple:
    return (
        job["domain"],
        job["eval"],
        job["method"],
        round(float(job["budget"]), 2),
        int(job["steps"]),
        bool(job.get("use_language", True)),
        job.get("lam"),
        job.get("reservoir"),
    )


def group_runs(runs: list[dict[str, Any]]) -> dict[tuple, list[dict[str, Any]]]:
    out: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        out[condition_key(r["job"])].append(r)
    for v in out.values():
        v.sort(key=lambda r: r["job"]["seed"])
    return out


def success_matrix(rs: list[dict[str, Any]], expert: dict[str, Any], moving: bool) -> np.ndarray:
    """[시드, 에피소드] 성공 행렬(에피소드 순서는 평가 사양 순서로 같다)."""

    return np.stack([episode_success(r["closed_loop_episodes"], expert, moving) for r in rs]).astype(float)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 4000, seed: int = 0) -> dict[str, float]:
    """에피소드 대응 부트스트랩: mean(a) - mean(b)의 95% CI. a, b: [시드, 에피소드]."""

    da = a.mean(0) - b.mean(0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(da), size=(n_boot, len(da)))
    boots = da[idx].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": float(da.mean()), "lo": float(lo), "hi": float(hi), "p_le0": float((boots <= 0).mean())}


def summarize_condition(rs: list[dict[str, Any]], expert: dict[str, Any], moving: bool) -> dict[str, Any]:
    succ = success_matrix(rs, expert, moving)
    cl = [r["closed_loop"]["overall"] for r in rs]
    ol = [r.get("open_loop", {}) for r in rs]
    sel = [r["selection"] for r in rs]

    def ms(vals: list[Any]) -> tuple[float, float]:
        return _mean_std([float(v) if v is not None else np.nan for v in vals])

    hz = []
    for r in rs:
        eps = r["closed_loop_episodes"]
        s = episode_success(eps, expert, moving)
        mask = np.array([bool(e["hazard"]) for e in eps])
        hz.append(float(s[mask].mean()) if mask.any() else np.nan)
    return {
        "n_seeds": len(rs),
        "success": ms(list(succ.mean(1))),
        "hazard_success": ms(hz),
        "collision": ms([c["collision_rate"] for c in cl]),
        "collision_moving": ms([c.get("collision_moving_rate") for c in cl]),
        "speed_error": ms([c.get("speed_error") for c in cl]),
        "headway_error": ms([c.get("headway_error") for c in cl]),
        "rms_jerk": ms([c.get("rms_jerk") for c in cl]),
        "progress": ms([c.get("progress") for c in cl]),
        "min_ttc": ms([c.get("min_ttc_mean") for c in cl]),
        "mae": ms([o.get("mae") for o in ol]),
        "mae_hazard": ms([o.get("mae_hazard") for o in ol]),
        "mae_hard_brake": ms([o.get("mae_hard_brake") for o in ol]),
        "hazard_frame_recall": ms([s.get("hazard_frame_recall") for s in sel]),
        "hazard_event_recall": ms([s.get("hazard_event_recall") for s in sel]),
        "label_entropy_norm": ms([s.get("label_entropy_norm") for s in sel]),
        "group_coverage": ms([s.get("group_coverage") for s in sel]),
        "n_train_frames": rs[0]["n_train_frames"],
        "_succ": succ,
    }


def _write_table(path_stem: Path, header: list[str], rows: list[list[str]]) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    with (path_stem.with_suffix(".csv")).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    path_stem.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")


def build_report(root: Path, steps: int | None = None) -> dict[str, Any]:
    """주 실험·언어 절제·로봇·파일럿 집계. comma는 build_comma_report에서."""

    runs = load_runs(root)
    groups = group_runs(runs)
    out_dir = root / "summary"
    tables = out_dir / "tables"
    stats: dict[str, Any] = {}
    cfgs = {d: CurationSuiteConfig(root=root, domain=d) for d in ("driving", "robot")}
    from dataclasses import replace

    cfgs["robot"] = replace(cfgs["robot"], test_per_cell=4)
    experts: dict[tuple[str, str], dict[str, Any]] = {}

    def expert(domain: str, ev: str) -> dict[str, Any]:
        if (domain, ev) not in experts:
            experts[(domain, ev)] = expert_reference(cfgs[domain], ev)
        return experts[(domain, ev)]

    if steps is None:
        cand = [k[4] for k in groups if k[1] == "test" and k[0] == "driving"]
        steps = max(set(cand), key=cand.count) if cand else 3000
    stats["steps"] = steps

    # ---------------- 주 실험(주행, 테스트) ----------------
    summ: dict[tuple[str, float], dict[str, Any]] = {}
    for k, rs in groups.items():
        dom, ev, m, b, st, lang, lam, res = k
        if ev != "test" or st != steps or not lang or lam is not None:
            continue
        summ[(dom, m, b)] = summarize_condition(rs, expert(dom, "test"), dom == "robot")
    for dom in ("driving", "robot"):
        ex = expert(dom, "test")
        mv = dom == "robot"
        stats[f"{dom}_expert_success"] = float(episode_success(ex["episodes"], ex, mv).mean())
        stats[f"{dom}_expert_collision"] = float(ex["overall"]["collision_rate"])
        stats[f"{dom}_n_test"] = len(ex["episodes"])
    budgets = sorted({b for (d, m, b) in summ if d == "driving" and m != "full"})
    stats["budgets"] = budgets
    header = ["방법", "엣지 가능", "예산", "성공률", "위험 시나리오 성공률", "충돌률", "속도 오차(m/s)", "headway 오차(s)", "RMS jerk", "개루프 MAE", "위험 MAE", "위험 프레임 회수율"]
    rows = []
    full = summ.get(("driving", "full", 1.0))
    for b in budgets:
        for m in METHOD_ORDER:
            s = summ.get(("driving", m, b))
            if s is None:
                continue
            rows.append(
                [
                    METHOD_LABELS[m],
                    "O" if m in EDGE_OK else "X",
                    f"{int(round(b * 100))}%",
                    _fmt(*s["success"]),
                    _fmt(*s["hazard_success"]),
                    _fmt(*s["collision"]),
                    _fmt(*s["speed_error"], digits=2),
                    _fmt(*s["headway_error"], digits=2),
                    _fmt(*s["rms_jerk"], digits=2),
                    _fmt(*s["mae"], digits=3),
                    _fmt(*s["mae_hazard"], digits=3),
                    _fmt(*s["hazard_frame_recall"]),
                ]
            )
    if full:
        rows.append([METHOD_LABELS["full"], "-", "100%", _fmt(*full["success"]), _fmt(*full["hazard_success"]), _fmt(*full["collision"]), _fmt(*full["speed_error"], digits=2), _fmt(*full["headway_error"], digits=2), _fmt(*full["rms_jerk"], digits=2), _fmt(*full["mae"]), _fmt(*full["mae_hazard"]), "1.000"])
    _write_table(tables / "vla_main_driving", header, rows)

    # 대응 부트스트랩: 각 방법 vs 무작위(같은 예산), ours vs 각 방법
    boot: dict[str, Any] = {}
    for b in budgets:
        base = summ.get(("driving", "random", b))
        ours = summ.get(("driving", "ours", b))
        for m in METHOD_ORDER:
            s = summ.get(("driving", m, b))
            if s is None or base is None:
                continue
            boot[f"{m}_vs_random_b{b:.2f}"] = paired_bootstrap(s["_succ"], base["_succ"])
            if ours is not None and m != "ours":
                boot[f"ours_vs_{m}_b{b:.2f}"] = paired_bootstrap(ours["_succ"], s["_succ"])
        if ours is not None and full is not None:
            boot[f"ours_vs_full_b{b:.2f}"] = paired_bootstrap(ours["_succ"], full["_succ"])
    stats["bootstrap"] = boot
    for (dom, m, b), s in summ.items():
        for key in ("success", "hazard_success", "collision", "collision_moving", "speed_error", "headway_error", "rms_jerk", "mae", "mae_hazard", "hazard_frame_recall", "hazard_event_recall", "label_entropy_norm", "progress"):
            stats[f"{dom}:{m}:{b:.2f}:{key}"] = s[key][0]
            stats[f"{dom}:{m}:{b:.2f}:{key}_sd"] = s[key][1]
        stats[f"{dom}:{m}:{b:.2f}:n_train_frames"] = s["n_train_frames"]

    # 시나리오별 성공률(주 예산 10%)
    main_b = 0.10 if 0.10 in budgets else (budgets[len(budgets) // 2] if budgets else None)
    stats["main_budget"] = main_b
    scen_rows = []
    if main_b is not None:
        ex = expert("driving", "test")
        scen_names = sorted({e["scenario"] for e in ex["episodes"]})
        for m in METHOD_ORDER + ["full"]:
            key = ("driving", m, 1.0 if m == "full" else main_b)
            if key not in groups and (key[0], "test", key[1], key[2], steps, True, None, None) not in groups:
                pass
            rs = groups.get(("driving", "test", m, 1.0 if m == "full" else main_b, steps, True, None, None))
            if not rs:
                continue
            row = [METHOD_LABELS[m]]
            for sc in scen_names:
                vals = []
                for r in rs:
                    eps = r["closed_loop_episodes"]
                    succ = episode_success(eps, ex, False)
                    mask = np.array([e["scenario"] == sc for e in eps])
                    vals.append(float(succ[mask].mean()))
                row.append(f"{np.mean(vals):.2f}")
                stats[f"driving:{m}:scen:{sc}"] = float(np.mean(vals))
            scen_rows.append(row)
        _write_table(tables / "vla_scenarios_driving", ["방법"] + scen_names, scen_rows)

    # ---------------- 언어 절제 ----------------
    lang_rows = []
    for m, b in (("full", 1.0), ("ours", main_b), ("random", main_b)):
        for lang in (True, False):
            rs = groups.get(("driving", "test", m, b, steps, lang, None, None))
            if not rs:
                continue
            s = summarize_condition(rs, expert("driving", "test"), False)
            by_style: dict[str, list[float]] = defaultdict(list)
            hw: dict[str, list[float]] = defaultdict(list)
            for r in rs:
                for st_name, agg in r["closed_loop"].get("by_style", {}).items():
                    by_style[st_name].append(agg.get("speed_error") or np.nan)
                    hw[st_name].append(agg.get("headway_mean") or np.nan)
            tag = "lang" if lang else "nolang"
            stats[f"lang:{m}:{tag}:success"] = s["success"][0]
            stats[f"lang:{m}:{tag}:speed_error"] = s["speed_error"][0]
            stats[f"lang:{m}:{tag}:headway_error"] = s["headway_error"][0]
            for st_name in ("cautious", "normal", "brisk"):
                stats[f"lang:{m}:{tag}:headway_mean:{st_name}"] = float(np.nanmean(hw[st_name])) if hw[st_name] else float("nan")
            lang_rows.append(
                [
                    METHOD_LABELS[m],
                    "있음" if lang else "없음(절제)",
                    _fmt(*s["success"]),
                    _fmt(*s["speed_error"], digits=2),
                    _fmt(*s["headway_error"], digits=2),
                    " / ".join(f"{np.nanmean(hw[x]):.2f}" if hw[x] else "-" for x in ("cautious", "normal", "brisk")),
                ]
            )
    ex = expert("driving", "test")
    ex_hw = {st: ex["by_style"].get(st, {}).get("headway_mean") for st in ("cautious", "normal", "brisk")}
    lang_rows.append(["전문가", "-", _fmt(stats["driving_expert_success"]), _fmt(ex["overall"].get("speed_error") or np.nan, digits=2), _fmt(ex["overall"].get("headway_error") or np.nan, digits=2), " / ".join(f"{v:.2f}" if v else "-" for v in ex_hw.values())])
    for st_name, v in ex_hw.items():
        stats[f"lang:expert:headway_mean:{st_name}"] = v
    stats["driving_expert_speed_error"] = ex["overall"].get("speed_error")
    _write_table(tables / "vla_language_ablation", ["데이터", "언어 입력", "성공률", "속도 오차(m/s)", "headway 오차(s)", "평균 headway 신중/보통/민첩(s)"], lang_rows)

    # ---------------- 로봇 ----------------
    robot_rows = []
    for m in METHOD_ORDER + ["full"]:
        s = summ.get(("robot", m, 1.0 if m == "full" else 0.10))
        if s is None:
            continue
        robot_rows.append([METHOD_LABELS[m], _fmt(*s["success"]), _fmt(*s["hazard_success"]), _fmt(*s["collision_moving"]), _fmt(*s["speed_error"], digits=3), _fmt(*s["mae"], digits=3), _fmt(*s["hazard_frame_recall"])])
    if robot_rows:
        _write_table(tables / "vla_robot", ["방법", "성공률", "위험 시나리오 성공률", "주행 중 충돌률", "속도 오차(m/s)", "개루프 MAE", "위험 프레임 회수율"], robot_rows)
        rb_ours, rb_rand = summ.get(("robot", "ours", 0.10)), summ.get(("robot", "random", 0.10))
        if rb_ours and rb_rand:
            boot["robot_ours_vs_random_b0.10"] = paired_bootstrap(rb_ours["_succ"], rb_rand["_succ"])

    # ---------------- 파일럿(검증) ----------------
    pilot_rows = []
    for k, rs in sorted(groups.items(), key=lambda kv: (kv[0][2], kv[0][3], kv[0][4])):
        dom, ev, m, b, st, lang, lam, res = k
        if ev != "val" or dom != "driving" or lam is not None:
            continue
        s = summarize_condition(rs, expert("driving", "val"), False)
        pilot_rows.append([METHOD_LABELS[m], f"{int(round(b * 100))}%", str(st), str(len(rs)), _fmt(*s["success"]), _fmt(*s["collision"]), _fmt(*s["speed_error"], digits=2), f"{np.mean([r['train_log']['train_time_s'] for r in rs]):.0f}"])
        stats[f"pilot:{m}:{b:.2f}:{st}:success"] = s["success"][0]
        stats[f"pilot:{m}:{b:.2f}:{st}:success_sd"] = s["success"][1]
        stats[f"pilot:{m}:{b:.2f}:{st}:n"] = len(rs)
    _write_table(tables / "vla_pilot", ["방법", "예산", "경사 단계", "시드 수", "검증 성공률", "충돌률", "속도 오차", "학습 시간(s)"], pilot_rows)
    tune_path = root / "cache" / "tune_driving.json"
    if tune_path.exists():
        stats["tune"] = json.loads(tune_path.read_text(encoding="utf-8"))

    _figures(root, summ, budgets, stats)
    (out_dir / "vla_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    logger.info("VLA 리포트 저장: %s", out_dir)
    return stats


def _figures(root: Path, summ: dict[tuple[str, str, float], dict[str, Any]], budgets: list[float], stats: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import koreanize_matplotlib  # noqa: F401
    except ModuleNotFoundError:  # pragma: no cover
        pass
    fig_dir = root / "summary" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "random": "#9e9e9e",
        "uniform": "#bdbdbd",
        "action_trigger": "#ff9800",
        "rule_ittc": "#8d6e63",
        "uncertainty": "#7e57c2",
        "coreset": "#26a69a",
        "event": "#42a5f5",
        "ours": "#1565c0",
        "offline_loss": "#e57373",
        "oracle": "#212121",
    }
    styles = {"offline_loss": "--", "oracle": ":"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, key, title in zip(axes, ("success", "hazard_success", "hazard_frame_recall"), ("폐루프 성공률", "위험 시나리오 성공률", "선택 데이터의 위험 프레임 회수율")):
        for m in METHOD_ORDER:
            pts = [(b, summ[("driving", m, b)][key]) for b in budgets if ("driving", m, b) in summ]
            if not pts:
                continue
            xs = [100 * b for b, _ in pts]
            ys = [v[0] for _, v in pts]
            es = [v[1] for _, v in pts]
            ax.errorbar(xs, ys, yerr=es, label=METHOD_LABELS[m], color=colors[m], ls=styles.get(m, "-"), marker="o", ms=4, lw=2.2 if m == "ours" else 1.3, capsize=2)
        full = summ.get(("driving", "full", 1.0))
        if full and key != "hazard_frame_recall":
            ax.axhline(full[key][0], color="k", lw=0.8, ls="-.", label="전체 데이터(100%)")
        ax.set_xlabel("저장 예산(%)")
        ax.set_title(title)
        ax.set_xticks([100 * b for b in budgets])
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.5, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_vla_budget_curves.png", dpi=160)
    plt.close(fig)

    # 선택 데이터 구성(주 예산)
    b = stats.get("main_budget")
    if b is not None:
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        ms = [m for m in METHOD_ORDER if ("driving", m, b) in summ]
        rec = [summ[("driving", m, b)]["hazard_frame_recall"][0] for m in ms]
        ent = [summ[("driving", m, b)]["label_entropy_norm"][0] for m in ms]
        x = np.arange(len(ms))
        ax.bar(x - 0.2, rec, 0.4, label="위험 프레임 회수율", color="#1565c0")
        ax.bar(x + 0.2, ent, 0.4, label="라벨 분포 엔트로피(정규화)", color="#90caf9")
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in ms], rotation=30, ha="right", fontsize=8)
        ax.set_title(f"선택 데이터 구성(예산 {int(round(b * 100))}%)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_vla_selection.png", dpi=160)
        plt.close(fig)


def build_comma_report(root: Path) -> dict[str, Any]:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((root / "runs").glob("comma__*.json"))]
    if not runs:
        return {}
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        groups[(r["job"]["method"], round(r["job"]["budget"], 2))].append(r)
    stats: dict[str, Any] = {}
    rows = []
    order = ["random", "uniform", "action_trigger", "rule_ittc", "uncertainty", "coreset", "event", "event_sim", "ours", "oracle"]
    budgets = sorted({b for (_, b) in groups if b < 1.0})
    for b in budgets:
        for m in order:
            rs = groups.get((m, b))
            if not rs:
                continue
            vals = {k: _mean_std([r["open_loop"][k] for r in rs]) for k in ("mae", "mae_braking", "brake_onset_auroc")}
            rec = _mean_std([r["selection"].get("hazard_frame_recall") for r in rs])
            rows.append([METHOD_LABELS[m], "O" if m in EDGE_OK else "X", f"{int(round(b * 100))}%", _fmt(*vals["mae"]), _fmt(*vals["mae_braking"]), _fmt(*vals["brake_onset_auroc"]), _fmt(*rec), str(len(rs))])
            for k, v in vals.items():
                stats[f"comma:{m}:{b:.2f}:{k}"] = v[0]
                stats[f"comma:{m}:{b:.2f}:{k}_sd"] = v[1]
            stats[f"comma:{m}:{b:.2f}:hazard_frame_recall"] = rec[0]
    rs = groups.get(("full", 1.0))
    if rs:
        vals = {k: _mean_std([r["open_loop"][k] for r in rs]) for k in ("mae", "mae_braking", "brake_onset_auroc")}
        rows.append([METHOD_LABELS["full"], "-", "100%", _fmt(*vals["mae"]), _fmt(*vals["mae_braking"]), _fmt(*vals["brake_onset_auroc"]), "1.000", str(len(rs))])
        for k, v in vals.items():
            stats[f"comma:full:1.00:{k}"] = v[0]
    _write_table(root / "summary" / "tables" / "vla_comma", ["방법", "엣지 가능", "예산", "MAE(m/s²)", "제동 구간 MAE", "제동 시작 예측 AUROC", "제동 프레임 회수율", "실행 수(fold×시드)"], rows)
    path = root / "summary" / "vla_stats.json"
    base = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    base.update(stats)
    path.write_text(json.dumps(base, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return stats


def extra_stats(root: Path) -> dict[str, Any]:
    """풀 분포·점수기 품질·실행 시간·장치 정보 등 원고용 부가 수치."""

    from ..vla.pool import load_pool
    from .latency import device_info
    from .metrics import auroc
    from .vla_curation_suite import HAZARD_IDS, pool_path

    stats: dict[str, Any] = {}
    cache = root / "cache"
    for dom in ("driving", "robot"):
        cfg = CurationSuiteConfig(root=root, domain=dom)
        p = pool_path(cfg)
        if not (p.parent / f"{p.name}.npz").exists():
            continue
        pool, meta = load_pool(p)
        y = pool["y"]
        haz = np.isin(y, HAZARD_IDS)
        pre = "pool" if dom == "driving" else "robot_pool"
        stats[f"{pre}_episodes"] = len(meta["episodes"])
        stats[f"{pre}_frames"] = int(len(y))
        stats[f"{pre}_hazard_ratio"] = float(haz.mean())
        ep_change = np.ones(len(y), dtype=bool)
        ep_change[1:] = pool["ep"][1:] != pool["ep"][:-1]
        prev_haz = np.concatenate([[False], haz[:-1]])
        stats[f"{pre}_hazard_events"] = int((haz & (~prev_haz | ep_change)).sum())  # 에피소드 안 위험 구간 시작 수
        stats[f"{pre}_hazard_frames_10pct"] = float(haz.sum() * 0.10)
        stats[f"{pre}_class_ratio"] = [float((y == c).mean()) for c in range(6)]
        stats[f"{pre}_labels"] = meta.get("labels")
        sc = cache / f"scores_{dom}.npz"
        if sc.exists():
            z = np.load(sc)
            stats[f"{pre}_scorer_auroc"] = auroc(z["event_score"], haz)
            stats[f"{pre}_scorer_ms_per_frame_batch"] = float(z["batch_ms_per_frame"])
        styles = [e["style"] for e in meta["episodes"]]
        stats[f"{pre}_style_counts"] = {s: styles.count(s) for s in sorted(set(styles))}
        scen = [e["scenario"] for e in meta["episodes"]]
        stats[f"{pre}_scenario_counts"] = {s: scen.count(s) for s in sorted(set(scen))}
    for dom in ("driving", "robot"):
        p = cache / f"scorer_{dom}.json"
        if p.exists():
            info = json.loads(p.read_text(encoding="utf-8"))
            pre = "scorer" if dom == "driving" else "robot_scorer"
            for k in ("val_macro_f1", "val_hazard_auroc", "params", "train_seconds", "train_episodes", "temperature"):
                stats[f"{pre}_{k}"] = info.get(k)
    dev = device_info()
    stats["cpu_model"] = dev.get("cpu_model", dev.get("processor"))
    stats["cpu_count"] = dev.get("cpu_count")
    stats["torch_version"] = dev.get("torch")
    runs_path = root / "summary" / "curation_runs.json"
    if runs_path.exists():
        runs = json.loads(runs_path.read_text(encoding="utf-8"))
        tt = [r["train_log"]["train_time_s"] for r in runs if r["job"]["domain"] == "driving"]
        tot = [r["seconds"] for r in runs if r["job"]["domain"] == "driving"]
        if tt:
            stats["mean_train_min"] = float(np.mean(tt) / 60)
            stats["mean_eval_min"] = float((np.mean(tot) - np.mean(tt)) / 60)
            stats["total_hours"] = float(np.sum(tot) / 3600 / 4)
            stats["n_policy_runs"] = len(runs)
    path = root / "summary" / "vla_stats.json"
    base = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    base.update(stats)
    path.write_text(json.dumps(base, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return stats


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/exp_110_vla_curation")
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()
    build_report(Path(args.root), args.steps)
    build_comma_report(Path(args.root))
    extra_stats(Path(args.root))


if __name__ == "__main__":
    main()
