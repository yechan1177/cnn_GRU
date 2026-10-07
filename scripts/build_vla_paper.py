"""VLA 연계 큐레이션 논문 원고 빌더.

`paper/manuscript_vla_ko.template.md`와 `paper/sections_vla/*.md`의 자리표시자를 실험 결과로 채워
`paper/manuscript_vla_ko.md`를 만든다. 원고의 모든 수치는 결과 파일에서 온다(손으로 옮겨 적지 않는다).

자리표시자
- `{{s:키[:자릿수]}}`  : vla_stats.json 값(기본 3자리)
- `{{u:키[:자릿수]}}`  : `s:`와 같되 lo·hi·diff 키에도 부호(+)를 붙이지 않는다(비율형 값의 신뢰구간 등)
- `{{p:키[:자릿수]}}`  : 백분율(×100, 기본 1자리, '%' 포함)
- `{{pp:키[:자릿수]}}` : 백분율 포인트(×100, '%' 없이)
- `{{b:키}}`           : 대응 부트스트랩 차이 "+0.071 [95% CI 0.020, 0.120]"
- `{{table:이름}}`     : summary/tables/이름.md 또는 paper/sections_vla/tables/이름.md
- `{{fig:파일|캡션}}`  : paper/figures/파일 이미지와 캡션
- `{{section:이름}}`   : paper/sections_vla/이름.md 포함
- `{{s:k:경로}}`       : v3 KPI(v3_kpi.json) 값. 경로는 ':'로 구분(예: k:K1:value, k:comparisons:care_vs_mix_trigger_b0.02:lo)
- `{{s:k4:경로}}`      : v4 KPI(exp_130 v4_kpi.json) 값(예: k4:K1:value, k4:comparisons:v4_vs_v3_care:lo)
- `{{s:k3:경로}}`      : 세 번째 테스트(exp_130 third_kpi.json) 값(예: k3:kpi:K1:value, k3:tests:H-v3:diff, k3:loo:loo_p2:p_holm)
- `{{s:s4:경로}}`      : 4차 심사 대응 보조 분석(exp_130 supp_round4.json, `scripts/supp_round4_analysis.py`) 값(예: s4:ha_diff:diff, s4:pooled_23:K2:lo)
- `{{이름}}`           : vla_stats.json 최상위 값(문자열 그대로)

사용: python scripts/build_vla_paper.py [--root experiments/exp_110_vla_curation]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

PAPER = Path("paper")
SECTIONS = PAPER / "sections_vla"


def derived(stats: dict[str, Any]) -> dict[str, Any]:
    """원고에 쓰는 파생 수치."""

    d: dict[str, Any] = {}
    pilot_steps = 3000  # 시드 반복 파일럿은 T=3000에서 수행했다(심사 M2: T=6000 키를 읽던 오류 수정)
    for m, b in (("full", 1.0), ("random", 0.10), ("oracle", 0.10)):
        d[f"pilot_sd_{m}"] = stats.get(f"pilot:{m}:{b:.2f}:{pilot_steps}:success_sd", float("nan"))
    sds = [v for v in (d["pilot_sd_full"], d["pilot_sd_random"], d["pilot_sd_oracle"]) if isinstance(v, (int, float)) and math.isfinite(v)]
    sd_pilot = max(sds) if sds else float("nan")
    # 시드 3개 평균 차이의 표준오차 ≈ √2·sd/√3, 최소 검출 효과 ≈ 2.8·SE(양측 α=0.05, 검정력 0.8)
    d["pilot_se_diff"] = math.sqrt(2.0) * sd_pilot / math.sqrt(3.0)
    d["pilot_mde"] = 2.8 * d["pilot_se_diff"]
    sd_main = stats.get("driving:random:0.02:success_sd", float("nan"))
    d["main_sd_random_b0.02"] = sd_main
    d["main_mde_3seeds"] = 2.8 * math.sqrt(2.0) * sd_main / math.sqrt(3.0)
    d["main_mde_10seeds"] = 2.8 * math.sqrt(2.0) * sd_main / math.sqrt(10.0)
    d["main_mde_7seeds"] = 2.8 * math.sqrt(2.0) * sd_main / math.sqrt(7.0)
    # 손으로 쓰던 비율을 파생 키로(심사 N18)
    yo, ym, pt, po = stats.get("cost_yolo640_ms"), stats.get("cost_yolo320_ms"), stats.get("cost_scorer_torch_ms"), stats.get("cost_scorer_onnx_ms")
    if yo and po:
        d["cost_ratio_onnx_yolo640"] = po / yo
    if ym and pt:
        d["cost_ratio_torch_yolo320"] = pt / ym
    rs = stats.get("driving:random:0.02:hazard_share")
    shares = [stats.get(f"driving:{m}:0.02:hazard_share") for m in ("action_trigger", "event", "oracle", "offline_loss")]
    shares = [x for x in shares if isinstance(x, (int, float))]
    if rs and shares:
        d["risk_share_ratio_min"] = min(shares) / rs
        d["risk_share_ratio_max"] = max(shares) / rs
    succ = [stats.get(f"driving:{m}:0.02:success") for m in ("action_trigger", "event", "oracle", "offline_loss")]
    succ = [x for x in succ if isinstance(x, (int, float))]
    if succ:
        d["risk_methods_max_success_b0.02"] = max(succ)
    tune = stats.get("tune") or {}
    d["tune_lam"] = tune.get("lam", float("nan"))
    d["tune_reservoir"] = tune.get("reservoir", float("nan"))
    d["pool_scorer_auroc"] = stats.get("pool_scorer_auroc", float("nan"))
    for b in (0.01, 0.02, 0.05, 0.10):
        o, f = stats.get(f"driving:ours:{b:.2f}:success"), stats.get("driving:full:1.00:success")
        if o is not None and f:
            d[f"ratio_ours_full_b{b:.2f}"] = o / f
    return d


def pool_table(stats: dict[str, Any]) -> str:
    labels = stats.get("pool_labels") or ["normal_drive", "front_vehicle_follow", "brake_warning", "hard_brake_risk", "post_brake_recovery", "dense_traffic"]
    ko = {
        "normal_drive": "정상 주행",
        "front_vehicle_follow": "전방 추종",
        "brake_warning": "제동 경고(위험)",
        "hard_brake_risk": "강한 제동 위험(위험)",
        "post_brake_recovery": "제동 후 회복",
        "dense_traffic": "밀집",
    }
    ratios = stats.get("pool_class_ratio") or []
    rows = ["| 맥락 | 비율 |", "|---|---|"]
    for name, r in zip(labels, ratios):
        rows.append(f"| {ko.get(name, name)} | {100 * r:.1f}% |")
    styles = stats.get("pool_style_counts") or {}
    if styles:
        rows.append(f"| (지시문 스타일 신중/보통/민첩) | {styles.get('cautious', 0)} / {styles.get('normal', 0)} / {styles.get('brisk', 0)} 에피소드 |")
    return "\n".join(rows)


def fmt_num(v: Any, digits: int) -> str:
    if v is None:
        return "-"
    if isinstance(v, (int, float)):
        if not math.isfinite(float(v)):
            return "-"
        return f"{float(v):.{digits}f}".replace("-", "−")  # 음수 부호는 U+2212로 통일
    return str(v)


def render(text: str, stats: dict[str, Any], root: Path, depth: int = 0) -> str:
    def lookup(key: str) -> Any:
        if key in stats:
            return stats[key]
        # 계층 부트스트랩: hb:<비교키>:<diff|lo|hi|p_le0>, 시드 대응 t: hbt:<비교키>:<diff|lo|hi|t>
        if key.startswith(("hb:", "hbt:")):
            kind, comp, field = key.split(":", 2)
            entry = stats["hboot"][comp]
            return entry["seed_t"][field] if kind == "hbt" else entry[field]
        # 확증 실험 검정: ct:<검정키>:<diff|lo|hi|p_le0|p_holm>, 시드 대응 t: ctt:<검정키>:<diff|lo|hi|t>
        if key.startswith("sdiff:"):  # 시드별 차이: sdiff:<예산>:<인덱스>
            _, b, i = key.split(":")
            return stats[f"seed_diffs:ours_vs_random:{b}"][int(i)]
        if key.startswith("cp:"):  # 실주행 fold 대응 차이: cp:<비교>:<예산>:<지표>:<diff|lo|hi>
            _, comp, b, metric, field = key.split(":")
            return stats[f"comma_pair:{comp}:{b}:{metric}"][field]
        if key.startswith("s4:"):  # 4차 심사 보조 분석(supp_round4.json): s4:<경로...>
            vs4: Any = SUPP4
            for part in key.split(":")[1:]:
                vs4 = vs4[part]
            return vs4
        if key.startswith("k3:"):  # 세 번째 테스트(third_kpi.json): k3:<경로...>
            v3t: Any = KPI3
            for part in key.split(":")[1:]:
                v3t = v3t[part]
            return v3t
        if key.startswith("k4:"):  # v4 KPI: k4:<경로...>
            v4: Any = KPI4
            for part in key.split(":")[1:]:
                v4 = v4[part]
            return v4
        if key.startswith("k:"):  # v3 KPI: k:<경로...>
            v: Any = KPI
            for part in key.split(":")[1:]:
                v = v[part]
            return v
        if key.startswith(("ct:", "ctt:")):
            kind, comp, field = key.split(":", 2)
            entry = stats["confirm_tests"][comp]
            return entry["seed_t"][field] if kind == "ctt" else entry[field]
        raise KeyError(key)

    def repl(m: re.Match[str]) -> str:
        body = m.group(1)
        try:
            if body.startswith("section:"):
                return render((SECTIONS / f"{body[8:]}.md").read_text(encoding="utf-8"), stats, root, depth + 1)
            if body.startswith("table:"):
                name = body[6:]
                if name == "pool_distribution":
                    return pool_table(stats)
                for base in (root / "summary" / "tables", *V3_TABLES, SECTIONS / "tables"):
                    p = base / f"{name}.md"
                    if p.exists():
                        return p.read_text(encoding="utf-8").strip()
                return f"**[표 없음: {name}]**"
            if body.startswith("fig:"):
                fname, _, caption = body[4:].partition("|")
                num = caption.split(".")[0] if caption.startswith("그림") else fname
                return f"![{num}](figures/{fname})\n\n{caption}"
            if body.startswith(("s:", "p:", "pp:", "b:", "u:")):
                kind, _, rest = body.partition(":")
                parts = rest.split(":")
                digits = None
                if len(parts) > 1 and parts[-1].isdigit():
                    digits = int(parts[-1])
                    parts = parts[:-1]
                key = ":".join(parts)
                if kind == "b":
                    bt = stats["bootstrap"][key]
                    return f"{bt['diff']:+.3f} [95% CI {bt['lo']:+.3f}, {bt['hi']:+.3f}]".replace("-", "−")
                v = lookup(key)
                signed = key.startswith(("hb:", "hbt:", "ct:", "ctt:", "cp:", "sdiff:", "k:", "k4:", "k3:", "s4:")) and key.split(":")[-1] in ("diff", "lo", "hi") or key.startswith("sdiff:")
                if kind == "u":
                    kind = "s"
                    signed = False
                if kind == "s" and signed and isinstance(v, (int, float)) and math.isfinite(float(v)):
                    txt = f"{float(v):+.{3 if digits is None else digits}f}"
                    return txt.replace("-", "−")
                if kind == "s":
                    out = fmt_num(v, 3 if digits is None else digits)
                    if out == "-":
                        MISSING.append(f"{body} (값 없음/NaN)")
                    return out
                if not (isinstance(v, (int, float)) and math.isfinite(float(v))):
                    MISSING.append(f"{body} (값 없음/NaN/숫자 아님)")  # p·pp도 누락을 보고한다(3차 심사 대응 점검)
                    return "-"
                if kind == "p":
                    return fmt_num(100 * v, 1 if digits is None else digits) + "%"
                return fmt_num(100 * v, 1 if digits is None else digits)
            v = lookup(body)
            return fmt_num(v, 3) if isinstance(v, float) else str(v)
        except (KeyError, FileNotFoundError) as exc:
            MISSING.append(body)
            return f"**[누락: {body} ({exc.__class__.__name__})]**"

    return re.sub(r"\{\{([^{}]+)\}\}", repl, text)


MISSING: list[str] = []
KPI: dict[str, Any] = {}
KPI4: dict[str, Any] = {}
KPI3: dict[str, Any] = {}
SUPP4: dict[str, Any] = {}
V3_TABLES: list[Path] = []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/exp_110_vla_curation")
    ap.add_argument("--out", default=str(PAPER / "manuscript_vla_ko.md"))
    ap.add_argument("--v3-root", default="experiments/exp_120_vla_v3", help="v3 KPI 결과 루트(k: 자리표시자, v3 표·그림)")
    ap.add_argument("--v4-root", default="experiments/exp_130_vla_v4", help="v4 KPI 결과 루트(k4: 자리표시자, v4 표·그림)")
    args = ap.parse_args()
    root = Path(args.root)
    v3 = Path(args.v3_root)
    V3_TABLES.append(v3 / "summary" / "tables")
    v4r = Path(args.v4_root)
    V3_TABLES.append(v4r / "summary" / "tables")
    if (v4r / "summary" / "supp_round4.json").exists():
        SUPP4.update(json.loads((v4r / "summary" / "supp_round4.json").read_text(encoding="utf-8")))
    if (v4r / "summary" / "third_kpi.json").exists():
        KPI3.update(json.loads((v4r / "summary" / "third_kpi.json").read_text(encoding="utf-8")))
    if (v4r / "summary" / "v4_kpi.json").exists():
        KPI4.update(json.loads((v4r / "summary" / "v4_kpi.json").read_text(encoding="utf-8")))
        if (v4r / "summary" / "figures").exists():
            for p in (v4r / "summary" / "figures").glob("*.png"):
                shutil.copy2(p, PAPER / "figures" / p.name)
    if (v3 / "summary" / "v3_kpi.json").exists():
        KPI.update(json.loads((v3 / "summary" / "v3_kpi.json").read_text(encoding="utf-8")))
        for p in (v3 / "summary" / "figures").glob("*.png") if (v3 / "summary" / "figures").exists() else []:
            shutil.copy2(p, PAPER / "figures" / p.name)
    stats = json.loads((root / "summary" / "vla_stats.json").read_text(encoding="utf-8"))
    stats.update(derived(stats))
    fig_src = root / "summary" / "figures"
    if fig_src.exists():
        for p in fig_src.glob("*.png"):
            shutil.copy2(p, PAPER / "figures" / p.name)
    text = render((PAPER / "manuscript_vla_ko.template.md").read_text(encoding="utf-8"), stats, root)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"원고 저장: {args.out} ({len(text)}자)")
    if MISSING:
        print("누락 자리표시자:", sorted(set(MISSING)))


if __name__ == "__main__":
    main()
