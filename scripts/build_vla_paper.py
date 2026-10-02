"""VLA 연계 큐레이션 논문 원고 빌더.

`paper/manuscript_vla_ko.template.md`와 `paper/sections_vla/*.md`의 자리표시자를 실험 결과로 채워
`paper/manuscript_vla_ko.md`를 만든다. 원고의 모든 수치는 결과 파일에서 온다(손으로 옮겨 적지 않는다).

자리표시자
- `{{s:키[:자릿수]}}`  : vla_stats.json 값(기본 3자리)
- `{{p:키[:자릿수]}}`  : 백분율(×100, 기본 1자리, '%' 포함)
- `{{pp:키[:자릿수]}}` : 백분율 포인트(×100, '%' 없이)
- `{{b:키}}`           : 대응 부트스트랩 차이 "+0.071 [95% CI 0.020, 0.120]"
- `{{table:이름}}`     : summary/tables/이름.md 또는 paper/sections_vla/tables/이름.md
- `{{fig:파일|캡션}}`  : paper/figures/파일 이미지와 캡션
- `{{section:이름}}`   : paper/sections_vla/이름.md 포함
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
    steps = stats.get("steps", 3000)
    for m, b in (("full", 1.0), ("random", 0.10), ("oracle", 0.10)):
        d[f"pilot_sd_{m}"] = stats.get(f"pilot:{m}:{b:.2f}:{steps}:success_sd", float("nan"))
    n = stats.get("driving_n_test", 147) or 147
    sds = [v for v in (d["pilot_sd_random"], d["pilot_sd_oracle"]) if isinstance(v, (int, float)) and math.isfinite(v)]
    sd_seed = max(sds) if sds else float("nan")
    # 대응 차이의 표준오차: 시드 간 변동(평균 3개) 기준 sqrt(2)*sd/sqrt(3)
    d["pilot_se_diff"] = math.sqrt(2.0) * sd_seed / math.sqrt(3.0) if math.isfinite(sd_seed) else float("nan")
    d["pilot_mde"] = 2.8 * d["pilot_se_diff"] if math.isfinite(d["pilot_se_diff"]) else float("nan")  # α=0.05 양측, 검정력 0.8
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
        return f"{float(v):.{digits}f}"
    return str(v)


def render(text: str, stats: dict[str, Any], root: Path, depth: int = 0) -> str:
    def lookup(key: str) -> Any:
        if key in stats:
            return stats[key]
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
                for base in (root / "summary" / "tables", SECTIONS / "tables"):
                    p = base / f"{name}.md"
                    if p.exists():
                        return p.read_text(encoding="utf-8").strip()
                return f"**[표 없음: {name}]**"
            if body.startswith("fig:"):
                fname, _, caption = body[4:].partition("|")
                return f"![{caption}](figures/{fname})\n\n{caption}"
            if body.startswith(("s:", "p:", "pp:", "b:")):
                kind, _, rest = body.partition(":")
                parts = rest.split(":")
                digits = None
                if len(parts) > 1 and parts[-1].isdigit():
                    digits = int(parts[-1])
                    parts = parts[:-1]
                key = ":".join(parts)
                if kind == "b":
                    bt = stats["bootstrap"][key]
                    return f"{bt['diff']:+.3f} [95% CI {bt['lo']:+.3f}, {bt['hi']:+.3f}]"
                v = lookup(key)
                if kind == "s":
                    return fmt_num(v, 3 if digits is None else digits)
                if kind == "p":
                    return (fmt_num(100 * v, 1 if digits is None else digits) + "%") if isinstance(v, (int, float)) and math.isfinite(v) else "-"
                return fmt_num(100 * v, 1 if digits is None else digits) if isinstance(v, (int, float)) else "-"
            v = lookup(body)
            return fmt_num(v, 3) if isinstance(v, float) else str(v)
        except (KeyError, FileNotFoundError) as exc:
            MISSING.append(body)
            return f"**[누락: {body} ({exc.__class__.__name__})]**"

    return re.sub(r"\{\{([^{}]+)\}\}", repl, text)


MISSING: list[str] = []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/exp_110_vla_curation")
    ap.add_argument("--out", default=str(PAPER / "manuscript_vla_ko.md"))
    args = ap.parse_args()
    root = Path(args.root)
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
