from __future__ import annotations

"""실험 결과 JSON/CSV → 논문 원고(paper/manuscript_ko.md) 생성.

템플릿 문법
- {{이름}}                         : 아래 `named_values`에서 계산한 값
- {{table:CSV이름}}                : experiments/.../summary/tables/<CSV이름>.csv 를 Markdown 표로 삽입
- {{fig:파일이름|캡션}}             : summary/figures/<파일이름> 을 paper/figures로 복사 후 그림 삽입
- {{syn:모델:테스트셋:지표}}        : 합성 스위트 시드 평균 ± 표준편차 (예: syn:mc_cnn_gru_semantic_v2:main:macro_f1)
- {{synm:모델:테스트셋:지표}}       : 같은 값의 평균만
- {{comma:모델:test_20fps|test_10fps:지표}} / {{commam:...}}
- {{robot:모델:지표}} / {{robotm:...}}
- {{section:파일이름}}             : paper/sections/<파일이름>.md 를 (재귀적으로 치환해) 삽입
수치를 원고에 직접 쓰지 말고 반드시 이 문법으로 참조한다.
"""

import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "experiments" / "exp_100_paper_suite" / "summary"
PAPER = ROOT / "paper"


def _load(name: str) -> dict[str, Any] | None:
    path = SUMMARY / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


SYN = _load("synthetic_results.json")
COM = _load("comma_results.json")
VLA = _load("vla_results.json")
LAT = _load("latency_results.json")
MISSING: list[str] = []


def _fmt(vals: list[float], digits: int = 3, std: bool = True) -> str:
    vals = [float(v) for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return "N/A"
    if not std or len(vals) == 1:
        return f"{np.mean(vals):.{digits}f}"
    return f"{np.mean(vals):.{digits}f} ± {np.std(vals, ddof=1):.{digits}f}"


def _get(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit():
            cur = cur[int(part)]
        else:
            return None
    return cur


def _syn_vals(model: str, test: str, metric: str) -> list[float]:
    if not SYN:
        return []
    return [_get(r["results"][test], metric) for r in SYN["records"] if r["model"] == model and test in r["results"]]


def _comma_vals(model: str, split: str, metric: str) -> list[float]:
    if not COM:
        return []
    return [_get(r[split], metric) for r in COM["records"] if r["model"] == model and split in r]


def _robot_vals(model: str, metric: str) -> list[float]:
    if not VLA:
        return []
    return [_get(r["test"], metric) for r in VLA["robot"]["records"] if r["model"] == model]


def _table(name: str) -> str:
    path = SUMMARY / "tables" / f"{name}.csv"
    if not path.exists():
        MISSING.append(f"table:{name}")
        return f"> (표 {name} 없음)"
    with path.open(encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(out)


def _fig(spec: str) -> str:
    name, _, caption = spec.partition("|")
    src = SUMMARY / "figures" / name
    if not src.exists():
        MISSING.append(f"fig:{name}")
        return f"> (그림 {name} 없음)"
    (PAPER / "figures").mkdir(parents=True, exist_ok=True)
    shutil.copy(src, PAPER / "figures" / name)
    return f"![{caption}](figures/{name})\n\n*{caption}*"


def named_values() -> dict[str, str]:
    v: dict[str, str] = {}
    if SYN:
        ds = SYN["dataset"]
        v["synth_episodes"] = str(ds["episodes"])
        v["prop_f1"] = _fmt(_syn_vals("mc_cnn_gru_semantic_v2", "main", "macro_f1"))
        v["v1_f1"] = _fmt(_syn_vals("mc_cnn_gru_balanced_v1", "main", "macro_f1"))
        v["rule_v1_f1"] = _fmt(_syn_vals("rule_v1", "main", "macro_f1"))
        v["v1_f1_30"] = _fmt(_syn_vals("mc_cnn_gru_balanced_v1", "test_30fps_mid", "macro_f1"))
        v["prop_f1_30"] = _fmt(_syn_vals("mc_cnn_gru_semantic_v2", "test_30fps_mid", "macro_f1"))
        lc = np.array(ds["label_counts"]["train"]) + np.array(ds["label_counts"]["val"]) + np.array(ds["label_counts"]["test"])
        share = ", ".join(f"{l} {100 * c / lc.sum():.1f}%" for l, c in zip(SYN["labels"], lc, strict=True))
        v["synth_data_line"] = (
            f"생성된 메인 데이터는 {ds['episodes']}개 에피소드, {ds['frames']:,} 프레임이며 "
            f"에피소드 단위 분할은 {ds['split_episodes']['train']}/{ds['split_episodes']['val']}/{ds['split_episodes']['test']}이다. "
            f"라벨 비율은 {share}이다. 시뮬레이터에서 근접 접촉(충돌 처리)이 발생한 에피소드는 {ds['episodes_with_contact']}개로, 공격적 끼어들기 등 안전 임계 상황을 일부러 포함한 결과다."
        )
    if VLA:
        rd = VLA["robot"]["dataset"]
        v["robot_episodes"] = str(rd["episodes"])
        lc = np.array(rd["label_counts"]["train"]) + np.array(rd["label_counts"]["val"]) + np.array(rd["label_counts"]["test"])
        share = ", ".join(f"{l} {100 * c / lc.sum():.1f}%" for l, c in zip(VLA["robot"]["labels"], lc, strict=True))
        v["robot_data_line"] = f"{rd['episodes']}개 에피소드, {rd['frames']:,} 프레임을 생성했고 라벨 비율은 {share}이다. 근접 접촉 에피소드는 {rd['episodes_with_contact']}개다."
        v["robot_ft"] = _fmt(_robot_vals("robot_10pct_finetune_from_driving", "macro_f1"))
        v["robot_scratch"] = _fmt(_robot_vals("robot_10pct_scratch", "macro_f1"))
        recs = [r for r in VLA["comma_action_curation"]["records"] if r["model"].endswith("_action")]
        v["cur20_model"] = _fmt([r["curation"]["0.2"]["model"]["frame_coverage"] for r in recs], std=False)
        v["cur20_random"] = _fmt([r["curation"]["0.2"]["random_frame_coverage"] for r in recs], std=False)
    if COM:
        v["comma_prop_f1"] = _fmt(_comma_vals("mc_cnn_gru_semantic_v2", "test_20fps", "macro_f1"))
        v["comma_major_f1"] = _fmt(_comma_vals("majority", "test_20fps", "macro_f1"))
        v["comma_prop_auroc"] = _fmt(_comma_vals("mc_cnn_gru_semantic_v2", "test_20fps", "braking_auroc"))
        lc = COM["dataset"]["label_counts"]
        v["comma_data_line"] = "라벨 분포는 " + ", ".join(f"{l} {c:,}" for l, c in zip(COM["labels"], lc, strict=True)) + " 프레임이다."
    if LAT:
        t = {x["model"]: x for x in LAT["temporal"]}
        if "mc_cnn_gru_semantic_v2" in t:
            v["lat_prop_ms"] = f"{t['mc_cnn_gru_semantic_v2']['torch_cpu_t1']['p50_ms']:.2f}"
            v["prop_params"] = f"{t['mc_cnn_gru_semantic_v2']['params']:,}"
        y = [x for x in LAT.get("yolo", []) if x["imgsz"] == 640 and x["threads"] == 4]
        if y:
            v["lat_yolo_ms"] = f"{y[0]['p50_ms']:.1f}"
        v["cpu_model"] = str(LAT["device"].get("cpu_model", LAT["device"].get("processor")))
        v["cpu_count"] = str(LAT["device"].get("cpu_count"))
    log = ROOT / "data" / "processed" / "comma_speedchallenge" / "extract.log"
    if log.exists():
        m = re.search(r"완료: (\d+) 프레임, ([\d.]+) s", log.read_text(encoding="utf-8"))
        if m:
            v["extract_ms"] = f"{1000 * float(m.group(2)) / int(m.group(1)):.1f}"
    return v


VALUES = named_values()


def resolve(text: str, depth: int = 0) -> str:
    def repl(m: re.Match[str]) -> str:
        key = m.group(1).strip()
        kind, _, rest = key.partition(":")
        if key in VALUES:
            return VALUES[key]
        if kind == "table":
            return _table(rest)
        if kind == "fig":
            return _fig(rest)
        if kind == "section":
            path = PAPER / "sections" / f"{rest}.md"
            if path.exists() and depth < 3:
                return resolve(path.read_text(encoding="utf-8"), depth + 1)
            MISSING.append(key)
            return f"> (섹션 {rest} 없음)"
        parts = rest.split(":")
        if kind in {"syn", "synm"} and len(parts) == 3:
            return _fmt(_syn_vals(*parts), std=kind == "syn")
        if kind in {"comma", "commam"} and len(parts) == 3:
            return _fmt(_comma_vals(*parts), std=kind == "comma")
        if kind in {"robot", "robotm"} and len(parts) == 2:
            return _fmt(_robot_vals(*parts), std=kind == "robot")
        MISSING.append(key)
        return "N/A"

    return re.sub(r"\{\{([^{}]+)\}\}", repl, text)


def main() -> None:
    template = (PAPER / "manuscript_ko.template.md").read_text(encoding="utf-8")
    out = resolve(template)
    out = re.sub(r"<!-- 이 파일은 템플릿이다.*?-->\n", "<!-- 자동 생성 파일: scripts/build_paper.py (직접 수정 금지) -->\n", out, flags=re.S)
    (PAPER / "manuscript_ko.md").write_text(out, encoding="utf-8")
    print(f"생성: {PAPER / 'manuscript_ko.md'}")
    if MISSING:
        print("채우지 못한 항목:", sorted(set(MISSING)), file=sys.stderr)


if __name__ == "__main__":
    main()
