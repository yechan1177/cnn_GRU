from __future__ import annotations

"""실험 결과 JSON → 논문용 표(Markdown/CSV)와 그림(PNG) 생성.

모든 수치는 `*_results.json`에서만 읽는다(수기 입력 없음).
"""

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

NAMES: dict[str, str] = {
    "majority": "다수 클래스",
    "rule_v1": "룰(v1 특징, 2026-03 비교군 형태)",
    "rule_v2": "룰(v2 특징)",
    "rule_v2_ego": "룰(v2 특징)",
    "mlp_last_v2": "MLP(마지막 프레임, 시간정보 없음)",
    "gru_v2": "GRU",
    "cnn_gru_single_v2": "단일채널 CNN-GRU(문헌형)",
    "mc_cnn_gru_balanced_v2": "멀티채널 CNN-GRU(균등 분할, v2)",
    "mc_cnn_gru_semantic_v2": "멀티채널 CNN-GRU(의미 그룹, v2)",
    "mc_cnn_gru_semantic_v2_focal": "의미 그룹(v2) + CB-focal 손실",
    "mc_cnn_gru_semantic_v2_w16": "의미 그룹(v2) + 창 16",
    "mc_cnn_gru_semantic_v2_ema": "의미 그룹(v2) + 인과 EMA 평활화",
    "mc_cnn_gru_balanced_v1": "2026-03 구성(v1 특징, 균등 분할)",
    "mc_cnn_gru_balanced_v1_hybrid": "2026-03 구성 + 하이브리드 룰 게이트",
    "mc_cnn_gru_semantic_v1": "멀티채널 의미 그룹(v1 특징)",
    "mc_cnn_gru_semantic_v2_action": "멀티채널(의미 그룹, v2) + 행동 보조 head",
    "mc_cnn_gru_semantic_v1v2": "멀티채널 의미 그룹(v1+v2 결합 특징)",
    "mc_cnn_gru_semantic_v1v2_w16": "멀티채널 의미 그룹(v1+v2 결합) + 창 16",
    "driving_pretrained_zero_shot": "주행 사전학습 → 로봇 zero-shot",
    "robot_10pct_scratch": "로봇 10% 데이터, 처음부터 학습",
    "robot_10pct_finetune_from_driving": "로봇 10% 데이터, 주행 사전학습 후 미세조정",
}

# 색상: dataviz 기준 팔레트 슬롯 1~3(고정 순서) + 중립 회색
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
C_NEUTRAL = "#a8a7a0"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PROPOSED = "mc_cnn_gru_semantic_v2"  # 하위 실험(comma/VLA)에 쓰는 기준 구성


def _name(key: str) -> str:
    return NAMES.get(key, key)


def _ms(values: list[float], digits: int = 3) -> str:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.{digits}f}"
    return f"{np.mean(vals):.{digits}f} ± {np.std(vals, ddof=1):.{digits}f}"


def _mean(values: list[float]) -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _setup_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import koreanize_matplotlib  # noqa: F401  (나눔고딕 등록)
    except ModuleNotFoundError:  # pragma: no cover
        logger.warning("한글 폰트 패키지가 없어 그림의 한글이 깨질 수 있습니다(pip install koreanize-matplotlib).")
    plt.rcParams.update(
        {
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.unicode_minus": False,
            "figure.dpi": 150,
        }
    )
    return plt


# ----------------------------------------------------------------------
def _group_records(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        out[r["model"]].append(r)
    return out


def synthetic_section(res: dict[str, Any], fig_dir: Path, tab_dir: Path) -> str:
    plt = _setup_matplotlib()
    labels = res["labels"]
    groups = _group_records(res["records"])
    order = [
        "majority", "rule_v1", "rule_v2", "mlp_last_v2", "gru_v2", "cnn_gru_single_v2",
        "mc_cnn_gru_balanced_v2", "mc_cnn_gru_semantic_v2", "mc_cnn_gru_semantic_v2_focal",
        "mc_cnn_gru_semantic_v2_w16", "mc_cnn_gru_semantic_v2_ema",
        "mc_cnn_gru_balanced_v1", "mc_cnn_gru_balanced_v1_hybrid", "mc_cnn_gru_semantic_v1",
    ]
    order = [m for m in order if m in groups]
    md = []
    ds = res["dataset"]
    md.append("### 합성 6맥락 벤치마크: 데이터")
    md.append(
        f"- 에피소드 {ds['episodes']}개(각 30초, 15fps), 총 {ds['frames']:,} 프레임, "
        f"분할(에피소드 단위) train/val/test = {ds['split_episodes']['train']}/{ds['split_episodes']['val']}/{ds['split_episodes']['test']}"
    )
    md.append(f"- 시나리오 구성: " + ", ".join(f"{k} {v}" for k, v in ds["scenario_counts"].items()))
    md.append(f"- 근접 접촉(시뮬레이터 상 충돌 처리)이 발생한 에피소드: {ds['episodes_with_contact']}개")
    rows = [[labels[i]] + [str(ds["label_counts"][p][i]) for p in ("train", "val", "test")] for i in range(len(labels))]
    md.append(_md_table(["라벨", "train", "val", "test"], rows))

    # 메인 테스트 표
    header = ["모델", "파라미터", "정확도", "macro-F1", "macro-F1 95% CI", "제동 이벤트 recall", "제동 프레임 recall", "오경보/분", "검출 지연 중앙값(s)", "라벨 전환/분", "ECE"]
    rows = []
    for m in order:
        rs = groups[m]
        main = [r["results"]["main"] for r in rs]
        ci = [r["results"]["main"].get("ci95", {}).get("macro_f1") for r in rs]
        ci = [c for c in ci if c]
        lat = [x["median_latency_s"] for x in main if x["median_latency_s"] is not None]
        rows.append([
            _name(m),
            f"{rs[0].get('params', 0):,}",
            _ms([x["accuracy"] for x in main]),
            _ms([x["macro_f1"] for x in main]),
            f"[{np.mean([c[0] for c in ci]):.3f}, {np.mean([c[1] for c in ci]):.3f}]" if ci else "-",
            _ms([x["event_recall"] for x in main]),
            _ms([x["event_frame_recall"] for x in main]),
            _ms([x["false_alarms_per_min"] for x in main], 2),
            _ms(lat, 2) if lat else "-",
            _ms([x["flicker_per_min"] for x in main], 1),
            _ms([x.get("ece") for x in main]) if main[0].get("ece") is not None else "-",
        ])
    gt_flicker = groups[order[0]][0]["results"]["main"]["gt_flicker_per_min"]
    md.append("\n### 표 S1. 합성 벤치마크 테스트 결과 (평균 ± 표준편차, 시드 3개)")
    md.append(_md_table(header, rows))
    md.append(f"\n- 정답 라벨 자체의 전환 빈도: {gt_flicker:.1f}회/분. CI는 에피소드 단위 bootstrap(시드별 구간 평균).")
    _write_csv(tab_dir / "synthetic_main.csv", header, rows)

    # 클래스별 F1
    sel = [m for m in ("rule_v1", "mc_cnn_gru_balanced_v1", "mc_cnn_gru_balanced_v1_hybrid", "mc_cnn_gru_semantic_v2", "mc_cnn_gru_semantic_v2_focal") if m in groups]
    rows = []
    for m in sel:
        pcs = np.array([[np.nan if v is None else v for v in r["results"]["main"]["per_class_f1"]] for r in groups[m]], dtype=float)
        rows.append([_name(m)] + [f"{np.nanmean(pcs[:, i]):.3f}" for i in range(len(labels))])
    md.append("\n### 표 S2. 클래스별 F1 (테스트, 시드 평균)")
    md.append(_md_table(["모델"] + labels, rows))
    _write_csv(tab_dir / "synthetic_per_class.csv", ["모델"] + labels, rows)

    # 강건성
    tests = [("test_10fps_mid", "10fps"), ("main", "15fps(기준)"), ("test_30fps_mid", "30fps"), ("test_15fps_low", "노이즈 low"), ("test_15fps_high", "노이즈 high")]
    rob_models = [m for m in ("rule_v1", "rule_v2", "mc_cnn_gru_balanced_v1", "mc_cnn_gru_balanced_v1_hybrid", "mc_cnn_gru_semantic_v1", "mc_cnn_gru_balanced_v2", "mc_cnn_gru_semantic_v2", "mc_cnn_gru_semantic_v2_focal") if m in groups]
    rows = []
    for m in rob_models:
        rows.append([_name(m)] + [_ms([r["results"][t]["macro_f1"] for r in groups[m] if t in r["results"]]) for t, _ in tests])
    md.append("\n### 표 S3. 강건성: 학습 15fps/mid 노이즈 → 테스트 조건 변화 (macro-F1)")
    md.append("동일 테스트 시드(같은 물리 상황)를 FPS/노이즈만 바꿔 다시 생성했다.\n")
    md.append(_md_table(["모델"] + [n for _, n in tests], rows))
    _write_csv(tab_dir / "synthetic_robustness.csv", ["모델"] + [n for _, n in tests], rows)

    # 큐레이션/불확실성
    rows = []
    for m in [x for x in ("mc_cnn_gru_balanced_v1", "mc_cnn_gru_semantic_v2", "mc_cnn_gru_semantic_v2_focal") if x in groups]:
        main = [r["results"]["main"] for r in groups[m]]
        rows.append([_name(m), _ms([x.get("error_auroc_entropy") for x in main])] + [_ms([x["curation_event_recall"][b] for x in main]) for b in ("0.05", "0.1", "0.2", "0.3")])
    md.append("\n### 표 S4. 불확실성/큐레이션 (테스트)")
    md.append("오류 탐지 AUROC는 예측 엔트로피로 오분류 프레임을 구분하는 성능, 회수율은 상위 k% 프레임 선택 시 제동 이벤트 프레임 포함 비율(무작위 선택 기대값 = k).\n")
    md.append(_md_table(["모델", "오류 탐지 AUROC", "5%", "10%", "20%", "30%"], rows))

    # 그림 1: macro-F1 막대
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(order) + 1.0))
    means = [_mean([r["results"]["main"]["macro_f1"] for r in groups[m]]) for m in order]
    stds = [np.std([r["results"]["main"]["macro_f1"] for r in groups[m]], ddof=1) if len(groups[m]) > 1 else 0.0 for m in order]
    def _color(m: str) -> str:
        if m.startswith("rule") or m == "majority":
            return C_NEUTRAL
        return C_ORANGE if "_v1" in m and "_v1v2" not in m else C_BLUE

    colors = [_color(m) for m in order]
    y = np.arange(len(order))[::-1]
    ax.barh(y, means, xerr=stds, color=colors, height=0.6, error_kw={"ecolor": INK2, "lw": 1, "capsize": 2})
    for yi, v in zip(y, means, strict=True):
        ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [_name(m) for m in order], fontsize=8)
    ax.set_xlabel("테스트 macro-F1 (시드 3개 평균 ± 표준편차)")
    ax.set_xlim(0, min(1.0, max(means) + 0.12))
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[Patch(color=C_BLUE, label="특징 v2(제안)"), Patch(color=C_ORANGE, label="특징 v1(2026-03)"), Patch(color=C_NEUTRAL, label="비학습 기준선")],
        frameon=False, fontsize=8, loc="lower right",
    )
    ax.set_title("합성 6맥락 벤치마크", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_synthetic_macro_f1.png")
    plt.close(fig)

    # 그림 2: FPS 강건성
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    fps_tests = [("test_10fps_mid", 10), ("main", 15), ("test_30fps_mid", 30)]
    for m, color, marker in (("mc_cnn_gru_balanced_v1", C_ORANGE, "s"), ("mc_cnn_gru_semantic_v2", C_BLUE, "o")):
        if m not in groups:
            continue
        ys = [_mean([r["results"][t]["macro_f1"] for r in groups[m]]) for t, _ in fps_tests]
        xs = [f for _, f in fps_tests]
        ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=7, label=_name(m))
        ax.annotate(f"{ys[-1]:.3f}", (xs[-1], ys[-1]), textcoords="offset points", xytext=(6, -3), fontsize=8, color=INK2)
    ax.set_xticks([10, 15, 30])
    ax.set_xlabel("테스트 FPS (학습은 15fps)")
    ax.set_ylabel("macro-F1")
    ax.legend(frameon=False, fontsize=8, loc="center left")
    ax.set_title("FPS 변화에 대한 강건성", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_fps_robustness.png")
    plt.close(fig)
    return "\n".join(md)


def synthetic_extra_section(res: dict[str, Any], tab_dir: Path) -> str:
    labels = res["labels"]
    groups = _group_records(res["records"])
    header = ["모델", "파라미터", "macro-F1(15fps)", "30fps(8프레임 연속 창)", "30fps(2프레임 간격 창)", "제동 이벤트 recall"] + [f"F1 {l}" for l in labels]
    rows = []
    for m, rs in groups.items():
        main = [r["results"]["main"] for r in rs]
        pcs = np.array([[np.nan if v is None else v for v in x["per_class_f1"]] for x in main], dtype=float)
        rows.append(
            [_name(m), f"{rs[0]['params']:,}", _ms([x["macro_f1"] for x in main]),
             _ms([r["results"]["test_30fps_mid"]["macro_f1"] for r in rs]),
             _ms([r["results"]["test_30fps_mid_dilated"]["macro_f1"] for r in rs if "test_30fps_mid_dilated" in r["results"]]),
             _ms([x["event_recall"] for x in main])]
            + [f"{np.nanmean(pcs[:, i]):.3f}" for i in range(len(labels))]
        )
    _write_csv(tab_dir / "synthetic_extra.csv", header, rows)
    md = ["### 표 S5. 보충 실험: v1+v2 결합 특징과 시간 기준 창 (시드 3개)"]
    md.append("시간 기준 창: 30fps 입력에서 2프레임 간격으로 8개를 골라 학습(15fps) 때와 같은 0.53초를 보게 한다.\n")
    md.append(_md_table(header, rows))
    return "\n".join(md)


def comma_section(res: dict[str, Any], fig_dir: Path, tab_dir: Path) -> str:
    groups = _group_records(res["records"])
    order = [m for m in ("majority", "rule_v2_ego", "mlp_last_v2", "gru_v2", "cnn_gru_single_v2", "mc_cnn_gru_balanced_v2", "mc_cnn_gru_semantic_v2", "mc_cnn_gru_semantic_v2_w16", "mc_cnn_gru_balanced_v1", "mc_cnn_gru_semantic_v1") if m in groups]
    ds = res["dataset"]
    md = ["### 실주행 영상(comma.ai speedchallenge): 데이터"]
    md.append(f"- 20fps 단일 영상 {ds['frames']:,} 프레임(약 17분), 라벨 분포(" + ", ".join(f"{l} {c}" for l, c in zip(res["labels"], ds["label_counts"], strict=True)) + ")")
    md.append("- 라벨은 속도 센서 값에서만 계산(시각 특징과 독립). 10개 시간 블록 교차검증, 블록 경계 ±1초 제외.")
    header = ["모델", "파라미터", "macro-F1 (20fps)", "제동 AUROC", "제동 이벤트 recall", "제동 프레임 recall", "오경보/분", "macro-F1 (10fps 재계산)"]
    rows = []
    for m in order:
        rs = groups[m]
        t20 = [r["test_20fps"] for r in rs]
        t10 = [r["test_10fps"] for r in rs if "test_10fps" in r]
        rows.append([
            _name(m),
            f"{rs[0].get('params', 0):,}",
            _ms([x["macro_f1"] for x in t20]),
            _ms([x.get("braking_auroc") for x in t20]) if t20[0].get("braking_auroc") is not None else "-",
            _ms([x["event_recall"] for x in t20]),
            _ms([x["event_frame_recall"] for x in t20]),
            _ms([x["false_alarms_per_min"] for x in t20], 2),
            _ms([x["macro_f1"] for x in t10]) if t10 else "-",
        ])
    md.append("\n### 표 R1. 실주행 영상 자차 운동 상태 추정 (10-fold 블록 교차검증 평균 ± 표준편차)")
    md.append(_md_table(header, rows))
    _write_csv(tab_dir / "comma_main.csv", header, rows)
    s2r = res.get("sim_to_real")
    if s2r:
        rows = [[k, f"{v['braking_auroc_all']:.3f}", f"{v['braking_auroc_moving']:.3f}"] for k, v in s2r.items()]
        md.append("\n### 표 R2. 합성→실영상 zero-shot: 합성 데이터로만 학습한 모델의 제동 구간 판별 AUROC")
        md.append("점수 = P(brake_warning)+P(hard_brake_risk). '이동 중'은 정지 프레임 제외.\n")
        md.append(_md_table(["모델/점수", "AUROC(전체)", "AUROC(이동 중)"], rows))
    return "\n".join(md)


def latency_section(res: dict[str, Any], tab_dir: Path) -> str:
    dev = res["device"]
    md = ["### 지연시간 측정 환경"]
    md.append(f"- {dev.get('cpu_model', dev.get('processor'))}, 논리 코어 {dev.get('cpu_count')}개, PyTorch {dev.get('torch')}, CUDA 사용 {dev.get('cuda_available')}")
    md.append("- Jetson Orin Nano: **미측정**(실기기 필요). 아래 수치는 클라우드 CPU 컨테이너 기준이다.")
    header = ["temporal 모델", "파라미터", "창", "PyTorch 1스레드 p50(ms)", "PyTorch 4스레드 p50(ms)", "ONNX RT 1스레드 p50(ms)", "ONNX 크기(KB)", "ONNX 최대 오차"]
    rows = []
    for t in res["temporal"]:
        rows.append([
            t["model"], f"{t['params']:,}", str(t["window"]),
            f"{t['torch_cpu_t1']['p50_ms']:.3f}", f"{t['torch_cpu_t4']['p50_ms']:.3f}",
            f"{t['onnx_cpu_t1']['p50_ms']:.3f}" if "onnx_cpu_t1" in t else "-",
            f"{t['onnx_bytes'] / 1024:.0f}" if "onnx_bytes" in t else "-",
            f"{t['onnx_max_abs_diff']:.1e}" if "onnx_max_abs_diff" in t else "-",
        ])
    md.append("\n### 표 L1. temporal 모델 지연시간(창 1개, 배치 1)")
    md.append(_md_table(header, rows))
    _write_csv(tab_dir / "latency_temporal.csv", header, rows)
    rows = [[str(y["imgsz"]), str(y["threads"]), f"{y['p50_ms']:.1f}", f"{y['p95_ms']:.1f}", f"{1000.0 / y['p50_ms']:.1f}"] for y in res.get("yolo", [])]
    if rows:
        md.append("\n### 표 L2. YOLOv8n(3클래스) CPU 지연시간")
        md.append(_md_table(["입력 크기", "스레드", "p50(ms)", "p95(ms)", "환산 FPS(p50)"], rows))
    return "\n".join(md)


def vla_section(res: dict[str, Any], fig_dir: Path, tab_dir: Path) -> str:
    plt = _setup_matplotlib()
    md = []
    recs = res["comma_action_curation"]["records"]
    by = _group_records(recs)
    md.append("### 표 V1. 행동(미래 가감속) 보조 head — 실주행 영상, 10-fold")
    rows = []
    for m, rs in by.items():
        ctx = [r["context"] for r in rs]
        row = [_name(m), _ms([c["macro_f1"] for c in ctx]), _ms([c["event_recall"] for c in ctx])]
        if "action_mae_model" in rs[0]:
            row += [
                _ms([r["action_mae_model"][0] for r in rs]), _ms([r["action_mae_model"][1] for r in rs]),
                _ms([r["action_mae_train_mean"][0] for r in rs]), _ms([r["action_mae_zero"][0] for r in rs]),
                _ms([r["action_corr_model"][0] for r in rs]),
            ]
        else:
            row += ["-"] * 5
        rows.append(row)
    header = ["모델", "맥락 macro-F1", "제동 이벤트 recall", "행동 MAE +0.5s", "행동 MAE +1.0s", "MAE(학습 평균 예측)", "MAE(0 예측)", "상관계수 +0.5s"]
    md.append(_md_table(header, rows))
    _write_csv(tab_dir / "vla_action.csv", header, rows)

    # 큐레이션 곡선
    budgets = ("0.1", "0.2", "0.3")
    act = by.get("mc_cnn_gru_semantic_v2_action", next(iter(by.values())))
    model_cov = [_mean([r["curation"][b]["model"]["event_coverage"] for r in act]) for b in budgets]
    model_fcov = [_mean([r["curation"][b]["model"]["frame_coverage"] for r in act]) for b in budgets]
    rand_cov = [_mean([r["curation"][b]["random_event_coverage"] for r in act]) for b in budgets]
    rand_fcov = [_mean([r["curation"][b]["random_frame_coverage"] for r in act]) for b in budgets]
    md.append("\n### 표 V2. 저장 예산 대비 제동 이벤트 회수율 — 실주행 영상(2초 클립 단위)")
    rows = [[f"{int(float(b) * 100)}%", f"{mc:.3f}", f"{rc:.3f}", f"{mf:.3f}", f"{rf:.3f}"] for b, mc, rc, mf, rf in zip(budgets, model_cov, rand_cov, model_fcov, rand_fcov, strict=True)]
    md.append(_md_table(["저장 예산", "이벤트 회수(모델)", "이벤트 회수(무작위)", "제동 프레임 회수(모델)", "제동 프레임 회수(무작위)"], rows))
    fig, ax = plt.subplots(figsize=(5.0, 3.3))
    xs = [float(b) * 100 for b in budgets]
    ax.plot(xs, model_fcov, color=C_BLUE, lw=2, marker="o", ms=7, label="맥락 모델 점수로 선택")
    ax.plot(xs, rand_fcov, color=C_NEUTRAL, lw=2, marker="s", ms=7, ls="--", label="무작위 선택")
    ax.set_xlabel("저장 예산(전체 프레임 대비 %)")
    ax.set_ylabel("제동 프레임 회수율")
    ax.set_xticks(xs)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("자동 큐레이션 효율(실주행 영상)", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_curation_comma.png")
    plt.close(fig)

    # 로봇
    rob = res["robot"]
    rg = _group_records(rob["records"])
    md.append("\n### 로봇(AMR) 도메인 데이터")
    ds = rob["dataset"]
    md.append(f"- 에피소드 {ds['episodes']}개, {ds['frames']:,} 프레임, 시나리오: " + ", ".join(f"{k} {v}" for k, v in ds["scenario_counts"].items()))
    md.append("- 라벨: " + ", ".join(rob["labels"]) + f" / 근접 접촉 발생 에피소드 {ds['episodes_with_contact']}개")
    rows = [[l] + [str(ds["label_counts"][p][i]) for p in ("train", "val", "test")] for i, l in enumerate(rob["labels"])]
    md.append(_md_table(["라벨", "train", "val", "test"], rows))
    md.append("\n### 표 V3. 로봇 도메인 맥락 인식 및 주행→로봇 전이 (테스트, 시드 평균 ± 표준편차)")
    rows = []
    order = [m for m in ("mlp_last_v2", "gru_v2", "cnn_gru_single_v2", "mc_cnn_gru_balanced_v2", "mc_cnn_gru_semantic_v2", "mc_cnn_gru_balanced_v1", "driving_pretrained_zero_shot", "robot_10pct_scratch", "robot_10pct_finetune_from_driving") if m in rg]
    for m in order:
        t = [r["test"] for r in rg[m]]
        rows.append([_name(m), f"{rg[m][0]['params']:,}", _ms([x["macro_f1"] for x in t]), _ms([x["event_recall"] for x in t]), _ms([x["event_frame_recall"] for x in t]), _ms([x["false_alarms_per_min"] for x in t], 2)])
    header = ["모델", "파라미터", "macro-F1", "감속/정지 이벤트 recall", "감속/정지 프레임 recall", "오경보/분"]
    md.append(_md_table(header, rows))
    _write_csv(tab_dir / "robot.csv", header, rows)
    tr = [m for m in ("driving_pretrained_zero_shot", "robot_10pct_scratch", "robot_10pct_finetune_from_driving", "mc_cnn_gru_semantic_v2") if m in rg]
    if tr:
        fig, ax = plt.subplots(figsize=(5.6, 3.0))
        vals = [_mean([r["test"]["macro_f1"] for r in rg[m]]) for m in tr]
        names = ["주행 모델\nzero-shot", "로봇 10%\n처음부터", "로봇 10%\n주행 사전학습", "로봇 100%\n처음부터"][: len(tr)]
        colors = [C_NEUTRAL, C_NEUTRAL, C_BLUE, C_NEUTRAL][: len(tr)]
        ax.bar(range(len(tr)), vals, color=colors, width=0.55)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8, color=INK2)
        ax.set_xticks(range(len(tr)), names, fontsize=8)
        ax.set_ylabel("로봇 테스트 macro-F1")
        ax.grid(axis="x", visible=False)
        ax.set_title("주행 → 실내 이동로봇 전이", loc="left", color=INK, fontsize=11)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_robot_transfer.png")
        plt.close(fig)
    ex = res.get("export", {})
    md.append("\n### 표 V4. VLA 학습용 내보내기 결과(LeRobot v2 구조)")
    md.append(_md_table(["데이터셋", "에피소드", "프레임"], [[k, str(v["episodes"]), f"{v['frames']:,}"] for k, v in ex.items()]))
    return "\n".join(md)


def build_report(out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    parts = ["# 자동 실험 결과 요약 (exp_100_paper_suite)", "", "이 문서는 `python -m vcp.experiments.run_suite report`가 결과 JSON에서 자동 생성한다. 수기 수정 금지.", ""]
    loaders = [
        ("synthetic_results.json", lambda r: synthetic_section(r, fig_dir, tab_dir)),
        ("synthetic_extra_results.json", lambda r: synthetic_extra_section(r, tab_dir)),
        ("comma_results.json", lambda r: comma_section(r, fig_dir, tab_dir)),
        ("vla_results.json", lambda r: vla_section(r, fig_dir, tab_dir)),
        ("latency_results.json", lambda r: latency_section(r, tab_dir)),
    ]
    for fname, fn in loaders:
        path = out_dir / fname
        if path.exists():
            parts.append(fn(json.loads(path.read_text(encoding="utf-8"))))
            parts.append("")
        else:
            parts.append(f"> `{fname}` 없음(해당 실험 미실행)\n")
    report = out_dir / "RESULTS.md"
    report.write_text("\n".join(parts), encoding="utf-8")
    logger.info("리포트 저장: %s", report)
    return report
