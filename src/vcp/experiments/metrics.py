from __future__ import annotations

"""평가 지표.

프레임 지표(정확도, macro-F1, 클래스별 F1)만으로는 이벤트 검출 성능과 오경보가
드러나지 않으므로 이벤트 단위 지표, 라벨 깜빡임, 보정 오차(ECE), bootstrap 신뢰구간을 함께 계산한다.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    mat = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(mat, (y_true.astype(np.int64), y_pred.astype(np.int64)), 1)
    return mat


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    mat = confusion(y_true, y_pred, n_classes)
    tp = np.diag(mat).astype(float)
    fp = mat.sum(axis=0) - tp
    fn = mat.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), np.nan)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """정답에 존재하는 클래스만 평균한 macro-F1."""

    f1 = per_class_f1(y_true, y_pred, n_classes)
    present = np.isin(np.arange(n_classes), np.unique(y_true))
    vals = f1[present]
    vals = np.nan_to_num(vals, nan=0.0)
    return float(vals.mean()) if len(vals) else 0.0


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """True 연속 구간 [start, end) 목록."""

    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


@dataclass(slots=True)
class EventStats:
    n_events: int
    detected: int
    latencies_s: list[float]
    false_alarms: int
    minutes: float


def event_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
    t: np.ndarray,
    event_classes: tuple[int, ...],
    pre_tol_s: float = 0.5,
    fa_tol_s: float = 1.0,
    min_pred_frames: int = 2,
) -> EventStats:
    """이벤트 단위 검출 지표.

    - GT 이벤트: 에피소드 내에서 event_classes 라벨이 연속된 구간
    - 검출: [시작 - pre_tol, 끝] 안에 예측 이벤트 프레임이 하나라도 있으면 검출
    - 지연: 첫 예측 프레임 시각 - GT 시작 시각(음수 = 선행 검출)
    - 오경보: min_pred_frames 이상 이어진 예측 이벤트 구간 중 GT 이벤트(±fa_tol)와 겹치지 않는 구간 수
    """

    n_events = detected = false_alarms = 0
    latencies: list[float] = []
    total_s = 0.0
    is_true = np.isin(y_true, event_classes)
    is_pred = np.isin(y_pred, event_classes)
    for g in np.unique(group):
        sel = np.where(group == g)[0]
        tt, gt, pr = t[sel], is_true[sel], is_pred[sel]
        if len(tt) > 1:
            total_s += float(tt[-1] - tt[0]) + float(np.median(np.diff(tt)))
        gt_runs = _runs(gt)
        for s, e in gt_runs:
            n_events += 1
            window = (tt >= tt[s] - pre_tol_s) & (tt <= tt[e - 1])
            hits = np.where(window & pr)[0]
            if len(hits):
                detected += 1
                latencies.append(float(tt[hits[0]] - tt[s]))
        for s, e in _runs(pr):
            if e - s < min_pred_frames:
                continue
            overlap = any(
                (tt[s] <= tt[ge - 1] + fa_tol_s) and (tt[e - 1] >= tt[gs] - fa_tol_s) for gs, ge in gt_runs
            )
            if not overlap:
                false_alarms += 1
    return EventStats(n_events, detected, latencies, false_alarms, total_s / 60.0)


def flicker_per_min(y: np.ndarray, group: np.ndarray, t: np.ndarray) -> float:
    """분당 라벨 전환 횟수(에피소드 경계 제외)."""

    changes = 0
    total_s = 0.0
    for g in np.unique(group):
        sel = np.where(group == g)[0]
        yy = y[sel]
        changes += int((yy[1:] != yy[:-1]).sum())
        if len(sel) > 1:
            total_s += float(t[sel][-1] - t[sel][0])
    return changes / max(1e-9, total_s / 60.0)


def boundary_f1_tolerance(
    true_boundary: np.ndarray,
    pred_score: np.ndarray,
    group: np.ndarray,
    threshold: float,
    tol: int = 2,
) -> dict[str, float]:
    """허용 오차(±tol 프레임) 경계 F1. 예측은 임계값 이상 구간의 시작 프레임."""

    tp = fp = fn = 0
    for g in np.unique(group):
        sel = np.where(group == g)[0]
        gt = np.where(true_boundary[sel] > 0)[0]
        runs = _runs(pred_score[sel] >= threshold)
        pred = np.array([s for s, _ in runs], dtype=np.int64)
        used = np.zeros(len(gt), dtype=bool)
        for p in pred:
            if len(gt):
                d = np.abs(gt - p)
                d[used] = 10**9
                j = int(np.argmin(d))
                if d[j] <= tol:
                    used[j] = True
                    tp += 1
                    continue
            fp += 1
        fn += int((~used).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney 기반 AUROC(동점 평균 순위)."""

    labels = labels.astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def summarize(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
    t: np.ndarray,
    n_classes: int,
    event_classes: tuple[int, ...],
    probs: np.ndarray | None = None,
) -> dict[str, Any]:
    """한 번의 평가 결과를 사전으로 정리한다."""

    ev = event_metrics(y_true, y_pred, group, t, event_classes)
    is_true = np.isin(y_true, event_classes)
    is_pred = np.isin(y_pred, event_classes)
    out: dict[str, Any] = {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": macro_f1(y_true, y_pred, n_classes),
        "per_class_f1": [None if np.isnan(v) else float(v) for v in per_class_f1(y_true, y_pred, n_classes)],
        "event_frame_recall": float((is_pred & is_true).sum() / max(1, is_true.sum())),
        "event_frame_precision": float((is_pred & is_true).sum() / max(1, is_pred.sum())),
        "event_recall": ev.detected / max(1, ev.n_events),
        "n_events": ev.n_events,
        "median_latency_s": float(np.median(ev.latencies_s)) if ev.latencies_s else None,
        "false_alarms_per_min": ev.false_alarms / max(1e-9, ev.minutes),
        "flicker_per_min": flicker_per_min(y_pred, group, t),
        "gt_flicker_per_min": flicker_per_min(y_true, group, t),
    }
    if probs is not None:
        out["ece"] = expected_calibration_error(probs, y_true)
    return out


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
    n_classes: int,
    n_boot: int = 300,
    seed: int = 0,
    event_classes: tuple[int, ...] | None = None,
) -> dict[str, list[float]]:
    """그룹(에피소드/블록) 재표집 bootstrap 95% 신뢰구간(macro-F1, 이벤트 프레임 recall)."""

    rng = np.random.default_rng(seed)
    groups = np.unique(group)
    pos = {g: np.where(group == g)[0] for g in groups}
    f1s, recs = [], []
    for _ in range(n_boot):
        pick = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([pos[g] for g in pick])
        f1s.append(macro_f1(y_true[idx], y_pred[idx], n_classes))
        if event_classes:
            tr = np.isin(y_true[idx], event_classes)
            pr = np.isin(y_pred[idx], event_classes)
            recs.append(float((tr & pr).sum() / max(1, tr.sum())))
    out = {"macro_f1": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))]}
    if recs:
        out["event_frame_recall"] = [float(np.percentile(recs, 2.5)), float(np.percentile(recs, 97.5))]
    return out
