from __future__ import annotations

"""룰 기반 비교군과 하이브리드 룰 게이트(벡터화 구현).

모든 임계값은 검증(val) 분할에서만 grid search로 고르고, 테스트 분할에는 고정값으로 적용한다.
(2026-03 버전은 같은 검증셋으로 튜닝과 보고를 함께 해 성능이 과대평가되었다.)
"""

import itertools
from collections.abc import Callable
from typing import Any

import numpy as np

from ..features.registry import get_feature_spec
from .metrics import macro_f1


def _col(X: np.ndarray, version: str, key: str) -> np.ndarray:
    return X[:, get_feature_spec(version).index(key)]


# ----------------------------------------------------------------------
# 6맥락(주행) 룰
# ----------------------------------------------------------------------
def rule_v1_driving(X: np.ndarray, labels: list[str], p: dict[str, float]) -> np.ndarray:
    """2026-03 'YOLO+rule' 비교군과 같은 형태의 v1 특징 룰."""

    li = {n: i for i, n in enumerate(labels)}
    roi = _col(X, "v1", "roi_risk")
    center = _col(X, "v1", "center_closeness")
    looming = _col(X, "v1", "looming_score")
    occl = _col(X, "v1", "occlusion_score")
    motion = _col(X, "v1", "motion_delta")
    det = _col(X, "v1", "det_norm")
    warn = (roi >= p["roi_thr"]) & (center >= p["center_thr"]) & (looming >= p["looming_warn_thr"])
    hard = warn & (looming >= p["looming_hard_thr"]) & ((occl >= p["occlusion_thr"]) | (motion <= p["motion_stop_thr"]))
    out = np.full(len(X), li["normal_drive"], dtype=np.int64)
    out[roi >= p["roi_thr"]] = li["front_vehicle_follow"]
    out[det >= p["dense_thr"]] = li["dense_traffic"]
    out[warn] = li["brake_warning"]
    out[hard] = li["hard_brake_risk"]
    return out


RULE_V1_GRID = {
    "roi_thr": [0.25, 0.30, 0.35, 0.45],
    "center_thr": [0.6, 0.72, 0.8],
    "looming_warn_thr": [0.006, 0.02, 0.05, 0.1],
    "looming_hard_thr": [0.014, 0.05, 0.1, 0.2],
    "occlusion_thr": [0.5, 0.58],
    "motion_stop_thr": [0.01, 0.05],
    "dense_thr": [0.15, 0.2, 0.3, 1.1],
}


def rule_v2_driving(X: np.ndarray, labels: list[str], p: dict[str, float]) -> np.ndarray:
    """v2 특징(역 TTC, 선행차 크기, VRU 접근) 기반 룰."""

    li = {n: i for i, n in enumerate(labels)}
    present = _col(X, "v2", "lead_present") > 0.5
    size = _col(X, "v2", "lead_size")
    inv_ttc = _col(X, "v2", "lead_inv_ttc")
    vru = _col(X, "v2", "vru_approach")
    vru_size = _col(X, "v2", "vru_size")
    det = _col(X, "v2", "det_norm")
    out = np.full(len(X), li["normal_drive"], dtype=np.int64)
    out[present & (size >= p["follow_size"])] = li["front_vehicle_follow"]
    out[det >= p["dense_thr"]] = li["dense_traffic"]
    warn = (present & (inv_ttc >= p["warn_inv_ttc"])) | ((vru >= p["vru_thr"]) & (vru_size >= p["vru_size"]))
    hard = present & (inv_ttc >= p["hard_inv_ttc"]) & (size >= p["hard_size"])
    out[warn] = li["brake_warning"]
    out[hard] = li["hard_brake_risk"]
    return out


RULE_V2_GRID = {
    "follow_size": [0.05, 0.1, 0.15, 0.25],
    "dense_thr": [0.15, 0.2, 0.3, 1.1],
    "warn_inv_ttc": [0.1, 0.2, 0.3, 0.5],
    "hard_inv_ttc": [0.3, 0.5, 0.8],
    "hard_size": [0.1, 0.2, 0.3],
    "vru_thr": [0.05, 0.1, 0.2, 1.1],
    "vru_size": [0.05, 0.1],
}


# ----------------------------------------------------------------------
# 자차 운동 상태(실데이터 comma) 룰
# ----------------------------------------------------------------------
def rule_v2_ego(X: np.ndarray, labels: list[str], p: dict[str, float]) -> np.ndarray:
    """선행차 접근/이탈 신호만으로 자차 운동 상태를 추정하는 단순 룰."""

    li = {n: i for i, n in enumerate(labels)}
    present = _col(X, "v2", "lead_present") > 0.5
    size = _col(X, "v2", "lead_size")
    rate = _col(X, "v2", "lead_scale_rate")
    out = np.full(len(X), li["cruise"], dtype=np.int64)
    out[present & (rate <= -p["accel_rate"])] = li["accelerating"]
    out[present & (rate >= p["brake_rate"])] = li["braking"]
    out[present & (size >= p["stop_size"]) & (np.abs(rate) < p["still_rate"])] = li["stopped"]
    return out


RULE_EGO_GRID = {
    "accel_rate": [0.02, 0.05, 0.1, 1.1],
    "brake_rate": [0.02, 0.05, 0.1, 0.2],
    "stop_size": [0.2, 0.3, 0.4, 1.1],
    "still_rate": [0.01, 0.02, 0.05],
}


def tune_rule(
    fn: Callable[[np.ndarray, list[str], dict[str, float]], np.ndarray],
    grid: dict[str, list[float]],
    X_val: np.ndarray,
    y_val: np.ndarray,
    labels: list[str],
) -> tuple[dict[str, float], float]:
    """검증셋 macro-F1 최대 파라미터를 고른다."""

    keys = list(grid)
    best_p: dict[str, float] = {}
    best = -1.0
    for values in itertools.product(*(grid[k] for k in keys)):
        p = dict(zip(keys, values, strict=True))
        score = macro_f1(y_val, fn(X_val, labels, p), len(labels))
        if score > best:
            best, best_p = score, p
    return best_p, best


# ----------------------------------------------------------------------
# 하이브리드 룰 게이트(배포 UI의 predict_sequence와 같은 논리, v1 특징 전용)
# ----------------------------------------------------------------------
def hybrid_gate_v1(
    probs: np.ndarray,
    boundary: np.ndarray,
    X_last: np.ndarray,
    labels: list[str],
    p: dict[str, float],
) -> np.ndarray:
    li = {n: i for i, n in enumerate(labels)}
    fi, wi, hi = li["front_vehicle_follow"], li["brake_warning"], li["hard_brake_risk"]
    roi = _col(X_last, "v1", "roi_risk")
    center = _col(X_last, "v1", "center_closeness")
    looming = _col(X_last, "v1", "looming_score")
    occl = _col(X_last, "v1", "occlusion_score")
    motion = _col(X_last, "v1", "motion_delta")
    warn_feat = (roi >= p["roi_thr"]) & (center >= p["center_thr"]) & (looming >= p["looming_warn_thr"])
    hard_feat = warn_feat & (looming >= p["looming_hard_thr"]) & ((occl >= p["occlusion_thr"]) | (motion <= p["motion_stop_thr"]))
    warn_sig = (boundary >= p["warn_boundary_thr"]) & ((probs[:, fi] >= p["follow_prob_thr"]) | (probs[:, wi] >= p["warn_prob_thr"]))
    hard_sig = (boundary >= p["hard_boundary_thr"]) & (
        (probs[:, fi] >= p["follow_prob_thr"]) | (probs[:, wi] >= p["warn_prob_thr"]) | (probs[:, hi] >= p["hard_prob_thr"])
    )
    warn = warn_feat | warn_sig
    hard = hard_feat | hard_sig
    boosted = probs.copy()
    boosted[:, wi] += np.where(warn, p["warn_boost"], 0.0)
    boosted[:, hi] += np.where(hard, p["hard_boost"], 0.0)
    final = boosted.argmax(axis=1)
    final = np.where(hard & ((probs[:, hi] >= p["hard_prob_thr"]) | (boundary >= p["hard_boundary_thr"])), hi, final)
    final = np.where(~hard & warn & ((probs[:, wi] >= p["warn_prob_thr"]) | (boundary >= p["warn_boundary_thr"])), wi, final)
    return final


HYBRID_DEFAULT: dict[str, float] = {
    "roi_thr": 0.30,
    "center_thr": 0.72,
    "looming_warn_thr": 0.006,
    "looming_hard_thr": 0.014,
    "occlusion_thr": 0.58,
    "motion_stop_thr": 0.01,
    "warn_boundary_thr": 0.82,
    "hard_boundary_thr": 0.93,
    "warn_prob_thr": 0.05,
    "hard_prob_thr": 0.03,
    "follow_prob_thr": 0.20,
    "warn_boost": 0.22,
    "hard_boost": 0.30,
}

HYBRID_GRID = {
    "roi_thr": [0.30, 0.45],
    "center_thr": [0.72, 0.8],
    "looming_warn_thr": [0.006, 0.05, 0.1],
    "looming_hard_thr": [0.014, 0.1, 0.2],
    "warn_boundary_thr": [0.82, 0.9, 1.01],
    "hard_boundary_thr": [0.93, 1.01],
}


def tune_hybrid(
    probs: np.ndarray,
    boundary: np.ndarray,
    X_last: np.ndarray,
    y: np.ndarray,
    labels: list[str],
) -> tuple[dict[str, float], float]:
    keys = list(HYBRID_GRID)
    best, best_p = -1.0, dict(HYBRID_DEFAULT)
    for values in itertools.product(*(HYBRID_GRID[k] for k in keys)):
        p = dict(HYBRID_DEFAULT)
        p.update(dict(zip(keys, values, strict=True)))
        score = macro_f1(y, hybrid_gate_v1(probs, boundary, X_last, labels, p), len(labels))
        if score > best:
            best, best_p = score, p
    return best_p, best


RULES: dict[str, tuple[Callable[..., np.ndarray], dict[str, list[float]], str]] = {
    "rule_v1": (rule_v1_driving, RULE_V1_GRID, "v1"),
    "rule_v2": (rule_v2_driving, RULE_V2_GRID, "v2"),
    "rule_v2_ego": (rule_v2_ego, RULE_EGO_GRID, "v2"),
}


def describe_params(p: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v}" for k, v in p.items())
