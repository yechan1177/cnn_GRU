"""저장 예산 기반 클립 선별(자동 큐레이션).

온디바이스에서 모든 프레임을 저장/전송하지 않고, 맥락 모델의 이벤트 점수와 불확실성이
높은 구간만 고정 길이 클립으로 남긴다. VLA 데이터셋 구축 비용을 줄이는 핵심 단계다.

구성
- `select_clips`, `event_coverage`: 기존 VLA 스위트(`experiments/vla_suite.py`)가 쓰는 단순 선별·평가 함수.
- `METHODS`, `clip_starts`, `select`, `selection_stats`: 논문 비교 실험용 10개 선별법과 선별 품질 지표
  (계약: `docs/28a_VLA_모듈_인터페이스_계약.md`, 방법 정의: `docs/30_큐레이션_방법_정의.md`).

통제 변수
- 모든 방법은 같은 클립 길이(`clip_len`)와 같은 클립 수 k = max(1, round(budget_ratio·N/clip_len))를 고른다.
  따라서 선택 프레임 수가 방법에 관계없이 k·clip_len으로 같다(유효 클립 수가 k보다 적으면 모두 고른다).
- 클립은 같은 그룹(에피소드/블록) 안에서만 만들고 서로 겹치지 않는다(`clip_starts`).
- 점수 동률은 `rng`로 만든 무작위 키로 깬다(그룹 순서에 따른 편향 방지).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from ..features.semantic_v2 import V2_KEYS

logger = logging.getLogger(__name__)

METHODS: tuple[str, ...] = (
    "random",
    "uniform",
    "action_trigger",
    "rule_ittc",
    "uncertainty",
    "event",
    "coreset",
    "offline_loss",
    "oracle",
    "ours",
)

# 역 TTC(선행 차량 1/TTC) 특징의 v2 특징 벡터 내 인덱스. rule_ittc 입력은 보통 `X_v2[:, ITTC_FEATURE_INDEX]`다.
ITTC_FEATURE_INDEX: int = V2_KEYS.index("lead_inv_ttc")


# ---------------------------------------------------------------------------
# 기존 API(유지)
# ---------------------------------------------------------------------------
def select_clips(
    score: np.ndarray,
    group: np.ndarray,
    clip_len: int,
    budget_ratio: float,
    rng: np.random.Generator | None = None,
    random_baseline: bool = False,
) -> np.ndarray:
    """전체 프레임 중 budget_ratio 비율만큼을 clip_len 길이 클립으로 선택한 마스크를 반환한다.

    - 클립은 같은 그룹(에피소드/블록) 안에서 겹치지 않게 고른다.
    - random_baseline=True면 점수를 무시하고 균일 무작위로 고른다(비교군).
    - 기존 실험 재현성을 위해 전역 격자(0, clip_len, 2·clip_len, ...) 시작점을 그대로 쓴다.
      새 비교 실험은 `select`를 쓴다.
    """

    n = len(score)
    starts = np.arange(0, n - clip_len + 1, clip_len)
    valid = np.array([group[s] == group[s + clip_len - 1] for s in starts], dtype=bool)
    starts = starts[valid]
    if random_baseline:
        rng = rng or np.random.default_rng(0)
        order = rng.permutation(len(starts))
    else:
        clip_scores = np.array([score[s : s + clip_len].max() for s in starts])
        order = np.argsort(-clip_scores, kind="mergesort")
    k = max(1, int(round(budget_ratio * n / clip_len)))
    mask = np.zeros(n, dtype=bool)
    for s in starts[order[:k]]:
        mask[s : s + clip_len] = True
    return mask


def event_coverage(selected: np.ndarray, is_event: np.ndarray, group: np.ndarray) -> dict[str, float]:
    """선택 마스크가 이벤트 프레임/이벤트 구간을 얼마나 포함하는지.

    - frame_coverage: 선택된 이벤트 프레임 / 전체 이벤트 프레임
    - event_coverage: 한 프레임이라도 선택된 이벤트 구간 / 전체 이벤트 구간
      (이벤트 구간 = 같은 그룹 안에서 연속한 이벤트 프레임 묶음)

    구현은 그룹별 반복 대신 그룹 기준 안정 정렬 후 벡터 연산으로 계산한다(그룹별로 프레임을 모아
    구간을 세던 이전 구현과 결과가 같다). 대규모 풀(수십만 프레임, 수천 그룹)에서도 빠르다.
    """

    selected = np.asarray(selected, dtype=bool)
    is_event = np.asarray(is_event, dtype=bool)
    group = np.asarray(group)
    frame_cov = float((selected & is_event).sum() / max(1, is_event.sum()))
    order = np.argsort(group, kind="stable")
    g, ev, sm = group[order], is_event[order], selected[order]
    if len(ev) == 0:
        return {"frame_coverage": frame_cov, "event_coverage": 0.0, "n_events": 0}
    new_run = ev.copy()
    new_run[1:] &= ~ev[:-1] | (g[1:] != g[:-1])
    n_ev = int(new_run.sum())
    if n_ev == 0:
        return {"frame_coverage": frame_cov, "event_coverage": 0.0, "n_events": 0}
    run_id = np.cumsum(new_run) - 1
    hit_runs = np.unique(run_id[ev & sm])
    hit = len(hit_runs)
    return {"frame_coverage": frame_cov, "event_coverage": hit / max(1, n_ev), "n_events": n_ev}


# ---------------------------------------------------------------------------
# 비교 실험용 선별법
# ---------------------------------------------------------------------------
def clip_starts(group: np.ndarray, clip_len: int) -> np.ndarray:
    """그룹 경계를 넘지 않는 비중첩 클립 시작점(int64, 오름차순)을 반환한다.

    그룹 값이 같은 연속 구간(run)마다 구간 시작점에서 clip_len 간격으로 클립을 자르고,
    구간 끝에 clip_len보다 짧게 남는 꼬리는 버린다. 같은 그룹 id가 떨어진 두 구간에 나타나면
    각 구간을 따로 자른다(구간을 이어 붙인 클립은 만들지 않는다).
    """

    if clip_len < 1:
        raise ValueError(f"clip_len은 1 이상이어야 합니다: {clip_len}")
    group = np.asarray(group)
    n = len(group)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    change = np.flatnonzero(group[1:] != group[:-1]) + 1
    run_start = np.concatenate([[0], change]).astype(np.int64)
    run_len = np.diff(np.concatenate([run_start, [n]]))
    n_clips = run_len // clip_len
    total = int(n_clips.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    offsets = np.cumsum(n_clips) - n_clips
    within = np.arange(total, dtype=np.int64) - np.repeat(offsets, n_clips)
    return np.repeat(run_start, n_clips) + within * clip_len


def _clip_index(starts: np.ndarray, clip_len: int) -> np.ndarray:
    """클립 시작점 [K] → 클립 내 프레임 인덱스 [K, clip_len]."""

    return starts[:, None] + np.arange(clip_len, dtype=np.int64)[None, :]


def _check_input(name: str, arr: np.ndarray | None, n: int, method: str) -> np.ndarray:
    """방법에 필요한 입력이 있는지, 길이가 N인지 확인한다."""

    if arr is None:
        raise ValueError(f"method='{method}'에는 {name} 입력이 필요합니다.")
    arr = np.asarray(arr)
    if arr.shape[0] != n:
        raise ValueError(f"{name} 길이({arr.shape[0]})가 group 길이({n})와 다릅니다.")
    return arr


def _rank_desc(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """점수 내림차순 순서. 동률은 무작위로 깨고, NaN은 최하위로 보낸다."""

    s = np.asarray(scores, dtype=np.float64)
    s = np.where(np.isnan(s), -np.inf, s)
    tie = rng.random(len(s))
    # lexsort는 마지막 키가 1순위다.
    return np.lexsort((tie, -s))


def _take_with_cap(order: np.ndarray, clip_group: np.ndarray, k: int, cap: int | None) -> np.ndarray:
    """순서대로 k개를 고르되 그룹당 최대 cap개로 제한한다. 고른 클립 인덱스를 반환한다."""

    if cap is None:
        return order[:k]
    if cap < 1:
        raise ValueError(f"per_group_cap은 1 이상이어야 합니다: {cap}")
    _, inv = np.unique(clip_group[order], return_inverse=True)
    # 각 클립이 순서상 자기 그룹에서 몇 번째인지(0부터) 계산한다.
    sort_idx = np.argsort(inv, kind="stable")
    sorted_inv = inv[sort_idx]
    first = np.concatenate([[True], sorted_inv[1:] != sorted_inv[:-1]])
    run_start = np.maximum.accumulate(np.where(first, np.arange(len(inv)), 0))
    rank_in_group = np.empty(len(inv), dtype=np.int64)
    rank_in_group[sort_idx] = np.arange(len(inv)) - run_start
    allowed = order[rank_in_group < cap]
    return allowed[:k]


def _allocate_proportional(weights: np.ndarray, capacity: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """총 k개를 weights에 비례해 나누되 capacity를 넘지 않게 한다(최대 나머지 방식, 동률은 무작위)."""

    alloc = np.zeros(len(weights), dtype=np.int64)
    remaining = int(min(k, capacity.sum()))
    active = capacity > 0
    while remaining > 0 and active.any():
        w = np.where(active, weights, 0.0).astype(np.float64)
        quota = remaining * w / w.sum()
        room = capacity - alloc
        base = np.minimum(np.floor(quota).astype(np.int64), room)
        alloc += base
        remaining -= int(base.sum())
        room = capacity - alloc
        if remaining > 0:
            frac = np.where(active & (room > 0), quota - np.floor(quota), -1.0)
            cand = np.flatnonzero(frac >= 0)
            order = cand[np.lexsort((rng.random(len(cand)), -frac[cand]))]
            give = order[:remaining]
            alloc[give] += 1
            remaining -= len(give)
        active = (capacity - alloc) > 0
    return alloc


def _uniform_select(starts: np.ndarray, group: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """그룹 길이에 비례해 클립 수를 배분하고, 각 그룹 안에서 등간격으로 클립을 고른다."""

    clip_group = group[starts]
    uniq_clip_groups, g_inv, g_count = np.unique(clip_group, return_inverse=True, return_counts=True)
    uniq_groups, frame_count = np.unique(group, return_counts=True)
    # 그룹 길이(프레임 수)를 가중치로 쓴다. 클립이 하나도 없는 그룹은 uniq_clip_groups에 나타나지 않는다.
    weights = frame_count[np.searchsorted(uniq_groups, uniq_clip_groups)].astype(np.float64)
    alloc = _allocate_proportional(weights, g_count.astype(np.int64), k, rng)
    chosen: list[np.ndarray] = []
    order = np.argsort(g_inv, kind="stable")  # 그룹별 클립(시간순)
    offsets = np.concatenate([[0], np.cumsum(g_count)])
    for gi in np.flatnonzero(alloc):
        n_g, m = int(g_count[gi]), int(alloc[gi])
        local = np.floor((np.arange(m) + 0.5) * n_g / m).astype(np.int64)
        chosen.append(order[offsets[gi] + local])
    return np.concatenate(chosen) if chosen else np.zeros(0, dtype=np.int64)


def _coreset_select(emb: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-center greedy(Sener & Savarese, 2018). 표준화된 임베딩 [K, D]에서 k개 중심을 고른다.

    첫 중심은 rng로 무작위로 고르고, 이후 매 단계 현재 중심 집합까지의 최소 거리가 가장 큰 클립을 추가한다.
    거리 갱신은 ||x||² − 2x·c + ||c||² 행렬-벡터 곱으로 계산해 단계당 O(K·D)다(전체 O(k·K·D)).
    """

    n = emb.shape[0]
    x = emb.astype(np.float32, copy=False)
    sq = np.einsum("ij,ij->i", x, x)
    min_d = np.full(n, np.inf, dtype=np.float32)
    chosen = np.empty(k, dtype=np.int64)
    c = int(rng.integers(n))
    for i in range(k):
        chosen[i] = c
        d = sq - 2.0 * (x @ x[c]) + sq[c]
        np.minimum(min_d, d, out=min_d)
        min_d[c] = -np.inf
        if i + 1 < k:
            c = int(np.argmax(min_d))
    return chosen


def select(
    method: str,
    budget_ratio: float,
    group: np.ndarray,
    clip_len: int,
    rng: np.random.Generator,
    *,
    event_score: np.ndarray | None = None,
    entropy: np.ndarray | None = None,
    features: np.ndarray | None = None,
    action: np.ndarray | None = None,
    ittc: np.ndarray | None = None,
    oracle: np.ndarray | None = None,
    loss: np.ndarray | None = None,
    lam: float = 0.5,
    reservoir: float = 0.3,
    per_group_cap: int | None = None,
) -> np.ndarray:
    """선별법 `method`로 예산 budget_ratio만큼 클립을 골라 프레임 마스크 bool [N]을 반환한다.

    방법별 클립 점수(클립 = 같은 그룹 안의 연속 clip_len 프레임, 상위 k개 선택)
    - random: 무작위 k개
    - uniform: 그룹 길이 비례 배분 후 그룹 안에서 등간격
    - action_trigger: max(−action) (감속 크기). action이 [N, D]면 모든 열의 최댓값
    - rule_ittc: max(ittc) (역 TTC, 보통 `X_v2[:, ITTC_FEATURE_INDEX]`)
    - uncertainty: mean(entropy)
    - event: max(event_score)
    - coreset: [mean(features), std(features)] 임베딩(표준화) 위의 k-center greedy
    - offline_loss: mean(loss)
    - oracle: mean(oracle) (위험 프레임 비율)
    - ours: max(event_score) + lam·mean(entropy) 상위 round((1−reservoir)·k)개
      (per_group_cap 적용) + 나머지는 선택되지 않은 클립에서 무작위

    Raises:
        ValueError: 알 수 없는 방법, 잘못된 예산·클립 길이, 필요한 입력 누락·길이 불일치.
    """

    if method not in METHODS:
        raise ValueError(f"알 수 없는 선별법: {method} (가능: {METHODS})")
    if not (0.0 < budget_ratio <= 1.0):
        raise ValueError(f"budget_ratio는 (0, 1] 범위여야 합니다: {budget_ratio}")
    if not (0.0 <= reservoir <= 1.0):
        raise ValueError(f"reservoir는 [0, 1] 범위여야 합니다: {reservoir}")
    group = np.asarray(group)
    n = len(group)
    starts = clip_starts(group, clip_len)
    n_clips = len(starts)
    mask = np.zeros(n, dtype=bool)
    if n_clips == 0:
        logger.warning("유효 클립이 없습니다(N=%d, clip_len=%d). 빈 마스크를 반환합니다.", n, clip_len)
        return mask
    k_req = max(1, int(round(budget_ratio * n / clip_len)))
    k = min(k_req, n_clips)
    if k < k_req:
        logger.warning("요청 클립 수 %d가 유효 클립 수 %d보다 많아 %d개만 고릅니다.", k_req, n_clips, k)
    idx = _clip_index(starts, clip_len)
    t0 = time.perf_counter()

    if method == "random":
        chosen = rng.permutation(n_clips)[:k]
    elif method == "uniform":
        chosen = _uniform_select(starts, group, k, rng)
    elif method == "action_trigger":
        a = _check_input("action", action, n, method).astype(np.float64)
        decel = -a if a.ndim == 1 else (-a.reshape(n, -1)).max(axis=1)
        chosen = _rank_desc(decel[idx].max(axis=1), rng)[:k]
    elif method == "rule_ittc":
        v = _check_input("ittc", ittc, n, method).astype(np.float64)
        chosen = _rank_desc(v[idx].max(axis=1), rng)[:k]
    elif method == "uncertainty":
        v = _check_input("entropy", entropy, n, method).astype(np.float64)
        chosen = _rank_desc(v[idx].mean(axis=1), rng)[:k]
    elif method == "event":
        v = _check_input("event_score", event_score, n, method).astype(np.float64)
        chosen = _rank_desc(v[idx].max(axis=1), rng)[:k]
    elif method == "coreset":
        f = _check_input("features", features, n, method).astype(np.float32).reshape(n, -1)
        clip_f = f[idx]  # [K, L, D]
        emb = np.concatenate([clip_f.mean(axis=1), clip_f.std(axis=1)], axis=1)
        mu, sd = emb.mean(axis=0), emb.std(axis=0)
        emb = (emb - mu) / np.where(sd > 1e-8, sd, 1.0)
        chosen = _coreset_select(emb, k, rng)
    elif method == "offline_loss":
        v = _check_input("loss", loss, n, method).astype(np.float64)
        chosen = _rank_desc(v[idx].mean(axis=1), rng)[:k]
    elif method == "oracle":
        v = _check_input("oracle", oracle, n, method).astype(bool).astype(np.float64)
        chosen = _rank_desc(v[idx].mean(axis=1), rng)[:k]
    else:  # ours
        ev = _check_input("event_score", event_score, n, method).astype(np.float64)
        en = _check_input("entropy", entropy, n, method).astype(np.float64)
        score = ev[idx].max(axis=1) + lam * en[idx].mean(axis=1)
        n_top = int(round((1.0 - reservoir) * k))
        top = _take_with_cap(_rank_desc(score, rng), group[starts], n_top, per_group_cap)
        rest = np.setdiff1d(np.arange(n_clips), top, assume_unique=True)
        extra = rng.permutation(rest)[: k - len(top)]
        chosen = np.concatenate([top, extra])

    chosen = np.asarray(chosen, dtype=np.int64)
    if len(chosen) != k or len(np.unique(chosen)) != k:
        raise RuntimeError(f"{method}: 고른 클립 수가 {len(np.unique(chosen))}개로 예산 {k}개와 다릅니다.")
    mask[idx[chosen].ravel()] = True
    logger.debug(
        "선별 %s: 예산 %.3f, 클립 %d/%d개(%d 프레임), %.3f초",
        method, budget_ratio, k, n_clips, int(mask.sum()), time.perf_counter() - t0,
    )
    return mask


def selection_stats(
    mask: np.ndarray,
    labels: np.ndarray,
    group: np.ndarray,
    n_classes: int,
    hazard_ids: tuple[int, ...],
) -> dict[str, Any]:
    """선택 마스크의 품질 지표를 계산한다(모든 값은 JSON 직렬화 가능한 파이썬 타입).

    반환 키
    - n_selected, selected_ratio: 선택 프레임 수와 전체 대비 비율
    - class_hist: 선택 프레임의 클래스 비율 [n_classes] (선택이 없으면 모두 0)
    - label_entropy: 선택 라벨 분포의 섀넌 엔트로피(nat), label_entropy_norm: log(n_classes)로 나눈 값
    - hazard_frame_recall: 위험 프레임(labels ∈ hazard_ids) 중 선택된 비율
    - hazard_event_recall, n_hazard_events: 위험 이벤트 구간 회수율과 구간 수(`event_coverage`)
    - group_coverage, n_groups_selected, n_groups: 선택 프레임이 하나라도 있는 그룹 비율
    """

    mask = np.asarray(mask, dtype=bool)
    labels = np.asarray(labels).astype(np.int64)
    group = np.asarray(group)
    if not (len(mask) == len(labels) == len(group)):
        raise ValueError("mask, labels, group 길이가 같아야 합니다.")
    if n_classes < 1:
        raise ValueError(f"n_classes는 1 이상이어야 합니다: {n_classes}")
    n_sel = int(mask.sum())
    counts = np.bincount(labels[mask], minlength=n_classes)[:n_classes].astype(np.float64)
    hist = counts / counts.sum() if counts.sum() > 0 else np.zeros(n_classes)
    nz = hist[hist > 0]
    ent = float(-(nz * np.log(nz)).sum()) if len(nz) else 0.0
    ent_norm = ent / float(np.log(n_classes)) if n_classes > 1 else 0.0
    is_hazard = np.isin(labels, np.asarray(hazard_ids, dtype=np.int64))
    cov = event_coverage(mask, is_hazard, group)
    uniq = np.unique(group)
    sel_groups = np.unique(group[mask])
    return {
        "n_selected": n_sel,
        "selected_ratio": n_sel / max(1, len(mask)),
        "class_hist": hist.tolist(),
        "label_entropy": ent,
        "label_entropy_norm": ent_norm,
        "hazard_frame_recall": float(cov["frame_coverage"]),
        "hazard_event_recall": float(cov["event_coverage"]),
        "n_hazard_frames": int(is_hazard.sum()),
        "n_hazard_events": int(cov["n_events"]),
        "group_coverage": len(sel_groups) / max(1, len(uniq)),
        "n_groups_selected": int(len(sel_groups)),
        "n_groups": int(len(uniq)),
    }
