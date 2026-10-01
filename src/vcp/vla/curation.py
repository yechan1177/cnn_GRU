from __future__ import annotations

"""저장 예산 기반 클립 선별(자동 큐레이션).

온디바이스에서 모든 프레임을 저장/전송하지 않고, 맥락 모델의 이벤트 점수와 불확실성이
높은 구간만 고정 길이 클립으로 남긴다. VLA 데이터셋 구축 비용을 줄이는 핵심 단계다.
"""

import numpy as np


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
    """선택 마스크가 이벤트 프레임/이벤트 구간을 얼마나 포함하는지."""

    frame_cov = float((selected & is_event).sum() / max(1, is_event.sum()))
    n_ev = hit = 0
    for g in np.unique(group):
        sel = np.where(group == g)[0]
        ev, sm = is_event[sel], selected[sel]
        padded = np.concatenate([[False], ev, [False]])
        diff = np.diff(padded.astype(np.int8))
        for s, e in zip(np.where(diff == 1)[0], np.where(diff == -1)[0], strict=True):
            n_ev += 1
            hit += int(sm[s:e].any())
    return {"frame_coverage": frame_cov, "event_coverage": hit / max(1, n_ev), "n_events": n_ev}
