from __future__ import annotations

"""프레임 시퀀스 -> 고정 길이 창(window) 구성과 누수 없는 분할 유틸리티."""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class FrameTable:
    """에피소드(또는 시간 블록) 단위로 이어진 프레임 테이블."""

    X: np.ndarray  # [N, D] float32
    y: np.ndarray  # [N] int
    group: np.ndarray  # [N] int, 에피소드/블록 ID (분할 단위)
    boundary: np.ndarray  # [N] int8
    t: np.ndarray  # [N] float32 (초)
    extra: dict[str, np.ndarray]


def window_index(group: np.ndarray, window: int, stride: int = 1, mask: np.ndarray | None = None) -> np.ndarray:
    """각 대상 프레임에 대한 과거 window 인덱스 행렬 [M, W]를 만든다.

    - 같은 group 안에서만 과거 프레임을 사용한다(그룹 경계를 넘지 않음).
    - 그룹 시작 부분은 -1(=0 벡터 패딩)로 채운다. 런타임 실행기의 초기 0 패딩과 같다.
    - mask가 주어지면 해당 프레임만 대상 프레임으로 사용한다.
    """

    n = len(group)
    idx = np.arange(n)
    starts = np.zeros(n, dtype=np.int64)
    change = np.ones(n, dtype=bool)
    change[1:] = group[1:] != group[:-1]
    start_positions = np.where(change)[0]
    starts = start_positions[np.searchsorted(start_positions, idx, side="right") - 1]

    targets = idx
    if mask is not None:
        targets = targets[mask[targets]]
    if stride > 1:
        rel = targets - starts[targets]
        targets = targets[(rel % stride) == 0]

    offsets = np.arange(-window + 1, 1)
    mat = targets[:, None] + offsets[None, :]
    valid = mat >= starts[targets][:, None]
    return np.where(valid, mat, -1).astype(np.int64)


def gather_windows(X: np.ndarray, index: np.ndarray) -> np.ndarray:
    """인덱스 행렬로 [M, W, D] 텐서를 만든다(-1은 0 벡터)."""

    padded = np.vstack([X, np.zeros((1, X.shape[1]), dtype=X.dtype)])
    safe = np.where(index < 0, len(X), index)
    return padded[safe]


def split_groups(groups: np.ndarray, val_ratio: float, test_ratio: float, seed: int) -> dict[str, np.ndarray]:
    """그룹(에피소드) 단위 무작위 분할. 같은 에피소드가 두 분할에 들어가지 않는다."""

    unique = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(unique)
    n_test = max(1, int(round(len(unique) * test_ratio)))
    n_val = max(1, int(round(len(unique) * val_ratio)))
    return {
        "test": np.sort(perm[:n_test]),
        "val": np.sort(perm[n_test : n_test + n_val]),
        "train": np.sort(perm[n_test + n_val :]),
    }


def blocked_folds(n_frames: int, n_blocks: int, purge: int) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
    """단일 장시간 영상용 시간 블록 교차검증.

    영상을 n_blocks개의 연속 블록으로 나누고, fold k는 블록 k를 test, 블록 k+1을 val,
    나머지를 train으로 쓴다. 블록 경계 앞뒤 purge 프레임은 어느 분할에서도 대상 프레임으로
    쓰지 않아 인접 창의 정보 누수를 막는다.
    반환: (프레임별 블록 ID, fold별 {train/val/test: 블록 ID 배열, purge_mask})
    """

    block = np.minimum((np.arange(n_frames) * n_blocks) // n_frames, n_blocks - 1).astype(np.int64)
    edge = np.zeros(n_frames, dtype=bool)
    change = np.where(block[1:] != block[:-1])[0] + 1
    for c in change:
        edge[max(0, c - purge) : min(n_frames, c + purge)] = True
    folds = []
    for k in range(n_blocks):
        test = np.array([k])
        val = np.array([(k + 1) % n_blocks])
        train = np.array([b for b in range(n_blocks) if b not in (k, (k + 1) % n_blocks)])
        folds.append({"train": train, "val": val, "test": test, "purge_mask": ~edge})
    return block, folds
