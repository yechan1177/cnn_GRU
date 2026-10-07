from __future__ import annotations

"""VLA-lite 관측 규격(학습·폐루프 공통).

- 영상: `render_frame`으로 만든 64×64 RGB 프레임 2장(현재 t, 과거 t−HISTORY_OFFSET)을 채널로 쌓는다.
  에피소드 초반(t < HISTORY_OFFSET)에는 과거 프레임으로 프레임 0을 쓴다.
- proprio: 자차 속도 / speed_scale(domain). 목표 속도는 넣지 않는다(지시문으로만 전달).
- 행동 정규화: 정책 출력 가속도 / accel_scale(domain).
- 검출 특징(v3, docs/28a "v3 추가 계약"): 노이즈 검출에서 v1(16)+v2(16)=32차원 특징을 인과적으로 계산한다.
  풀(`pool.py`)과 폐루프(`closed_loop.py`)가 같은 `OnlineFeatureTracker`를 써서 계산 경로가 같다.
  정책 관측 `"features"`는 [현재 t, max(t−HISTORY_OFFSET, 0)] 두 프레임을 쌓은 float32 [B,2,32]다.
- 특징 이력 일반화(v4, docs/36 6절·docs/29 10절): `"features"`는 float32 [B,H,32]이고, 인덱스 k(0..H−1)는
  프레임 max(t − k·s, 0)(에피소드 시작에서 잘라 냄)의 특징이다. k=0이 현재 프레임이다.
  기본값 H=2, s=2(`DEFAULT_FEATURE_HISTORY`, `DEFAULT_FEATURE_STRIDE`)이면 v3와 똑같은 [B,2,32]다.
  영상 관측은 v4에서도 (t, t−HISTORY_OFFSET) 2장 그대로다.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..features import build_feature_extractor

IMG_SIZE: tuple[int, int] = (64, 64)  # (H, W)
HISTORY_OFFSET: int = 2

# 검출 특징 설정(풀·폐루프 공통). sim.dataset과 같은 값이다.
FEATURE_CONF_THRESHOLD: float = 0.45
FEATURE_MAX_DET: int = 30
FEATURE_DIM: int = 32  # v1 16 + v2 16

# 특징 이력 기본값(v3와 같은 [t, t−2]). v4 개발 설정은 H=8, s=2(docs/36 3절)다.
DEFAULT_FEATURE_HISTORY: int = 2
DEFAULT_FEATURE_STRIDE: int = 2

_SPEED_SCALE: dict[str, float] = {"driving": 30.0, "robot": 3.0}
_ACCEL_SCALE: dict[str, float] = {"driving": 4.0, "robot": 1.0}


def _key(domain: str, table: dict[str, float]) -> str:
    key = str(domain).strip().lower()
    if key not in table:
        raise KeyError(f"지원하지 않는 도메인: {domain} (지원: {sorted(table)})")
    return key


def speed_scale(domain: str) -> float:
    """속도 정규화 상수(m/s). driving 30.0, robot 3.0."""

    return _SPEED_SCALE[_key(domain, _SPEED_SCALE)]


def accel_scale(domain: str) -> float:
    """가속도 정규화 상수(m/s^2). driving 4.0, robot 1.0."""

    return _ACCEL_SCALE[_key(domain, _ACCEL_SCALE)]


def history_index(t: int | np.ndarray) -> int | np.ndarray:
    """프레임 t의 과거 프레임 인덱스(t−HISTORY_OFFSET, 0 미만이면 0). 에피소드 내 상대 인덱스 기준."""

    if isinstance(t, np.ndarray):
        return np.maximum(t - HISTORY_OFFSET, 0)
    return max(int(t) - HISTORY_OFFSET, 0)


def stack_frames(cur: np.ndarray, prev: np.ndarray) -> np.ndarray:
    """uint8 [...,H,W,3] 두 장을 [...,H,W,6]으로 쌓는다(앞 3채널=현재, 뒤 3채널=과거)."""

    if cur.shape != prev.shape or cur.shape[-1] != 3:
        raise ValueError(f"프레임 형태 불일치: cur={cur.shape}, prev={prev.shape}")
    return np.concatenate([cur, prev], axis=-1).astype(np.uint8, copy=False)


def proprio(ego_v: float | np.ndarray, domain: str) -> np.ndarray:
    """float32 [...,1] = ego_v / speed_scale(domain)."""

    v = np.asarray(ego_v, dtype=np.float32) / np.float32(speed_scale(domain))
    return v[..., None]


class OnlineFeatureTracker:
    """에피소드 1개의 v1+v2 검출 특징을 프레임 순서대로 계산하는 상태 추적기.

    v1·v2 추출기는 직전 프레임 상태를 가지므로 에피소드마다 새로 만들거나 `reset()`해야 한다.
    `update(frame)`은 (v1 16개, v2 16개) 리스트를 돌려준다. 풀은 이 값을 그대로 X_v1/X_v2로 저장하고,
    폐루프는 `update_vector`로 float32 [32](v1 → v2 순서, `X_v1v2`와 같다)를 받는다.
    """

    def __init__(self, conf_threshold: float = FEATURE_CONF_THRESHOLD, max_det: int = FEATURE_MAX_DET) -> None:
        self.conf_threshold = float(conf_threshold)
        self.max_det = int(max_det)
        self._v1 = build_feature_extractor("v1", conf_threshold=self.conf_threshold, max_det=self.max_det)
        self._v2 = build_feature_extractor("v2", conf_threshold=self.conf_threshold, max_det=self.max_det)

    def reset(self) -> None:
        """추출기 내부 상태(직전 프레임 기억)를 지운다."""

        self._v1.reset()
        self._v2.reset()

    def update(self, frame: Any) -> tuple[list[float], list[float]]:
        """프레임 1개(FrameDetections)를 넣고 (v1 특징, v2 특징)을 반환한다. 호출 순서는 v1 → v2로 고정한다."""

        return self._v1.update(frame), self._v2.update(frame)

    def update_vector(self, frame: Any) -> np.ndarray:
        """프레임 1개를 넣고 float32 [FEATURE_DIM] = concat(v1, v2)를 반환한다."""

        f1, f2 = self.update(frame)
        out = np.asarray(f1 + f2, dtype=np.float32)
        if out.shape != (FEATURE_DIM,):
            raise ValueError(f"특징 차원 불일치: {out.shape} (기대 {FEATURE_DIM})")
        return out


def _check_history(history: int, stride: int) -> tuple[int, int]:
    """특징 이력 길이 H와 간격 s를 검사해 int로 돌려준다(둘 다 1 이상)."""

    h, st = int(history), int(stride)
    if h != history or st != stride:
        raise ValueError(f"history·stride는 정수여야 한다: history={history}, stride={stride}")
    if h < 1 or st < 1:
        raise ValueError(f"history·stride는 1 이상이어야 한다: history={h}, stride={st}")
    return h, st


def feature_history_offsets(history: int = DEFAULT_FEATURE_HISTORY, stride: int = DEFAULT_FEATURE_STRIDE) -> np.ndarray:
    """특징 이력 오프셋 int64 [H] = (0, s, 2s, …, (H−1)s). 인덱스 k는 프레임 t − k·s를 가리킨다."""

    h, st = _check_history(history, stride)
    return np.arange(h, dtype=np.int64) * st


def stack_feature_history(
    feats: np.ndarray,
    t: int | np.ndarray,
    history: int = DEFAULT_FEATURE_HISTORY,
    stride: int = DEFAULT_FEATURE_STRIDE,
) -> np.ndarray:
    """에피소드 특징 시퀀스 [T,D]에서 프레임 t의 관측 특징 [...,H,D]를 만든다.

    인덱스 k(0..H−1)는 feats[max(t − k·stride, 0)]이다(k=0 현재 프레임, 에피소드 시작에서 잘라 냄).
    기본값(H=2, s=2)이면 v3와 같은 (feats[t], feats[max(t−2,0)])이다(영상의 `history_index`와 같은 경계 규칙).
    학습·검증 시 풀 특징으로 관측을 재구성할 때 쓴다. t가 배열 [M]이면 [M,H,D]를 반환한다.
    """

    feats = np.asarray(feats)
    if feats.ndim != 2:
        raise ValueError(f"feats는 [T,D]여야 한다: {feats.shape}")
    offsets = feature_history_offsets(history, stride)
    t_arr = np.asarray(t, dtype=np.int64)
    idx = np.maximum(t_arr[..., None] - offsets, 0)  # [..., H]
    return feats[idx].astype(np.float32, copy=False)


class FeatureHistory:
    """lockstep 배치 B개 에피소드의 `"features"` 관측 [B,H,FEATURE_DIM]을 만드는 버퍼(폐루프용).

    에피소드마다 `OnlineFeatureTracker`를 두고, `update(frames)`를 프레임 0부터 한 번씩 부르면
    인덱스 k(0..H−1)에 프레임 max(t − k·stride, 0)의 특징을 쌓아 돌려준다(k=0 현재 프레임).
    슬롯 (H−1)·stride + 1개의 링 버퍼를 쓴다. 기본값(H=2, s=2)이면 v3와 같은 [B,2,D](슬롯 3개)다.
    학습 쪽 `PolicyData.observation(idx, history, stride)`와 같은 규칙이다(docs/36 6절).
    """

    def __init__(
        self,
        batch: int,
        history: int = DEFAULT_FEATURE_HISTORY,
        stride: int = DEFAULT_FEATURE_STRIDE,
        conf_threshold: float = FEATURE_CONF_THRESHOLD,
        max_det: int = FEATURE_MAX_DET,
    ) -> None:
        if batch <= 0:
            raise ValueError(f"batch는 양수여야 한다: {batch}")
        self.batch = int(batch)
        self.history, self.stride = _check_history(history, stride)
        self.trackers = [OnlineFeatureTracker(conf_threshold, max_det) for _ in range(self.batch)]
        self._offsets = feature_history_offsets(self.history, self.stride)
        self._slots = (self.history - 1) * self.stride + 1
        self._buf = np.zeros((self._slots, self.batch, FEATURE_DIM), dtype=np.float32)
        self.t = -1

    def update(self, frames: Sequence[Any]) -> np.ndarray:
        """에피소드별 현재 프레임(FrameDetections) B개를 넣고 float32 [B,H,FEATURE_DIM]을 반환한다(새 배열)."""

        if len(frames) != self.batch:
            raise ValueError(f"프레임 수 {len(frames)} != 배치 {self.batch}")
        self.t += 1
        slot = self.t % self._slots
        for i, frame in enumerate(frames):
            self._buf[slot, i] = self.trackers[i].update_vector(frame)
        idx = np.maximum(self.t - self._offsets, 0) % self._slots  # [H]
        # [H,B,D] → [B,H,D]. 팬시 인덱싱이 복사본을 만들므로 다음 update가 반환값을 덮어쓰지 않는다
        return np.ascontiguousarray(self._buf[idx].transpose(1, 0, 2))
