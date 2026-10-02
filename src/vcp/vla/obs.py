from __future__ import annotations

"""VLA-lite 관측 규격(학습·폐루프 공통).

- 영상: `render_frame`으로 만든 64×64 RGB 프레임 2장(현재 t, 과거 t−HISTORY_OFFSET)을 채널로 쌓는다.
  에피소드 초반(t < HISTORY_OFFSET)에는 과거 프레임으로 프레임 0을 쓴다.
- proprio: 자차 속도 / speed_scale(domain). 목표 속도는 넣지 않는다(지시문으로만 전달).
- 행동 정규화: 정책 출력 가속도 / accel_scale(domain).
"""

import numpy as np

IMG_SIZE: tuple[int, int] = (64, 64)  # (H, W)
HISTORY_OFFSET: int = 2

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
