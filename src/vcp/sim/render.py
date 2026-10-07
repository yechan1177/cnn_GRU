from __future__ import annotations

"""GT 투영 박스 → 저해상도(기본 64×64) RGB 프레임 렌더러.

VLA-lite 정책의 시각 입력을 만든다. 실사 렌더링이 아니라 "장면 구조를 보존한 도식 영상"이다.
- 지평선 위: 하늘(주행) / 벽(로봇)
- 지평선 아래: 지면, 도로(통로) 사다리꼴, 차선(바닥 테이프). 지평선 y(피치)에 따라 함께 움직인다.
- 객체: 먼 것부터 그려 가까운 객체가 덮도록 하고, 클래스별 색에 깊이에 따른 미세한 명암을 준다.

결정적이며 난수를 쓰지 않는다. 배경(하늘·도로·차선)은 지평선 위치를 출력 픽셀의 1/4 단위로
양자화한 키로 캐시하므로 프레임당 비용은 배경 복사 + 박스 수만큼의 슬라이싱이다.
"""

import logging
from functools import lru_cache

import numpy as np

from ..vla.obs import IMG_SIZE
from .camera import CameraConfig
from .world import DRIVING, ROBOT, DomainProfile

logger = logging.getLogger(__name__)

# 도메인별 배경 색(RGB)
_PALETTE: dict[str, dict[str, tuple[int, int, int]]] = {
    "driving": {
        "sky": (140, 180, 225),
        "ground": (95, 120, 75),
        "road": (88, 88, 94),
        "line": (235, 235, 235),
    },
    "robot": {
        "sky": (178, 176, 168),  # 창고 벽
        "ground": (122, 120, 114),  # 콘크리트 바닥
        "road": (158, 152, 134),  # 통로 바닥
        "line": (226, 188, 40),  # 노란 안전 테이프
    },
}
# 클래스별 객체 색(person=0, vehicle=1, bike=2)
_CLASS_COLORS = np.asarray([(220, 60, 60), (50, 85, 205), (235, 160, 30)], dtype=np.float32)
_UNKNOWN_COLOR = np.asarray((255, 0, 255), dtype=np.float32)
# 깊이에 따른 명암: 최대 거리에서 색을 이만큼 어둡게 한다
_DEPTH_SHADE = 0.3
# 차선(테이프) 실제 폭(m)과 도로 갓길 여유(m)
_LINE_W_M = 0.15
_SHOULDER_M = 0.5
# 지평선 양자화 단위(출력 픽셀): 1/4 px
_HORIZON_Q = 4.0


def _profile(domain: str) -> DomainProfile:
    key = str(domain).strip().lower()
    if key == "driving":
        return DRIVING
    if key == "robot":
        return ROBOT
    raise KeyError(f"지원하지 않는 도메인: {domain}")


@lru_cache(maxsize=4)
def default_camera(domain: str) -> CameraConfig:
    """도메인 기본 카메라(SimEnv 기본값과 같다)."""

    prof = _profile(domain)
    return CameraConfig(cam_height_m=prof.cam_height_m, max_range_m=prof.max_range_m)


@lru_cache(maxsize=2048)
def _background(
    domain: str, h: int, w: int, cam_w: int, cam_h_px: int, cam_height_m: float, hq: int
) -> np.ndarray:
    """양자화된 지평선(hq / _HORIZON_Q, 출력 픽셀)에 대한 배경 이미지(읽기 전용)."""

    prof = _profile(domain)
    pal = _PALETTE[prof.name]
    sx, sy = w / cam_w, h / cam_h_px
    hy_out = hq / _HORIZON_Q
    img = np.empty((h, w, 3), dtype=np.uint8)
    rows = np.arange(h, dtype=np.float32) + 0.5
    cols = np.arange(w, dtype=np.float32) + 0.5 - 0.5 * w
    below = rows > hy_out
    img[~below] = pal["sky"]
    img[below] = pal["ground"]
    # 원본 픽셀 기준 지평선 아래 거리 → 횡방향 1 m가 출력 픽셀 몇 개인지(행별)
    dy_orig = np.maximum(0.0, (rows - hy_out) / sy)
    px_per_m = (dy_orig / cam_height_m * sx)[:, None]  # [h,1]
    lane_w = prof.lane_w
    road_half = 1.5 * lane_w + _SHOULDER_M
    road = (np.abs(cols)[None, :] < road_half * px_per_m) & below[:, None]
    img[road] = pal["road"]
    line = np.zeros((h, w), dtype=bool)
    # 행당 수평 이동량(기울기)만큼 선 폭을 넓혀 비스듬한 선이 끊기지 않게 한다
    dx_per_row = sx / (sy * cam_height_m)
    for lat in (-1.5 * lane_w, -0.5 * lane_w, 0.5 * lane_w, 1.5 * lane_w):
        line_half = np.maximum(0.5 * max(1.0, abs(lat) * dx_per_row), 0.5 * _LINE_W_M * px_per_m)
        line |= np.abs(cols[None, :] - lat * px_per_m) < line_half
    line &= below[:, None] & (px_per_m > 0.05)
    img[line] = pal["line"]
    img.setflags(write=False)
    return img


def render_frame(
    boxes: np.ndarray,
    horizon_y: float,
    domain: str = "driving",
    size: tuple[int, int] = IMG_SIZE,
    cam: CameraConfig | None = None,
) -> np.ndarray:
    """GT 박스 목록으로 uint8 [H,W,3] 프레임을 그린다.

    Args:
        boxes: float32 [M,6] = (x1, y1, x2, y2, cls_id, depth_m). 원본 카메라 픽셀 좌표(기본 640×480).
        horizon_y: 지평선 y(원본 픽셀).
        domain: "driving" | "robot"(배경 색·차선 간격이 다르다).
        size: 출력 (H, W).
        cam: 원본 카메라 설정. None이면 도메인 기본값.
    """

    cam = cam or default_camera(domain)
    h, w = int(size[0]), int(size[1])
    sx, sy = w / cam.width, h / cam.height
    hq = int(round(float(horizon_y) * sy * _HORIZON_Q))
    img = _background(str(domain).strip().lower(), h, w, cam.width, cam.height, float(cam.cam_height_m), hq).copy()
    if boxes is None or len(boxes) == 0:
        return img
    b = np.asarray(boxes, dtype=np.float32).reshape(-1, 6)
    order = np.argsort(-b[:, 5], kind="stable")  # 먼 것부터
    b = b[order]
    x1 = np.clip(np.floor(b[:, 0] * sx), 0, w - 1).astype(np.int32)
    y1 = np.clip(np.floor(b[:, 1] * sy), 0, h - 1).astype(np.int32)
    x2 = np.clip(np.maximum(np.ceil(b[:, 2] * sx), x1 + 1), 1, w).astype(np.int32)
    y2 = np.clip(np.maximum(np.ceil(b[:, 3] * sy), y1 + 1), 1, h).astype(np.int32)
    cls = b[:, 4].astype(np.int32)
    valid = (cls >= 0) & (cls < len(_CLASS_COLORS))
    base = np.where(valid[:, None], _CLASS_COLORS[np.clip(cls, 0, len(_CLASS_COLORS) - 1)], _UNKNOWN_COLOR)
    shade = 1.0 - _DEPTH_SHADE * np.clip(b[:, 5] / max(1e-6, cam.max_range_m), 0.0, 1.0)
    colors = np.clip(base * shade[:, None] + 0.5, 0, 255).astype(np.uint8)
    for i, (a, bb, c, d) in enumerate(zip(x1.tolist(), y1.tolist(), x2.tolist(), y2.tolist())):
        img[bb:d, a:c] = colors[i]
    return img
