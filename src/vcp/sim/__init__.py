"""물리 기반 합성 주행 시나리오 생성기.

- `world`: 차량/보행자 운동(IDM 자차 제어, 스크립트 기반 주변 객체)
- `camera`: 핀홀 카메라 투영 + 검출기 노이즈 모델
- `labels`: GT 물리량(TTC, 자차 감속도 등) 기반 6맥락 라벨
- `dataset`: 에피소드 생성, 특징 계산, 저장/로드
"""

from .camera import CameraConfig, DetectorNoiseConfig, NOISE_LEVELS
from .labels import CONTEXT_LABELS
from .world import SCENARIO_TYPES, Episode, simulate_episode

__all__ = [
    "CameraConfig",
    "DetectorNoiseConfig",
    "NOISE_LEVELS",
    "CONTEXT_LABELS",
    "SCENARIO_TYPES",
    "Episode",
    "simulate_episode",
]
