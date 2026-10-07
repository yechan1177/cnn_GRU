"""검출 박스 -> 의미 특징 벡터 변환 모듈.

검출기(YOLO, 합성 검출 시뮬레이터 등)와 무관하게 동일한 특징 계산을 쓰기 위해
박스 목록만 입력으로 받는 순수 계산 모듈로 분리했다.
"""

from .base import Detection, FrameDetections, frame_from_dict
from .registry import (
    FEATURE_SPECS,
    FeatureSpec,
    build_feature_extractor,
    get_feature_spec,
    semantic_channel_groups,
)
from .semantic_v1 import SemanticFeatureV1
from .semantic_v2 import SemanticFeatureV2

__all__ = [
    "Detection",
    "FrameDetections",
    "frame_from_dict",
    "FEATURE_SPECS",
    "FeatureSpec",
    "build_feature_extractor",
    "get_feature_spec",
    "semantic_channel_groups",
    "SemanticFeatureV1",
    "SemanticFeatureV2",
]
