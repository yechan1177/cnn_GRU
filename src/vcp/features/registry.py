from __future__ import annotations

"""특징 버전 레지스트리와 의미 기반 채널 그룹 정의."""

from dataclasses import dataclass
from typing import Any

from .semantic_v1 import V1_KEYS, SemanticFeatureV1
from .semantic_v2 import V2_KEYS, SemanticFeatureV2


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """특징 버전별 key 목록과 의미 그룹(key 인덱스)."""

    name: str
    keys: tuple[str, ...]
    semantic_groups: tuple[tuple[str, tuple[int, ...]], ...]

    @property
    def dim(self) -> int:
        return len(self.keys)

    def index(self, key: str) -> int:
        return self.keys.index(key)


# v1: 실제 값 배치 기준(5=person 비율, 6=vehicle 비율, 8=person conf, 9=vehicle conf)
_V1_GROUPS = (
    ("global", (0, 1, 2, 3, 4)),
    ("vehicle", (6, 9, 13, 14)),
    ("person", (5, 8, 11, 12)),
    ("bike", (7, 10, 11, 15)),
)

_V2_GROUPS = (
    ("global", (0, 1, 2, 3, 4)),
    ("vehicle", (5, 6, 7, 8, 9, 10)),
    ("vru", (11, 12, 13, 14, 15)),
)

# v1+v2 결합(32차원): v1은 0-15, v2는 16-31
_V1V2_GROUPS = (
    ("global", (0, 1, 2, 3, 4, 16, 17, 18, 19, 20)),
    ("vehicle", (6, 9, 13, 14, 21, 22, 23, 24, 25, 26)),
    ("vru", (5, 7, 8, 10, 11, 12, 15, 27, 28, 29, 30, 31)),
)

FEATURE_SPECS: dict[str, FeatureSpec] = {
    "v1": FeatureSpec("v1", tuple(V1_KEYS), _V1_GROUPS),
    "v2": FeatureSpec("v2", tuple(V2_KEYS), _V2_GROUPS),
    "v1v2": FeatureSpec(
        "v1v2",
        tuple([f"v1.{k}" for k in V1_KEYS] + [f"v2.{k}" for k in V2_KEYS]),
        _V1V2_GROUPS,
    ),
}


class CombinedFeatureV1V2:
    """v1과 v2 특징을 이어 붙인 32차원 추출기(상호 보완 ablation용)."""

    def __init__(self, **kwargs: Any) -> None:
        self._v1 = build_feature_extractor("v1", **kwargs)
        self._v2 = build_feature_extractor("v2", **kwargs)
        self.keys = list(FEATURE_SPECS["v1v2"].keys)

    def reset(self) -> None:
        self._v1.reset()
        self._v2.reset()

    def update(self, frame: Any) -> list[float]:
        return self._v1.update(frame) + self._v2.update(frame)


def get_feature_spec(version: str) -> FeatureSpec:
    key = str(version).strip().lower()
    if key not in FEATURE_SPECS:
        raise KeyError(f"알 수 없는 특징 버전: {version} (지원: {sorted(FEATURE_SPECS)})")
    return FEATURE_SPECS[key]


def semantic_channel_groups(version: str) -> list[list[int]]:
    """멀티채널 CNN-GRU용 의미 그룹(인덱스 리스트)."""

    return [list(indices) for _, indices in get_feature_spec(version).semantic_groups]


def build_feature_extractor(version: str, **kwargs: Any) -> Any:
    key = get_feature_spec(version).name
    if key == "v1v2":
        return CombinedFeatureV1V2(**kwargs)
    if key == "v1":
        allowed = {k: v for k, v in kwargs.items() if k in {"conf_threshold", "max_det", "class_count"}}
        return SemanticFeatureV1(**allowed)
    allowed = {
        k: v
        for k, v in kwargs.items()
        if k in {"conf_threshold", "max_det", "tau_s", "iou_match", "lane_half_width"}
    }
    return SemanticFeatureV2(**allowed)
