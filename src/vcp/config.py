from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RuntimeConfig:
    """파이프라인 공통 실행 설정."""

    profile: str = "baseline"
    max_frames: int = 120
    seed: int = 7


@dataclass(slots=True)
class CameraConfig:
    """입력 소스 설정.

    source_type:
    - mock: 내장 mock 스트림 사용
    - image_folder: 이미지 폴더를 순차 프레임처럼 사용
    """

    fps: int = 15
    width: int = 224
    height: int = 224
    channels: int = 3
    source_type: str = "mock"
    image_dir: str | None = None
    image_pattern: str = "*.png"
    recursive: bool = True


@dataclass(slots=True)
class SpatialConfig:
    """Spatial encoder 설정."""

    model_name: str = "mock_mobilenetv3_small"
    feature_dim: int = 64
    weights_path: str | None = None
    device: str = "cpu"
    conf_threshold: float = 0.25
    max_det: int = 20
    input_size: int = 640
    fallback_to_mock: bool = True


@dataclass(slots=True)
class TemporalConfig:
    """Temporal encoder 설정."""

    model_name: str = "gru_mock"
    window_size: int = 8
    num_contexts: int = 4
    hidden_dim: int = 32
    checkpoint_path: str | None = None
    device: str = "cpu"
    dropout: float = 0.1


@dataclass(slots=True)
class ScoringConfig:
    """이벤트 점수 가중치와 임계값 설정."""

    context_weight: float = 0.35
    boundary_weight: float = 0.30
    uncertainty_weight: float = 0.20
    novelty_weight: float = 0.15
    high_threshold: float = 0.72
    mid_threshold: float = 0.48


@dataclass(slots=True)
class CurationConfig:
    """자동 정제 설정."""

    ring_buffer_size: int = 128
    pre_event_frames: int = 8
    post_event_frames: int = 8


@dataclass(slots=True)
class StorageConfig:
    """출력 저장 설정."""

    base_dir: str = "outputs"
    run_name: str = "baseline_default"
    save_raw_frames: bool = False


@dataclass(slots=True)
class PipelineConfig:
    """파이프라인 전체 설정."""

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    curation: CurationConfig = field(default_factory=CurationConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    labels: list[str] = field(
        default_factory=lambda: ["idle", "approach", "manipulate", "handover"]
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _update_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
    for key, value in updates.items():
        if hasattr(instance, key):
            setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> PipelineConfig:
    """YAML 설정 파일을 로드해 PipelineConfig로 변환한다."""
    cfg = PipelineConfig()
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        loaded: dict[str, Any] = yaml.safe_load(file) or {}

    if "runtime" in loaded:
        cfg.runtime = _update_dataclass(cfg.runtime, loaded["runtime"])
    if "camera" in loaded:
        cfg.camera = _update_dataclass(cfg.camera, loaded["camera"])
    if "spatial" in loaded:
        cfg.spatial = _update_dataclass(cfg.spatial, loaded["spatial"])
    if "temporal" in loaded:
        cfg.temporal = _update_dataclass(cfg.temporal, loaded["temporal"])
    if "scoring" in loaded:
        cfg.scoring = _update_dataclass(cfg.scoring, loaded["scoring"])
    if "curation" in loaded:
        cfg.curation = _update_dataclass(cfg.curation, loaded["curation"])
    if "storage" in loaded:
        cfg.storage = _update_dataclass(cfg.storage, loaded["storage"])
    if "labels" in loaded and isinstance(loaded["labels"], list):
        cfg.labels = [str(label) for label in loaded["labels"]]

    return cfg
