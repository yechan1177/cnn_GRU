from __future__ import annotations

import logging
import math
from collections import deque
from pathlib import Path
from typing import Any

from ..config import TemporalConfig
from ..interfaces import TemporalEncoder
from ..schemas import PackedFeature, TemporalOutput

logger = logging.getLogger(__name__)

SYNTHETIC_FEATURE_KEYS: list[str] = [
    "det_norm",
    "mean_conf",
    "max_conf",
    "mean_area",
    "area_var",
    "vehicle_ratio",
    "person_ratio",
    "bike_ratio",
    "vehicle_conf",
    "person_conf",
    "bike_conf",
    "roi_risk",
    "motion_delta",
    "center_closeness",
    "looming_score",
    "occlusion_score",
]

try:  # pragma: no cover - torch 설치 여부는 실행 환경 의존
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None


def _softmax(values: list[float]) -> list[float]:
    if not values:
        return []
    max_val = max(values)
    exp_values = [math.exp(value - max_val) for value in values]
    denominator = sum(exp_values) or 1.0
    return [value / denominator for value in exp_values]


def _balanced_channel_groups(input_dim: int, group_count: int = 4) -> list[list[int]]:
    safe_dim = max(1, int(input_dim))
    safe_groups = max(1, min(int(group_count), safe_dim))
    base = safe_dim // safe_groups
    remainder = safe_dim % safe_groups

    groups: list[list[int]] = []
    start = 0
    for group_idx in range(safe_groups):
        width = base + (1 if group_idx < remainder else 0)
        end = min(safe_dim, start + max(1, width))
        groups.append(list(range(start, end)))
        start = end
    return [group for group in groups if group]


def build_temporal_channel_groups(
    input_dim: int,
    feature_keys: list[str] | None = None,
    grouping: str = "auto",
    feature_version: str | None = None,
) -> list[list[int]]:
    """입력 벡터를 멀티채널 CNN-GRU용 채널 그룹으로 분리한다.

    grouping
    - ``semantic``: 특징 버전(v1/v2)의 의미 그룹(global/vehicle/person/bike 등)
    - ``balanced``: 인덱스 균등 분할(2026-03 배포 체크포인트가 실제로 사용한 방식)
    - ``single``: 전체를 하나의 그룹(단일채널 CNN-GRU)
    - ``auto``: 기존 동작(가능하면 key 이름 기반 의미 그룹, 아니면 균등 분할)
    """

    safe_dim = max(1, int(input_dim))
    mode = str(grouping or "auto").strip().lower()
    if mode == "single":
        return [list(range(safe_dim))]
    if mode == "semantic":
        from ..features.registry import get_feature_spec

        spec = get_feature_spec(feature_version or "v1")
        if spec.dim != safe_dim:
            raise ValueError(f"의미 그룹 차원 불일치: spec={spec.dim}, input={safe_dim}")
        return [list(indices) for _, indices in spec.semantic_groups]
    if mode == "balanced":
        if safe_dim <= 8:
            return _balanced_channel_groups(safe_dim, group_count=4)
        global_group = list(range(0, min(5, safe_dim)))
        remaining = list(range(len(global_group), safe_dim))
        split_groups = _balanced_channel_groups(len(remaining), group_count=3) if remaining else []
        return [global_group] + [[remaining[idx] for idx in group] for group in split_groups]
    if safe_dim <= 8:
        return _balanced_channel_groups(safe_dim, group_count=4)

    if feature_keys and len(feature_keys) >= len(SYNTHETIC_FEATURE_KEYS):
        key_to_index = {name: idx for idx, name in enumerate(feature_keys)}
        if all(name in key_to_index for name in SYNTHETIC_FEATURE_KEYS):
            return [
                [key_to_index[name] for name in ["det_norm", "mean_conf", "max_conf", "mean_area", "area_var"]],
                [key_to_index[name] for name in ["vehicle_ratio", "vehicle_conf", "center_closeness", "looming_score"]],
                [key_to_index[name] for name in ["person_ratio", "person_conf", "roi_risk", "motion_delta"]],
                [key_to_index[name] for name in ["bike_ratio", "bike_conf", "roi_risk", "occlusion_score"]],
            ]

    global_group = list(range(0, min(5, safe_dim)))
    remaining = list(range(len(global_group), safe_dim))
    if not remaining:
        return [global_group]

    split_groups = _balanced_channel_groups(len(remaining), group_count=3)
    groups = [global_group]
    for group in split_groups:
        groups.append([remaining[idx] for idx in group])
    return [group for group in groups if group]


class GRUTemporalEncoderMock(TemporalEncoder):
    """GRU 형태를 단순화한 mock temporal encoder."""

    def __init__(self, cfg: TemporalConfig, labels: list[str]) -> None:
        self._window_size = max(2, int(cfg.window_size))
        self._num_contexts = max(2, int(cfg.num_contexts))
        self._buffer: deque[list[float]] = deque(maxlen=self._window_size)
        self._prev_probs: list[float] | None = None
        self._prev_pooled: list[float] | None = None

        if len(labels) < self._num_contexts:
            self._labels = [f"context_{idx}" for idx in range(self._num_contexts)]
        else:
            self._labels = labels[: self._num_contexts]

    def encode(self, packed: PackedFeature) -> TemporalOutput:
        self._buffer.append(packed.spatial_vector)
        pooled = self._pooled_vector()
        motion_delta = self._compute_motion_delta()

        logits: list[float] = []
        for context_idx in range(self._num_contexts):
            weight = (context_idx + 1) * 0.018
            score = 0.0
            for feature_idx, feature_val in enumerate(pooled[:32]):
                score += feature_val * ((feature_idx % 5) + 1) * weight
            score += math.sin((packed.frame_id * 0.27) + (context_idx * 0.9)) * 0.22
            score += motion_delta * (context_idx + 1) * 0.35
            score += context_idx * 0.02
            logits.append(score)

        probs = _softmax(logits)
        context_probs = {
            label: round(prob, 6) for label, prob in zip(self._labels, probs, strict=True)
        }
        boundary = self._compute_boundary_signal(probs, pooled)
        self._prev_probs = probs
        self._prev_pooled = pooled

        return TemporalOutput(
            frame_id=packed.frame_id,
            context_probs=context_probs,
            boundary_signal=round(boundary, 6),
        )

    def _pooled_vector(self) -> list[float]:
        if not self._buffer:
            return []
        dim = len(self._buffer[0])
        pooled = [0.0] * dim
        for vector in self._buffer:
            for idx, value in enumerate(vector):
                pooled[idx] += value
        count = float(len(self._buffer))
        return [value / count for value in pooled]

    def _compute_motion_delta(self) -> float:
        if len(self._buffer) < 2:
            return 0.0
        current = self._buffer[-1]
        previous = self._buffer[-2]
        dim = min(len(current), len(previous))
        if dim == 0:
            return 0.0
        delta = sum(abs(current[idx] - previous[idx]) for idx in range(dim)) / dim
        return min(1.0, delta * 12.0)

    def _compute_boundary_signal(self, probs: list[float], pooled: list[float]) -> float:
        if self._prev_probs is None or self._prev_pooled is None:
            return 0.0

        prob_dim = min(len(probs), len(self._prev_probs))
        prob_shift = (
            sum(abs(probs[idx] - self._prev_probs[idx]) for idx in range(prob_dim)) / 2.0
            if prob_dim > 0
            else 0.0
        )

        feature_dim = min(len(pooled), len(self._prev_pooled))
        feature_shift = (
            sum(abs(pooled[idx] - self._prev_pooled[idx]) for idx in range(feature_dim))
            / feature_dim
            if feature_dim > 0
            else 0.0
        )

        prob_component = min(1.0, prob_shift * 3.0)
        feature_component = min(1.0, feature_shift * 20.0)
        return min(1.0, (0.55 * prob_component) + (0.45 * feature_component))


if nn is not None:  # pragma: no branch

    class BaselineTemporalGRUNet(nn.Module):
        """단일 입력 GRU baseline."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_contexts: int,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.input_proj = nn.Linear(int(input_dim), int(hidden_dim))
            self.gru = nn.GRU(
                input_size=int(hidden_dim),
                hidden_size=int(hidden_dim),
                num_layers=1,
                batch_first=True,
            )
            self.dropout = nn.Dropout(p=max(0.0, min(0.5, float(dropout))))
            self.context_head = nn.Linear(int(hidden_dim), int(num_contexts))
            self.boundary_head = nn.Linear(int(hidden_dim), 1)
            self.uncertainty_head = nn.Linear(int(hidden_dim), 1)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            hidden = torch.relu(self.input_proj(x))
            gru_out, _ = self.gru(hidden)
            pooled = self.dropout(gru_out[:, -1, :])
            return {
                "context_logits": self.context_head(pooled),
                "boundary_logit": self.boundary_head(pooled).squeeze(-1),
                "uncertainty_logit": self.uncertainty_head(pooled).squeeze(-1),
            }

    class TemporalGRUNet(nn.Module):
        """멀티채널 CNN-GRU 기반 temporal 본체.

        - context_head: 클래스 분류
        - boundary_head: 이벤트 경계 점수
        - uncertainty_head: 불확실성 점수
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_contexts: int,
            dropout: float = 0.1,
            cnn_channels: int = 16,
            channel_groups: list[list[int]] | None = None,
        ) -> None:
            super().__init__()
            self.input_dim = int(input_dim)
            self.hidden_dim = int(hidden_dim)
            self.num_contexts = int(num_contexts)
            self.cnn_channels = max(4, int(cnn_channels))
            self.channel_groups = self._normalize_channel_groups(
                channel_groups or build_temporal_channel_groups(self.input_dim)
            )

            self.channel_blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(len(group), self.cnn_channels, kernel_size=3, padding=1),
                        nn.BatchNorm1d(self.cnn_channels),
                        nn.ReLU(),
                        nn.Conv1d(self.cnn_channels, self.cnn_channels, kernel_size=3, padding=1),
                        nn.ReLU(),
                    )
                    for group in self.channel_groups
                ]
            )
            self.fused_dim = self.cnn_channels * len(self.channel_groups)
            self.gru = nn.GRU(
                input_size=self.fused_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            self.dropout = nn.Dropout(p=max(0.0, min(0.5, float(dropout))))
            self.context_head = nn.Linear(self.hidden_dim, self.num_contexts)
            self.boundary_head = nn.Linear(self.hidden_dim, 1)
            self.uncertainty_head = nn.Linear(self.hidden_dim, 1)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            # x: [B, T, D]
            channel_features: list[torch.Tensor] = []
            for group, block in zip(self.channel_groups, self.channel_blocks, strict=True):
                sliced = x[:, :, group].transpose(1, 2)  # [B, C, T]
                encoded = block(sliced).transpose(1, 2)  # [B, T, cnn_channels]
                channel_features.append(encoded)

            fused = torch.cat(channel_features, dim=-1)
            gru_out, _ = self.gru(fused)
            pooled = self.dropout(gru_out.mean(dim=1))
            context_logits = self.context_head(pooled)
            boundary_logit = self.boundary_head(pooled).squeeze(-1)
            uncertainty_logit = self.uncertainty_head(pooled).squeeze(-1)
            return {
                "context_logits": context_logits,
                "boundary_logit": boundary_logit,
                "uncertainty_logit": uncertainty_logit,
            }

        def _normalize_channel_groups(self, groups: list[list[int]]) -> list[list[int]]:
            normalized: list[list[int]] = []
            for group in groups:
                valid = sorted(
                    {
                        int(index)
                        for index in group
                        if 0 <= int(index) < self.input_dim
                    }
                )
                if valid:
                    normalized.append(valid)
            if not normalized:
                normalized = _balanced_channel_groups(self.input_dim, group_count=4)
            return normalized

    class LiteratureCNNGRUNet(nn.Module):
        """문헌형 단일채널 CNN-GRU baseline (2026-03 비교군과 같은 파라미터 이름)."""

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_contexts: int,
            dropout: float = 0.1,
            cnn_channels: int = 24,
        ) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(int(input_dim), int(cnn_channels), kernel_size=3, padding=1),
                nn.BatchNorm1d(int(cnn_channels)),
                nn.ReLU(),
                nn.Conv1d(int(cnn_channels), int(cnn_channels), kernel_size=3, padding=1),
                nn.ReLU(),
            )
            self.gru = nn.GRU(
                input_size=int(cnn_channels),
                hidden_size=int(hidden_dim),
                num_layers=1,
                batch_first=True,
            )
            self.dropout = nn.Dropout(max(0.0, min(0.5, float(dropout))))
            self.context_head = nn.Linear(int(hidden_dim), int(num_contexts))
            self.boundary_head = nn.Linear(int(hidden_dim), 1)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            encoded = self.conv(x.transpose(1, 2)).transpose(1, 2)
            gru_out, _ = self.gru(encoded)
            pooled = self.dropout(gru_out.mean(dim=1))
            return {
                "context_logits": self.context_head(pooled),
                "boundary_logit": self.boundary_head(pooled).squeeze(-1),
            }

    class LastFrameMLP(nn.Module):
        """시간 정보를 쓰지 않는 ablation baseline(마지막 프레임 특징만 사용)."""

        def __init__(self, input_dim: int, hidden_dim: int, num_contexts: int, dropout: float = 0.1) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(int(input_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Dropout(max(0.0, min(0.5, float(dropout)))),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(),
            )
            self.context_head = nn.Linear(int(hidden_dim), int(num_contexts))
            self.boundary_head = nn.Linear(int(hidden_dim), 1)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            pooled = self.body(x[:, -1, :])
            return {
                "context_logits": self.context_head(pooled),
                "boundary_logit": self.boundary_head(pooled).squeeze(-1),
            }

else:

    class TemporalGRUNet:  # pragma: no cover - torch 미설치 환경 안내용
        """torch 미설치 환경에서의 placeholder."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ModuleNotFoundError("TemporalGRUNet 사용 시 torch 설치가 필요합니다.")

    class BaselineTemporalGRUNet:  # pragma: no cover
        """torch 미설치 환경에서의 placeholder."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ModuleNotFoundError("BaselineTemporalGRUNet 사용 시 torch 설치가 필요합니다.")

    LiteratureCNNGRUNet = BaselineTemporalGRUNet  # pragma: no cover
    LastFrameMLP = BaselineTemporalGRUNet  # pragma: no cover


def build_temporal_model(
    *,
    architecture: str,
    input_dim: int,
    hidden_dim: int,
    num_contexts: int,
    dropout: float,
    cnn_channels: int = 16,
    channel_groups: list[list[int]] | None = None,
) -> Any:
    arch = architecture.strip().lower()
    if arch in {"gru", "gru_baseline", "baseline_gru"}:
        return BaselineTemporalGRUNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_contexts=num_contexts,
            dropout=dropout,
        )
    if arch in {"single_channel_cnn_gru", "literature_cnn_gru"}:
        return LiteratureCNNGRUNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_contexts=num_contexts,
            dropout=dropout,
            cnn_channels=cnn_channels,
        )
    if arch in {"mlp_last", "last_frame_mlp"}:
        return LastFrameMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_contexts=num_contexts,
            dropout=dropout,
        )
    if arch in {"single_cnn_gru", "cnn_gru_single"}:
        return TemporalGRUNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_contexts=num_contexts,
            dropout=dropout,
            cnn_channels=cnn_channels,
            channel_groups=[list(range(max(1, int(input_dim))))],
        )
    return TemporalGRUNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_contexts=num_contexts,
        dropout=dropout,
        cnn_channels=cnn_channels,
        channel_groups=channel_groups,
    )


class GRUTemporalEncoderTorch(TemporalEncoder):
    """학습된 TemporalGRUNet 체크포인트를 사용하는 추론용 temporal encoder."""

    def __init__(self, cfg: TemporalConfig, labels: list[str], input_dim: int) -> None:
        if torch is None:
            raise ModuleNotFoundError("GRUTemporalEncoderTorch 사용 시 torch 설치가 필요합니다.")
        if not cfg.checkpoint_path:
            raise ValueError("gru_torch 모드에는 temporal.checkpoint_path가 필요합니다.")

        self._window_size = max(2, int(cfg.window_size))
        from ..utils.device import resolve_device

        self._device = torch.device(resolve_device(str(cfg.device)))
        checkpoint_path = Path(cfg.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Temporal 체크포인트를 찾을 수 없습니다: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        model_cfg = checkpoint.get("model_config", {})
        state_dict = checkpoint.get("state_dict", {})

        inferred_architecture = str(model_cfg.get("architecture", "")).strip().lower()
        if not inferred_architecture:
            if "input_proj.weight" in state_dict:
                inferred_architecture = "gru"
            elif "conv.0.weight" in state_dict:
                inferred_architecture = "single_channel_cnn_gru"
            else:
                inferred_architecture = "multichannel_cnn_gru"

        input_proj_weight = state_dict.get("input_proj.weight")
        context_head_weight = state_dict.get("context_head.weight")

        inferred_input_dim = input_dim
        if input_proj_weight is not None:
            inferred_input_dim = int(input_proj_weight.shape[1])

        inferred_hidden_dim = cfg.hidden_dim
        if input_proj_weight is not None:
            inferred_hidden_dim = int(input_proj_weight.shape[0])
        elif context_head_weight is not None:
            inferred_hidden_dim = int(context_head_weight.shape[1])

        inferred_num_contexts = cfg.num_contexts
        if context_head_weight is not None:
            inferred_num_contexts = int(context_head_weight.shape[0])

        ckpt_input_dim = int(model_cfg.get("input_dim", inferred_input_dim))
        ckpt_hidden_dim = int(model_cfg.get("hidden_dim", inferred_hidden_dim))
        ckpt_num_contexts = int(model_cfg.get("num_contexts", inferred_num_contexts))
        ckpt_dropout = float(model_cfg.get("dropout", cfg.dropout))
        conv_weight = state_dict.get("conv.0.weight")
        default_cnn = int(conv_weight.shape[0]) if conv_weight is not None else 16
        ckpt_cnn_channels = int(model_cfg.get("cnn_channels", default_cnn))
        raw_channel_groups = model_cfg.get("channel_groups", build_temporal_channel_groups(ckpt_input_dim))
        architecture = inferred_architecture

        self._input_dim = max(1, ckpt_input_dim)
        self._num_contexts = max(2, ckpt_num_contexts)
        self._channel_groups = [
            [int(item) for item in group]
            for group in raw_channel_groups
            if isinstance(group, list)
        ]
        if len(labels) < self._num_contexts:
            self._labels = [f"context_{idx}" for idx in range(self._num_contexts)]
        else:
            self._labels = labels[: self._num_contexts]

        self._model = build_temporal_model(
            architecture=architecture,
            input_dim=self._input_dim,
            hidden_dim=max(4, ckpt_hidden_dim),
            num_contexts=self._num_contexts,
            dropout=ckpt_dropout,
            cnn_channels=ckpt_cnn_channels,
            channel_groups=self._channel_groups,
        )
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()

        self._buffer: deque[list[float]] = deque(maxlen=self._window_size)

        logger.info(
            "GRUTemporalEncoderTorch 초기화: ckpt=%s, device=%s, input_dim=%s, contexts=%s",
            checkpoint_path,
            self._device,
            self._input_dim,
            self._num_contexts,
        )

    def encode(self, packed: PackedFeature) -> TemporalOutput:
        vector = self._normalize_vector(packed.spatial_vector)
        self._buffer.append(vector)

        seq = list(self._buffer)
        if len(seq) < self._window_size:
            padding = [[0.0] * self._input_dim for _ in range(self._window_size - len(seq))]
            seq = padding + seq

        assert torch is not None  # type narrowing
        x = torch.tensor(seq, dtype=torch.float32, device=self._device).unsqueeze(0)
        with torch.no_grad():
            output = self._model(x)

        context_logits = output["context_logits"][0]
        context_probs_tensor = torch.softmax(context_logits, dim=-1).detach().cpu()
        probs = [float(item) for item in context_probs_tensor.tolist()]

        boundary_logit = float(output["boundary_logit"][0].detach().cpu().item())
        boundary_signal = 1.0 / (1.0 + math.exp(-boundary_logit))

        context_probs = {
            label: round(prob, 6) for label, prob in zip(self._labels, probs, strict=True)
        }
        return TemporalOutput(
            frame_id=packed.frame_id,
            context_probs=context_probs,
            boundary_signal=round(boundary_signal, 6),
        )

    def _normalize_vector(self, vector: list[float]) -> list[float]:
        if len(vector) == self._input_dim:
            return [float(value) for value in vector]
        if len(vector) > self._input_dim:
            return [float(value) for value in vector[: self._input_dim]]
        padded = [float(value) for value in vector]
        padded.extend([0.0] * (self._input_dim - len(padded)))
        return padded


def build_temporal_encoder(
    cfg: TemporalConfig,
    labels: list[str],
    input_dim: int,
) -> TemporalEncoder:
    """설정에 맞는 temporal encoder 구현체를 생성한다."""

    model_name = cfg.model_name.strip().lower()
    if model_name in {"gru_torch", "gru_runtime", "gru_infer"}:
        try:
            return GRUTemporalEncoderTorch(cfg=cfg, labels=labels, input_dim=input_dim)
        except Exception as exc:
            if not getattr(cfg, "fallback_to_mock", True):
                raise
            logger.error(
                "GRUTemporalEncoderTorch 초기화 실패 -> mock encoder로 대체합니다. "
                "mock 출력은 학습된 모델이 아니므로 실험/배포에 사용하지 마십시오: %s",
                exc,
            )

    return GRUTemporalEncoderMock(cfg=cfg, labels=labels)
