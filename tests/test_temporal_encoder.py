from __future__ import annotations

from pathlib import Path

import pytest

from vcp.config import TemporalConfig
from vcp.components.temporal import (
    GRUTemporalEncoderTorch,
    SYNTHETIC_FEATURE_KEYS,
    TemporalGRUNet,
    build_temporal_channel_groups,
    build_temporal_encoder,
)
from vcp.schemas import PackedFeature


def test_build_temporal_encoder_fallback_when_checkpoint_missing() -> None:
    cfg = TemporalConfig(
        model_name="gru_torch",
        window_size=4,
        num_contexts=3,
        hidden_dim=16,
        checkpoint_path="missing_checkpoint.pt",
        device="cpu",
    )
    encoder = build_temporal_encoder(cfg=cfg, labels=["a", "b", "c"], input_dim=8)
    assert encoder.__class__.__name__ == "GRUTemporalEncoderMock"


def test_gru_temporal_encoder_torch_runtime(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    model = TemporalGRUNet(input_dim=8, hidden_dim=12, num_contexts=3, dropout=0.1)
    ckpt_path = tmp_path / "temporal_best.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {
                "input_dim": 8,
                "hidden_dim": 12,
                "num_contexts": 3,
                "dropout": 0.1,
            },
        },
        ckpt_path,
    )

    cfg = TemporalConfig(
        model_name="gru_torch",
        window_size=4,
        num_contexts=3,
        hidden_dim=12,
        checkpoint_path=str(ckpt_path),
        device="cpu",
    )
    encoder = GRUTemporalEncoderTorch(cfg=cfg, labels=["idle", "move", "event"], input_dim=8)

    output = encoder.encode(
        PackedFeature(
            frame_id=1,
            sensor_timestamp=0.1,
            system_timestamp=0.1,
            spatial_vector=[0.1] * 8,
            metadata={},
        )
    )

    assert len(output.context_probs) == 3
    assert 0.0 <= output.boundary_signal <= 1.0


def test_build_temporal_channel_groups_for_synthetic_features() -> None:
    groups = build_temporal_channel_groups(16, feature_keys=SYNTHETIC_FEATURE_KEYS)

    assert len(groups) == 4
    assert groups[0] == [0, 1, 2, 3, 4]
    assert 11 in groups[2]
    assert 10 in groups[3]
