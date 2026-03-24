from __future__ import annotations

from pathlib import Path

import pytest

from vcp.components.camera import build_camera_stream
from vcp.config import CameraConfig, RuntimeConfig


def test_image_folder_camera_stream_reads_images(tmp_path: Path) -> None:
    pil_image = pytest.importorskip("PIL.Image")

    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(2):
        image = pil_image.new("RGB", (32, 32), color=(idx * 40, idx * 40, idx * 40))
        image.save(image_dir / f"frame_{idx:03d}.png")

    camera_cfg = CameraConfig(
        fps=10,
        width=32,
        height=32,
        channels=3,
        source_type="image_folder",
        image_dir=str(image_dir),
        image_pattern="*.png",
        recursive=False,
    )
    runtime_cfg = RuntimeConfig(profile="test", max_frames=2, seed=1)

    stream = build_camera_stream(camera_cfg, runtime_cfg)
    frame = stream.read()
    assert frame is not None
    assert frame.raw_path is not None
    assert len(frame.pixels) == 32

    second = stream.read()
    assert second is not None
    assert stream.read() is None
