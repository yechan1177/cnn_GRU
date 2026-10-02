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


def test_video_camera_stream_reads_mp4(tmp_path) -> None:
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    from vcp.components.camera import VideoCameraStream
    from vcp.config import CameraConfig, RuntimeConfig

    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    for i in range(6):
        writer.write(np.full((48, 64, 3), i * 30, dtype=np.uint8))
    writer.release()
    stream = VideoCameraStream(CameraConfig(source_type="video", video_source=str(path)), RuntimeConfig(max_frames=4))
    frames = []
    while (pkt := stream.read()) is not None:
        frames.append(pkt)
    stream.close()
    assert len(frames) == 4
    assert frames[1].image is not None and abs(frames[1].sensor_timestamp - 0.1) < 1e-6
