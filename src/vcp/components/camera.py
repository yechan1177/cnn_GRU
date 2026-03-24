from __future__ import annotations

import logging
import math
import random
import time
from pathlib import Path

from ..config import CameraConfig, RuntimeConfig
from ..interfaces import CameraStream
from ..schemas import FramePacket

logger = logging.getLogger(__name__)


class MockCameraStream(CameraStream):
    """실시간 카메라를 대체하는 mock 스트림."""

    def __init__(self, camera_cfg: CameraConfig, runtime_cfg: RuntimeConfig) -> None:
        self._fps = max(1, int(camera_cfg.fps))
        self._max_frames = max(1, int(runtime_cfg.max_frames))
        self._seed = int(runtime_cfg.seed)
        self._frame_id = 0
        self._sensor_start_ts = 1000.0
        self._rng = random.Random(self._seed)

    def read(self) -> FramePacket | None:
        if self._frame_id >= self._max_frames:
            return None

        frame_id = self._frame_id
        self._frame_id += 1

        phase = (frame_id % 30) / 30.0
        spike = 1.0 if frame_id % 37 in (0, 1, 2) else 0.0
        pixels: list[float] = []
        for idx in range(32):
            base = (math.sin((2.0 * math.pi * phase) + (idx * 0.1)) + 1.0) * 0.5
            noise = self._rng.random() * 0.15
            value = min(1.0, max(0.0, base * 0.85 + noise + (0.2 * spike)))
            pixels.append(value)

        return FramePacket(
            frame_id=frame_id,
            sensor_timestamp=self._sensor_start_ts + (frame_id / self._fps),
            system_timestamp=time.time(),
            pixels=pixels,
            raw_path=None,
        )

    def close(self) -> None:
        return None


class ImageFolderCameraStream(CameraStream):
    """이미지 폴더를 프레임 스트림처럼 읽는 카메라 구현."""

    def __init__(self, camera_cfg: CameraConfig, runtime_cfg: RuntimeConfig) -> None:
        if not camera_cfg.image_dir:
            raise ValueError("camera.image_dir가 비어 있어 image_folder 입력을 사용할 수 없습니다.")

        image_root = Path(camera_cfg.image_dir)
        if not image_root.exists():
            raise FileNotFoundError(f"이미지 루트를 찾을 수 없습니다: {image_root}")

        pattern = camera_cfg.image_pattern or "*.png"
        patterns = [item.strip() for item in pattern.split(",") if item.strip()]
        if not patterns:
            patterns = ["*.png"]

        files: list[Path] = []
        for item in patterns:
            if camera_cfg.recursive:
                files.extend(sorted(image_root.rglob(item)))
            else:
                files.extend(sorted(image_root.glob(item)))

        deduped = sorted({path.resolve() for path in files if path.is_file()})
        if not deduped:
            raise FileNotFoundError(
                f"이미지 파일이 없습니다: root={image_root}, pattern={patterns}"
            )

        max_frames = max(1, int(runtime_cfg.max_frames))
        self._paths = deduped[:max_frames]
        self._fps = max(1, int(camera_cfg.fps))
        self._frame_id = 0
        self._sensor_start_ts = 0.0

        logger.info(
            "ImageFolderCameraStream 초기화: root=%s, pattern=%s, frames=%s",
            image_root,
            patterns,
            len(self._paths),
        )

    def read(self) -> FramePacket | None:
        if self._frame_id >= len(self._paths):
            return None

        path = self._paths[self._frame_id]
        frame_id = self._frame_id
        self._frame_id += 1

        pixels = self._sample_pixels(path)

        return FramePacket(
            frame_id=frame_id,
            sensor_timestamp=self._sensor_start_ts + (frame_id / self._fps),
            system_timestamp=time.time(),
            pixels=pixels,
            raw_path=str(path),
        )

    def close(self) -> None:
        return None

    @staticmethod
    def _sample_pixels(image_path: Path) -> list[float]:
        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "image_folder 입력에는 Pillow가 필요합니다. `pip install pillow` 후 재시도하세요."
            ) from exc

        with Image.open(image_path) as img:
            gray = img.convert("L")
            small = gray.resize((8, 4))
            data = list(small.tobytes())

        return [round(float(value) / 255.0, 6) for value in data]


def build_camera_stream(camera_cfg: CameraConfig, runtime_cfg: RuntimeConfig) -> CameraStream:
    """설정에 맞는 카메라 스트림 구현체를 생성한다."""

    source_type = camera_cfg.source_type.strip().lower()
    if source_type == "image_folder":
        return ImageFolderCameraStream(camera_cfg, runtime_cfg)
    return MockCameraStream(camera_cfg, runtime_cfg)
