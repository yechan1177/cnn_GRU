from __future__ import annotations

from ..interfaces import FeaturePacker
from ..schemas import FramePacket, PackedFeature


class SimpleFeaturePacker(FeaturePacker):
    """spatial feature와 프레임 메타데이터를 temporal 입력으로 정규화한다."""

    def pack(self, frame: FramePacket, spatial_vector: list[float]) -> PackedFeature:
        if spatial_vector:
            l1_norm = sum(abs(value) for value in spatial_vector)
            mean_val = sum(spatial_vector) / len(spatial_vector)
        else:
            l1_norm = 0.0
            mean_val = 0.0

        return PackedFeature(
            frame_id=frame.frame_id,
            sensor_timestamp=frame.sensor_timestamp,
            system_timestamp=frame.system_timestamp,
            spatial_vector=spatial_vector,
            metadata={
                "vector_dim": len(spatial_vector),
                "l1_norm": round(l1_norm, 6),
                "vector_mean": round(mean_val, 6),
                "raw_path": frame.raw_path,
            },
        )
