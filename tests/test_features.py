from __future__ import annotations

import json
from pathlib import Path

from vcp.features import Detection, FrameDetections, SemanticFeatureV1, SemanticFeatureV2, semantic_channel_groups

FIXTURE = Path(__file__).parent / "fixtures" / "v1_feature_regression.json"


def test_v1_matches_2026_03_implementation() -> None:
    """리팩터링한 v1 특징이 기존 YOLOSpatialEncoder 계산과 같은 값을 내는지(배포 체크포인트 호환)."""

    fx = json.loads(FIXTURE.read_text(encoding="utf-8"))
    ext = SemanticFeatureV1(conf_threshold=0.0, max_det=fx["max_det"])
    for i, (rows, expected) in enumerate(zip(fx["frames"], fx["expected_v1"], strict=True)):
        frame = FrameDetections(i, i / 15, fx["width"], fx["height"], [Detection(*r[:4], int(r[4]), r[5]) for r in rows])
        got = ext.update(frame)
        assert max(abs(a - b) for a, b in zip(got, expected, strict=True)) < 2e-6


def _approach_frames(fps: float, seconds: float = 2.0) -> list[FrameDetections]:
    """폭이 시간에 대해 지수적으로 커지는 선행차(역 TTC = 0.5/s)."""

    frames = []
    for i in range(int(seconds * fps)):
        t = i / fps
        w = 60.0 * (2.718281828 ** (0.5 * t))
        cx, bottom = 320.0, 300.0
        frames.append(FrameDetections(i, t, 640, 480, [Detection(cx - w / 2, bottom - 0.8 * w, cx + w / 2, bottom, 1, 0.9)]))
    return frames


def test_v2_scale_rate_is_fps_invariant() -> None:
    """같은 물리 상황이면 10/30fps에서 v2 역 TTC 값이 비슷해야 한다(v1 looming은 FPS 의존)."""

    rate_idx = SemanticFeatureV2.keys.index("lead_inv_ttc")
    loom_idx = SemanticFeatureV1.keys.index("looming_score")
    final_v2, final_v1 = {}, {}
    for fps in (10.0, 30.0):
        v2, v1 = SemanticFeatureV2(), SemanticFeatureV1()
        for f in _approach_frames(fps):
            a, b = v2.update(f), v1.update(f)
        final_v2[fps], final_v1[fps] = a[rate_idx], b[loom_idx]
    assert abs(final_v2[10.0] - 0.5) < 0.05 and abs(final_v2[30.0] - 0.5) < 0.05
    assert final_v1[10.0] > 2.0 * final_v1[30.0]


def test_v2_new_object_does_not_spike() -> None:
    """신규 객체 진입 첫 프레임은 변화율 0(v1의 looming 급등 제거)."""

    ext = SemanticFeatureV2()
    empty = FrameDetections(0, 0.0, 640, 480, [])
    ext.update(empty)
    out = ext.update(FrameDetections(1, 0.1, 640, 480, [Detection(250, 200, 390, 320, 1, 0.9)]))
    assert out[SemanticFeatureV2.keys.index("lead_inv_ttc")] == 0.0
    assert out[SemanticFeatureV2.keys.index("lead_present")] == 1.0


def test_semantic_groups_cover_all_features() -> None:
    for version in ("v1", "v2"):
        covered = sorted({i for g in semantic_channel_groups(version) for i in g})
        assert covered == list(range(16))
