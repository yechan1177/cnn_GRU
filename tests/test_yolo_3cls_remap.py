from __future__ import annotations

from vcp.tools.remap_yolo_dataset_to_3cls import remap_label_lines


def test_remap_label_lines_maps_and_drops_classes() -> None:
    class_names = ["bicycle", "bus", "car", "cng", "motorcycle", "other-vehicle", "person", "rickshaw"]
    lines = [
        "0 0.5 0.5 0.1 0.1",
        "1 0.5 0.5 0.2 0.2",
        "4 0.5 0.5 0.3 0.3",
        "6 0.5 0.5 0.4 0.4",
        "9 0.5 0.5 0.5 0.5",
    ]
    remapped, stats = remap_label_lines(lines, class_names)
    assert remapped == [
        "2 0.5 0.5 0.1 0.1",
        "1 0.5 0.5 0.2 0.2",
        "2 0.5 0.5 0.3 0.3",
        "0 0.5 0.5 0.4 0.4",
    ]
    assert stats["boxes_input"] == 4
    assert stats["boxes_kept"] == 4
    assert stats["boxes_dropped"] == 0
