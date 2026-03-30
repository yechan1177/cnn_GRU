from __future__ import annotations

from vcp.tools.build_rule_context_dataset import RuleState, compute_baseline, label_frame


def test_label_frame_warn_and_hard_conditions() -> None:
    baseline = {
        "det_norm": 0.04,
        "mean_area": 0.004,
        "vehicle_ratio": 0.15,
        "person_ratio": 0.0,
        "bike_ratio": 0.0,
        "roi_risk": 0.20,
        "center_closeness": 0.45,
        "roi_mean_area": 0.003,
        "roi_count_norm": 0.15,
    }

    warn_feature = {
        "det_norm": 0.05,
        "mean_area": 0.007,
        "vehicle_ratio": 0.2,
        "person_ratio": 0.4,
        "bike_ratio": 0.0,
        "roi_risk": 0.35,
        "center_closeness": 0.70,
        "roi_mean_area": 0.009,
        "roi_count_norm": 0.40,
    }
    label, boundary, event_active, _ = label_frame(warn_feature, baseline, RuleState())
    assert label == "brake_warning"
    assert boundary == 1
    assert event_active is True

    hard_feature = {
        **warn_feature,
        "person_ratio": 0.55,
        "roi_risk": 0.48,
        "center_closeness": 0.84,
        "roi_mean_area": 0.024,
    }
    label, boundary, event_active, _ = label_frame(hard_feature, baseline, RuleState())
    assert label == "hard_brake_risk"
    assert boundary == 1
    assert event_active is True


def test_compute_baseline_uses_recent_window() -> None:
    history = [
        {"mean_area": 0.01, "roi_risk": 0.1, "center_closeness": 0.2, "roi_mean_area": 0.01, "roi_count_norm": 0.2},
        {"mean_area": 0.02, "roi_risk": 0.2, "center_closeness": 0.3, "roi_mean_area": 0.02, "roi_count_norm": 0.3},
        {"mean_area": 0.03, "roi_risk": 0.3, "center_closeness": 0.4, "roi_mean_area": 0.03, "roi_count_norm": 0.4},
    ]
    baseline = compute_baseline(history, baseline_window=2)
    assert round(baseline["mean_area"], 6) == 0.025
    assert round(baseline["roi_risk"], 6) == 0.25
    assert round(baseline["center_closeness"], 6) == 0.35
    assert round(baseline["roi_mean_area"], 6) == 0.025
    assert round(baseline["roi_count_norm"], 6) == 0.35
