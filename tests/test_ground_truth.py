from __future__ import annotations

import pytest

from video_eval_harness.evaluation.metrics import compute_ground_truth_accuracy
from video_eval_harness.schemas import GroundTruthLabel, SegmentLabelResult


def _ground_truth(segment_id: str, primary_action: str) -> GroundTruthLabel:
    return GroundTruthLabel(
        video_id="vid_ground_truth",
        segment_id=segment_id,
        start_time_s=0.0,
        end_time_s=10.0,
        primary_action=primary_action,
    )


def _result(model_name: str, segment_id: str, primary_action: str) -> SegmentLabelResult:
    return SegmentLabelResult(
        run_id="run_ground_truth",
        video_id="vid_ground_truth",
        segment_id=segment_id,
        start_time_s=0.0,
        end_time_s=10.0,
        model_name=model_name,
        provider="openrouter",
        primary_action=primary_action,
        parsed_success=True,
        confidence=0.9,
        latency_ms=100.0,
        estimated_cost=0.001,
    )


def test_compute_ground_truth_accuracy_reports_exact_and_fuzzy_rates() -> None:
    ground_truth = [
        _ground_truth("seg-1", "walking"),
        _ground_truth("seg-2", "displaying a video test pattern"),
        _ground_truth("seg-3", "chopping vegetables"),
    ]
    results = [
        _result("model-a", "seg-1", "walking"),
        _result("model-a", "seg-2", "displaying a test pattern"),
        _result("model-a", "seg-3", "chopping vegetables"),
        _result("model-b", "seg-1", "running"),
        _result("model-b", "seg-2", "displaying a television test pattern"),
        _result("model-b", "seg-3", "chopping vegetables"),
    ]

    metrics = compute_ground_truth_accuracy(results, ground_truth)

    assert metrics["model-a"]["exact_match_rate"] == pytest.approx(2 / 3)
    assert metrics["model-a"]["fuzzy_match_rate"] == pytest.approx(1.0)
    assert metrics["model-a"]["evaluated_segments"] == 3
    assert metrics["model-b"]["exact_match_rate"] == pytest.approx(1 / 3)
    assert metrics["model-b"]["fuzzy_match_rate"] == pytest.approx(1.0)
    assert metrics["model-b"]["evaluated_segments"] == 3
