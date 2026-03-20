"""Tests for sweep-aware metrics."""

from __future__ import annotations

from video_eval_harness.evaluation.metrics import compute_sweep_metrics
from video_eval_harness.schemas import SegmentLabelResult


def make_result(
    model_name: str,
    segment_id: str,
    variant_id: str,
    variant_label: str,
    primary_action: str,
    *,
    parsed_success: bool = True,
    latency_ms: float = 100.0,
) -> SegmentLabelResult:
    return SegmentLabelResult(
        run_id="run_metrics_test",
        video_id="vid_test",
        segment_id=segment_id,
        start_time_s=0.0,
        end_time_s=10.0,
        model_name=model_name,
        provider="mock",
        primary_action=primary_action,
        parsed_success=parsed_success,
        latency_ms=latency_ms,
        estimated_cost=0.001,
        confidence=0.8,
        extraction_variant_id=variant_id,
        extraction_label=variant_label,
        num_frames_used=4,
        sampling_method_used=variant_label.split("_", 1)[0],
        sweep_id="sweep_123",
    )


def test_compute_sweep_metrics_with_synthetic_results():
    results = [
        make_result("model-a", "seg-1", "var_uniform", "uniform_4f", "walking"),
        make_result("model-a", "seg-2", "var_uniform", "uniform_4f", "standing"),
        make_result("model-a", "seg-1", "var_keyframe", "keyframe_4f", "walking"),
        make_result("model-a", "seg-2", "var_keyframe", "keyframe_4f", "standing"),
        make_result("model-b", "seg-1", "var_uniform", "uniform_4f", "walking"),
        make_result("model-b", "seg-2", "var_uniform", "uniform_4f", "standing"),
        make_result("model-b", "seg-1", "var_keyframe", "keyframe_4f", "walking"),
        make_result(
            "model-b",
            "seg-2",
            "var_keyframe",
            "keyframe_4f",
            "standing",
            parsed_success=False,
        ),
    ]

    metrics = compute_sweep_metrics(results)
    cell_map = {(cell.model_name, cell.variant_label): cell for cell in metrics["cells"]}

    assert cell_map[("model-a", "uniform_4f")].parse_success_rate == 1.0
    assert cell_map[("model-b", "keyframe_4f")].parse_success_rate == 0.5
    assert set(metrics["agreement_by_variant"].keys()) == {"uniform_4f", "keyframe_4f"}


def test_stability_score_perfect_agreement():
    results = [
        make_result("model-a", "seg-1", "var_uniform", "uniform_4f", "walking"),
        make_result("model-a", "seg-1", "var_keyframe", "keyframe_4f", "walking"),
        make_result("model-a", "seg-2", "var_uniform", "uniform_4f", "standing"),
        make_result("model-a", "seg-2", "var_keyframe", "keyframe_4f", "standing"),
        make_result("model-b", "seg-1", "var_uniform", "uniform_4f", "walking"),
        make_result("model-b", "seg-1", "var_keyframe", "keyframe_4f", "walking"),
        make_result("model-b", "seg-2", "var_uniform", "uniform_4f", "standing"),
        make_result("model-b", "seg-2", "var_keyframe", "keyframe_4f", "standing"),
    ]

    stability_map = {
        item.model_name: item for item in compute_sweep_metrics(results)["stability"]
    }

    assert stability_map["model-a"].self_agreement == 1.0
    assert stability_map["model-a"].rank_stability == 1.0


def test_stability_score_edge_cases():
    results = [
        make_result("model-a", "seg-1", "var_uniform", "uniform_4f", "walking"),
        make_result("model-a", "seg-1", "var_keyframe", "keyframe_4f", "running"),
        make_result("model-a", "seg-2", "var_uniform", "uniform_4f", "standing"),
        make_result("model-a", "seg-2", "var_keyframe", "keyframe_4f", "sitting"),
        make_result("model-b", "seg-1", "var_uniform", "uniform_4f", "walking"),
        make_result("model-b", "seg-1", "var_keyframe", "keyframe_4f", "walking"),
        make_result("model-b", "seg-2", "var_uniform", "uniform_4f", "standing"),
        make_result("model-b", "seg-2", "var_keyframe", "keyframe_4f", "standing"),
    ]

    stability_map = {
        item.model_name: item for item in compute_sweep_metrics(results)["stability"]
    }

    assert stability_map["model-a"].self_agreement == 0.0
    assert 0.0 <= stability_map["model-a"].rank_stability <= 1.0
