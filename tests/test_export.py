from __future__ import annotations

import json
from pathlib import Path

from video_eval_harness.evaluation.summaries import export_results
from video_eval_harness.schemas import SegmentLabelResult


def _result(
    segment_id: str,
    model_name: str,
    primary_action: str,
    *,
    extraction_label: str = "uniform_8f",
) -> SegmentLabelResult:
    return SegmentLabelResult(
        run_id="run_export_json",
        video_id="vid_export_json",
        segment_id=segment_id,
        start_time_s=0.0,
        end_time_s=10.0,
        model_name=model_name,
        provider="openrouter",
        primary_action=primary_action,
        parsed_success=True,
        confidence=0.9,
        latency_ms=123.0,
        estimated_cost=0.0012,
        extraction_variant_id="variant_uniform_8f",
        extraction_label=extraction_label,
        num_frames_used=8,
        sampling_method_used="uniform",
        sweep_id="sweep_export_json",
        prompt_version="action_label",
    )


def test_export_results_json_round_trips(tmp_path: Path) -> None:
    results = [
        _result("seg-1", "model-a", "walking"),
        _result("seg-2", "model-b", "running"),
    ]

    paths = export_results(results, tmp_path, "run_export_json", formats=["json"])

    assert len(paths) == 1
    json_path = paths[0]
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert len(payload) == 2
    assert payload[0]["model_name"] == "model-a"
    assert payload[0]["primary_action"] == "walking"
    assert payload[0]["extraction_label"] == "uniform_8f"
    assert {"model_name", "primary_action", "extraction_label"} <= payload[0].keys()
