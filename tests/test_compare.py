from __future__ import annotations

import re

from typer.testing import CliRunner

from video_eval_harness.cli import app
from video_eval_harness.schemas import RunConfig, SegmentLabelResult
from video_eval_harness.storage import Storage

runner = CliRunner()


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _make_result(
    run_id: str,
    segment_id: str,
    model_name: str,
    primary_action: str,
    *,
    parsed_success: bool = True,
    latency_ms: float = 100.0,
    confidence: float = 0.8,
    estimated_cost: float = 0.001,
) -> SegmentLabelResult:
    return SegmentLabelResult(
        run_id=run_id,
        video_id="vid_compare_test",
        segment_id=segment_id,
        start_time_s=0.0,
        end_time_s=10.0,
        model_name=model_name,
        provider="openrouter",
        primary_action=primary_action,
        parsed_success=parsed_success,
        latency_ms=latency_ms,
        confidence=confidence,
        estimated_cost=estimated_cost,
        prompt_version="concise",
    )


def test_compare_command_displays_expected_deltas(tmp_path) -> None:
    storage = Storage(str(tmp_path / "artifacts"))
    run_a = "run_compare_a"
    run_b = "run_compare_b"

    storage.save_run(RunConfig(run_id=run_a, models=["model-a", "model-b"], prompt_version="concise"))
    storage.save_run(RunConfig(run_id=run_b, models=["model-a", "model-b"], prompt_version="concise"))

    results_a = [
        _make_result(run_a, "seg-1", "model-a", "walking", latency_ms=100.0, confidence=0.6),
        _make_result(
            run_a,
            "seg-2",
            "model-a",
            "running",
            parsed_success=False,
            latency_ms=200.0,
            confidence=0.6,
        ),
        _make_result(run_a, "seg-1", "model-b", "standing", latency_ms=120.0, confidence=0.7),
        _make_result(run_a, "seg-2", "model-b", "standing", latency_ms=180.0, confidence=0.7),
    ]
    results_b = [
        _make_result(run_b, "seg-1", "model-a", "walking", latency_ms=80.0, confidence=0.8),
        _make_result(run_b, "seg-2", "model-a", "walking", latency_ms=120.0, confidence=0.8),
        _make_result(run_b, "seg-1", "model-b", "walking", latency_ms=100.0, confidence=0.8),
        _make_result(run_b, "seg-2", "model-b", "walking", latency_ms=140.0, confidence=0.8),
    ]

    for result in results_a + results_b:
        storage.save_label_result(result)

    outcome = runner.invoke(app, ["compare", run_a, run_b, "--artifacts", str(storage.artifacts_dir)])
    clean_output = _strip_ansi(outcome.output)

    assert outcome.exit_code == 0
    assert "Compare: run_compare_" in clean_output
    assert "Agreement Delta (run_b - run_a)" in clean_output
    assert "model-a" in clean_output
    assert "model-b" in clean_output
    assert "+50.0%" in clean_output
    assert "-50" in clean_output
    assert "+0.200" in clean_output
    assert "+50.0%" in clean_output


def test_compare_command_fails_when_run_missing(tmp_path) -> None:
    storage = Storage(str(tmp_path / "artifacts"))
    storage.save_run(RunConfig(run_id="existing-run", models=["model-a"], prompt_version="concise"))
    storage.save_label_result(_make_result("existing-run", "seg-1", "model-a", "walking"))

    outcome = runner.invoke(
        app,
        ["compare", "existing-run", "missing-run", "--artifacts", str(storage.artifacts_dir)],
    )

    assert outcome.exit_code == 1
    assert "No results for run missing-run" in _strip_ansi(outcome.output)
