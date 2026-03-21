from __future__ import annotations

import pytest

from video_eval_harness.evaluation.metrics import (
    compute_action_similarity,
    compute_agreement_matrix,
)
from video_eval_harness.schemas import SegmentLabelResult


def _make_result(
    segment_id: str,
    model_name: str,
    primary_action: str,
    *,
    parsed_success: bool = True,
) -> SegmentLabelResult:
    return SegmentLabelResult(
        run_id="run-1",
        video_id="video-1",
        segment_id=segment_id,
        start_time_s=0.0,
        end_time_s=5.0,
        model_name=model_name,
        provider="test-provider",
        primary_action=primary_action,
        parsed_success=parsed_success,
        confidence=0.9,
        latency_ms=25.0,
        estimated_cost=0.001,
    )


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("chopping vegetables", "chopping vegetables", 1.0),
        ("walking forward", "chopping vegetables", 0.0),
        ("the person is chopping vegetables", "chopping vegetables", 1.0),
    ],
)
def test_compute_action_similarity_exact_cases(
    left: str,
    right: str,
    expected: float,
) -> None:
    assert compute_action_similarity(left, right) == expected


@pytest.mark.parametrize(
    ("left", "right", "minimum"),
    [
        ("chopping vegetables", "cutting vegetables", 0.85),
        ("loading parts", "feeding components", 0.7),
        ("chopping", "chopping vegetables", 0.9),
        ("displaying test pattern", "showing test pattern", 0.85),
        ("loading parts into machine", "loading parts into press", 0.33),
    ],
)
def test_compute_action_similarity_semantic_matches(
    left: str,
    right: str,
    minimum: float,
) -> None:
    assert compute_action_similarity(left, right) >= minimum


def test_compute_agreement_matrix_uses_similarity_scores() -> None:
    results = [
        _make_result("segment-1", "model-a", "chopping vegetables"),
        _make_result("segment-1", "model-b", "cutting vegetables"),
        _make_result("segment-2", "model-a", "the person is chopping vegetables"),
        _make_result("segment-2", "model-b", "chopping vegetables"),
        _make_result("segment-3", "model-a", "walking forward"),
        _make_result("segment-3", "model-b", "chopping vegetables"),
    ]

    expected = (
        compute_action_similarity("chopping vegetables", "cutting vegetables")
        + compute_action_similarity("the person is chopping vegetables", "chopping vegetables")
        + compute_action_similarity("walking forward", "chopping vegetables")
    ) / 3

    agreement = compute_agreement_matrix(results)

    assert agreement["model-a"]["model-a"] == 1.0
    assert agreement["model-b"]["model-b"] == 1.0
    assert agreement["model-a"]["model-b"] == pytest.approx(expected)
    assert agreement["model-b"]["model-a"] == pytest.approx(expected)
