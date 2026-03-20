from __future__ import annotations

import pytest

from video_eval_harness.evaluation.metrics import compute_action_similarity


@pytest.mark.parametrize(
    ("left", "right", "minimum"),
    [
        (
            "displaying a video test pattern",
            "displaying a television test pattern",
            0.5,
        ),
        ("displaying test pattern", "displaying a video test pattern", 0.5),
        ("walking forward", "walking forward", 0.99),
    ],
)
def test_compute_action_similarity_real_run_pairs(left: str, right: str, minimum: float) -> None:
    assert compute_action_similarity(left, right) > minimum


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("", "", 1.0),
        ("", "walking", 0.0),
        ("walking", "", 0.0),
        ("walking", "walking", 1.0),
        ("chopping vegetables", "driving a car", 0.0),
    ],
)
def test_compute_action_similarity_edge_cases(left: str, right: str, expected: float) -> None:
    assert compute_action_similarity(left, right) == expected
