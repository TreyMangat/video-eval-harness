from __future__ import annotations

import pytest

from video_eval_harness.evaluation.metrics import compute_action_similarity


@pytest.mark.parametrize(
    ("left", "right", "minimum"),
    [
        ("displaying a video test pattern", "displaying a television test pattern", 0.5),
        ("displaying test pattern", "displaying a video test pattern", 0.5),
        ("chopping vegetables carefully", "chopping vegetables", 0.5),
        ("walking forward", "walking forward", 0.99),
    ],
)
def test_compute_action_similarity_real_run_pairs(
    left: str,
    right: str,
    minimum: float,
) -> None:
    assert compute_action_similarity(left, right) > minimum


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("chopping vegetables", "chopping vegetables"),
        ("walking", "walking"),
        ("displaying a video test pattern", "displaying a video test pattern"),
    ],
)
def test_compute_action_similarity_exact_matches_score_one(left: str, right: str) -> None:
    assert compute_action_similarity(left, right) == 1.0


@pytest.mark.parametrize(
    ("left", "right", "maximum"),
    [
        ("walking", "running", 0.3),
        ("chopping vegetables", "driving a car", 0.1),
        ("standing", "sitting", 0.3),
        ("opening door", "closing door", 0.3),
    ],
)
def test_compute_action_similarity_distinct_actions_stay_low(
    left: str,
    right: str,
    maximum: float,
) -> None:
    assert compute_action_similarity(left, right) < maximum


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("", "", 1.0),
        ("", "walking", 0.0),
        ("walking", "", 0.0),
        ("the person is walking", "walking", 1.0),
        ("someone is chopping vegetables.", "chopping vegetables", 1.0),
    ],
)
def test_compute_action_similarity_edge_cases(left: str, right: str, expected: float) -> None:
    assert compute_action_similarity(left, right) == expected
