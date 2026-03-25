from __future__ import annotations

from video_eval_harness.evaluation.metrics import (
    compute_taxonomy_agreement,
    compute_taxonomy_agreement_matrix,
)
from video_eval_harness.schemas import ActionLabel, SegmentLabelResult


def make_dense_result(
    model_name: str,
    segment_id: str,
    action_label: ActionLabel | None,
    *,
    parsed_success: bool = True,
) -> SegmentLabelResult:
    return SegmentLabelResult(
        run_id="run_dense_metrics",
        video_id="vid_dense",
        segment_id=segment_id,
        start_time_s=0.0,
        end_time_s=3.0,
        model_name=model_name,
        provider="mock",
        primary_action=action_label.action if action_label is not None else None,
        parsed_success=parsed_success,
        labeling_mode="dense",
        action_label=action_label,
    )


def test_compute_taxonomy_agreement_full_match() -> None:
    label_a = ActionLabel(
        verb="open",
        noun="door",
        verb_class=3,
        noun_class=7,
        action="open door",
        confidence=0.9,
    )
    label_b = ActionLabel(
        verb="open",
        noun="door",
        verb_class=3,
        noun_class=7,
        action="open door",
        confidence=0.8,
    )

    agreement = compute_taxonomy_agreement(label_a, label_b)

    assert agreement == {
        "verb_match": 1.0,
        "noun_match": 1.0,
        "action_match": 1.0,
    }


def test_compute_taxonomy_agreement_partial_match_same_verb() -> None:
    label_a = ActionLabel(
        verb="open",
        noun="door",
        verb_class=3,
        noun_class=7,
        action="open door",
        confidence=0.9,
    )
    label_b = ActionLabel(
        verb="open",
        noun="drawer",
        verb_class=3,
        noun_class=8,
        action="open drawer",
        confidence=0.8,
    )

    agreement = compute_taxonomy_agreement(label_a, label_b)

    assert agreement["verb_match"] == 1.0
    assert agreement["noun_match"] == 0.0
    assert agreement["action_match"] == 0.5


def test_compute_taxonomy_agreement_matrix_averages_model_pairs() -> None:
    open_door = ActionLabel(
        verb="open",
        noun="door",
        verb_class=3,
        noun_class=7,
        action="open door",
        confidence=0.9,
    )
    open_drawer = ActionLabel(
        verb="open",
        noun="drawer",
        verb_class=3,
        noun_class=8,
        action="open drawer",
        confidence=0.8,
    )
    close_door = ActionLabel(
        verb="close",
        noun="door",
        verb_class=4,
        noun_class=7,
        action="close door",
        confidence=0.85,
    )

    results = [
        make_dense_result("model-a", "seg-1", open_door),
        make_dense_result("model-b", "seg-1", open_door),
        make_dense_result("model-a", "seg-2", open_drawer),
        make_dense_result("model-b", "seg-2", close_door),
    ]

    matrix = compute_taxonomy_agreement_matrix(results, ["model-a", "model-b"])

    assert matrix["model-a"]["model-a"]["action_agreement"] == 1.0
    assert matrix["model-a"]["model-b"]["verb_agreement"] == 0.5
    assert matrix["model-a"]["model-b"]["noun_agreement"] == 0.5
    assert matrix["model-a"]["model-b"]["action_agreement"] == 0.5
    assert matrix["model-b"]["model-a"] == matrix["model-a"]["model-b"]


def test_compute_taxonomy_agreement_matrix_ignores_failed_or_missing_labels() -> None:
    open_door = ActionLabel(
        verb="open",
        noun="door",
        verb_class=3,
        noun_class=7,
        action="open door",
        confidence=0.9,
    )

    results = [
        make_dense_result("model-a", "seg-1", open_door),
        make_dense_result("model-b", "seg-1", None, parsed_success=False),
    ]

    matrix = compute_taxonomy_agreement_matrix(results, ["model-a", "model-b"])

    assert matrix["model-a"]["model-b"]["verb_agreement"] == 0.0
    assert matrix["model-a"]["model-b"]["noun_agreement"] == 0.0
    assert matrix["model-a"]["model-b"]["action_agreement"] == 0.0
