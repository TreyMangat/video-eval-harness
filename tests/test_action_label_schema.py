from __future__ import annotations

from video_eval_harness.schemas import ActionLabel, LabelingMode, RunConfig, SegmentLabelResult


def test_action_label_schema_round_trips_dense_fields() -> None:
    label = ActionLabel(
        verb="open",
        noun="door",
        verb_class=3,
        noun_class=7,
        action="open door",
        confidence=0.91,
    )

    assert label.verb == "open"
    assert label.noun == "door"
    assert label.action == "open door"
    assert label.verb_class == 3
    assert label.noun_class == 7
    assert label.confidence == 0.91


def test_segment_label_result_accepts_dense_action_label() -> None:
    result = SegmentLabelResult(
        run_id="run_dense_schema",
        video_id="vid_dense",
        segment_id="vid_dense_seg0000",
        start_time_s=0.0,
        end_time_s=3.0,
        model_name="model-a",
        provider="mock",
        primary_action="open door",
        parsed_success=True,
        labeling_mode="dense",
        action_label=ActionLabel(
            verb="open",
            noun="door",
            verb_class=3,
            noun_class=7,
            action="open door",
            confidence=0.88,
        ),
    )

    assert result.labeling_mode == "dense"
    assert result.action_label is not None
    assert result.action_label.verb == "open"
    assert result.action_label.noun == "door"
    assert result.primary_action == "open door"


def test_run_config_supports_dense_labeling_mode() -> None:
    config = RunConfig(
        run_id="run_dense_config",
        models=["model-a", "model-b"],
        prompt_version="dense_action_label",
        labeling_mode=LabelingMode.DENSE,
        taxonomy_path="configs/taxonomy.yaml",
    )

    assert config.labeling_mode == LabelingMode.DENSE
    assert config.taxonomy_path == "configs/taxonomy.yaml"
