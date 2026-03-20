"""Tests for sweep-aware storage behavior."""

from __future__ import annotations

import sqlite3

from video_eval_harness.schemas import SegmentLabelResult
from video_eval_harness.storage import Storage


def test_save_get_round_trip_for_sweep_results(tmp_path):
    storage = Storage(str(tmp_path / "artifacts"))

    result = SegmentLabelResult(
        run_id="run_sweep_test",
        video_id="vid_test",
        segment_id="seg_test",
        start_time_s=0.0,
        end_time_s=10.0,
        model_name="model-a",
        provider="mock",
        parsed_success=True,
        primary_action="walking",
        extraction_variant_id="variant_uniform",
        extraction_label="uniform_8f",
        num_frames_used=8,
        sampling_method_used="uniform",
        sweep_id="sweep_123",
    )

    storage.save_label_result(result)
    round_trip = storage.get_results_by_sweep("sweep_123")

    assert len(round_trip) == 1
    assert round_trip[0].extraction_variant_id == "variant_uniform"
    assert round_trip[0].sampling_method_used == "uniform"


def test_has_result_with_variant_id(tmp_path):
    storage = Storage(str(tmp_path / "artifacts"))

    shared = dict(
        run_id="run_sweep_test",
        video_id="vid_test",
        segment_id="seg_test",
        start_time_s=0.0,
        end_time_s=10.0,
        model_name="model-a",
        provider="mock",
        parsed_success=True,
        sweep_id="sweep_123",
    )

    storage.save_label_result(
        SegmentLabelResult(
            **shared,
            extraction_variant_id="variant_uniform",
            extraction_label="uniform_8f",
            num_frames_used=8,
            sampling_method_used="uniform",
        )
    )
    storage.save_label_result(
        SegmentLabelResult(
            **shared,
            extraction_variant_id="variant_keyframe",
            extraction_label="keyframe_8f",
            num_frames_used=8,
            sampling_method_used="keyframe",
        )
    )

    assert storage.has_result(
        "run_sweep_test", "seg_test", "model-a", "variant_uniform"
    )
    assert storage.has_result(
        "run_sweep_test", "seg_test", "model-a", "variant_keyframe"
    )
    assert not storage.has_result(
        "run_sweep_test", "seg_test", "model-a", "missing_variant"
    )


def test_migration_on_fresh_db(tmp_path):
    storage = Storage(str(tmp_path / "artifacts"))

    with sqlite3.connect(storage.db_path) as conn:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(label_results)").fetchall()
        }

    assert {
        "extraction_variant_id",
        "extraction_label",
        "num_frames_used",
        "sampling_method_used",
        "sweep_id",
    }.issubset(columns)
