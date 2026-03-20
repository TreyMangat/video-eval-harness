"""Tests for extraction sweep helpers and storage integration."""

from __future__ import annotations

import sqlite3

import yaml

from video_eval_harness.storage import Storage
from video_eval_harness.schemas import SegmentLabelResult
from video_eval_harness.sweep import (
    ExtractionVariant,
    SweepOrchestrator,
    parse_sweep_config,
)


def test_parse_non_sweep_yaml_format():
    raw = yaml.safe_load(
        """
        name: baseline
        models: [model-a, model-b]
        prompt_version: concise
        segmentation:
          mode: fixed_window
          window_size_s: 10
        extraction:
          num_frames: 8
          method: uniform
          image_format: jpg
        """
    )

    config = parse_sweep_config(raw)

    assert config.is_sweep is False
    assert len(config.variants) == 1
    assert config.variants[0].num_frames == 8
    assert config.variants[0].method == "uniform"


def test_parse_sweep_yaml_format():
    raw = yaml.safe_load(
        """
        name: sweep
        models: [model-a, model-b]
        prompt_version: concise
        segmentation:
          mode: fixed_window
          window_size_s: 10
        extraction:
          sweep:
            num_frames: [4, 8]
            methods: [uniform, keyframe]
          image_format: jpg
          image_quality: 90
        """
    )

    config = parse_sweep_config(raw)
    variants = config.variants

    assert config.is_sweep is True
    assert len(variants) == 4
    assert {variant.label for variant in variants} == {
        "uniform_4f",
        "uniform_8f",
        "keyframe_4f",
        "keyframe_8f",
    }


def test_variant_id_is_deterministic_and_unique():
    variant_a = ExtractionVariant(num_frames=8, method="uniform")
    variant_b = ExtractionVariant(num_frames=8, method="uniform")
    variant_c = ExtractionVariant(num_frames=8, method="keyframe")
    variant_d = ExtractionVariant(num_frames=16, method="uniform")

    assert variant_a.variant_id == variant_b.variant_id
    assert len({variant_a.variant_id, variant_c.variant_id, variant_d.variant_id}) == 3


def test_sweep_matrix_cardinality():
    raw = yaml.safe_load(
        """
        name: sweep
        models: [model-a, model-b, model-c]
        prompt_version: concise
        segmentation:
          mode: fixed_window
          window_size_s: 10
        extraction:
          sweep:
            num_frames: [4, 8]
            methods: [uniform, keyframe]
        """
    )

    config = parse_sweep_config(raw)
    orchestrator = SweepOrchestrator(config)
    plan = orchestrator.plan()

    assert len(config.variants) == 4
    assert len(plan) == 12
    assert {(item.model_name, item.variant.label) for item in plan} == {
        ("model-a", "uniform_4f"),
        ("model-a", "uniform_8f"),
        ("model-a", "keyframe_4f"),
        ("model-a", "keyframe_8f"),
        ("model-b", "uniform_4f"),
        ("model-b", "uniform_8f"),
        ("model-b", "keyframe_4f"),
        ("model-b", "keyframe_8f"),
        ("model-c", "uniform_4f"),
        ("model-c", "uniform_8f"),
        ("model-c", "keyframe_4f"),
        ("model-c", "keyframe_8f"),
    }


def test_storage_round_trip_by_sweep_id(tmp_path):
    storage = Storage(str(tmp_path / "artifacts"))

    base = dict(
        run_id="run_sweep_test",
        video_id="vid_test",
        segment_id="seg_test",
        start_time_s=0.0,
        end_time_s=10.0,
        model_name="model-a",
        provider="mock",
        parsed_success=True,
        primary_action="walking",
        sweep_id="sweep_123",
    )

    storage.save_label_result(
        SegmentLabelResult(
            **base,
            extraction_variant_id="variant_uniform",
            extraction_label="uniform_8f",
            num_frames_used=8,
            sampling_method_used="uniform",
        )
    )
    storage.save_label_result(
        SegmentLabelResult(
            **base,
            extraction_variant_id="variant_keyframe",
            extraction_label="keyframe_8f",
            num_frames_used=8,
            sampling_method_used="keyframe",
        )
    )

    sweep_results = storage.get_results_by_sweep("sweep_123")

    assert len(sweep_results) == 2
    assert [result.extraction_variant_id for result in sweep_results] == [
        "variant_keyframe",
        "variant_uniform",
    ]
    assert storage.has_result(
        "run_sweep_test", "seg_test", "model-a", "variant_uniform"
    )
    assert storage.has_result(
        "run_sweep_test", "seg_test", "model-a", "variant_keyframe"
    )


def test_storage_migrates_old_label_results_schema(tmp_path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    db_path = artifacts_dir / "vbench.db"

    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE label_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                video_id TEXT NOT NULL,
                segment_id TEXT NOT NULL,
                start_time_s REAL,
                end_time_s REAL,
                model_name TEXT NOT NULL,
                provider TEXT,
                primary_action TEXT,
                secondary_actions TEXT,
                description TEXT,
                objects TEXT,
                environment_context TEXT,
                confidence REAL,
                reasoning_summary_or_notes TEXT,
                uncertainty_flags TEXT,
                raw_response_text TEXT,
                parsed_success INTEGER,
                parse_error TEXT,
                latency_ms REAL,
                estimated_cost REAL,
                prompt_version TEXT,
                timestamp TEXT,
                UNIQUE(run_id, segment_id, model_name)
            );
            """
        )
        conn.execute(
            """
            INSERT INTO label_results (
                run_id,
                video_id,
                segment_id,
                start_time_s,
                end_time_s,
                model_name,
                provider,
                secondary_actions,
                objects,
                uncertainty_flags,
                parsed_success
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "run_old",
                "vid_old",
                "seg_old",
                0.0,
                10.0,
                "model-a",
                "mock",
                "[]",
                "[]",
                "[]",
                1,
            ),
        )

    storage = Storage(str(artifacts_dir))
    storage.save_label_result(
        SegmentLabelResult(
            run_id="run_old",
            video_id="vid_old",
            segment_id="seg_old",
            start_time_s=0.0,
            end_time_s=10.0,
            model_name="model-a",
            provider="mock",
            parsed_success=True,
            extraction_variant_id="variant_b",
            extraction_label="uniform_4f",
            num_frames_used=4,
            sampling_method_used="uniform",
            sweep_id="sweep_migrated",
        )
    )

    migrated_results = storage.get_run_results("run_old")

    assert len(migrated_results) == 2
    assert {result.extraction_variant_id for result in migrated_results} == {
        "",
        "variant_b",
    }
