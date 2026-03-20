"""Tests for sweep config parsing and variant planning."""

from __future__ import annotations

import yaml

from video_eval_harness.sweep import ExtractionVariant, SweepOrchestrator, parse_sweep_config


def test_parse_sweep_config_with_non_sweep_yaml():
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
        """
    )

    config = parse_sweep_config(raw)

    assert config.is_sweep is False
    assert len(config.variants) == 1
    assert config.variants[0].label == "uniform_8f"


def test_parse_sweep_config_with_sweep_yaml():
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
        """
    )

    config = parse_sweep_config(raw)

    assert config.is_sweep is True
    assert {variant.label for variant in config.variants} == {
        "uniform_4f",
        "uniform_8f",
        "keyframe_4f",
        "keyframe_8f",
    }


def test_variant_id_determinism():
    variant_a = ExtractionVariant(num_frames=8, method="uniform")
    variant_b = ExtractionVariant(num_frames=8, method="uniform")

    assert variant_a.variant_id == variant_b.variant_id


def test_matrix_cardinality():
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
    plan = SweepOrchestrator(config).plan()

    assert len(config.variants) == 4
    assert len(plan) == 12
