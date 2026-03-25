from __future__ import annotations

from pathlib import Path

from video_eval_harness.config import (
    BenchmarkConfig,
    ExtractionConfig,
    SegmentationConfig,
    apply_dense_defaults,
    load_taxonomy,
)
from video_eval_harness.schemas import VerbNounTaxonomy


def test_taxonomy_canonicalizes_synonyms_and_resolves_classes() -> None:
    taxonomy = VerbNounTaxonomy(
        verbs={"open": 3, "close": 4},
        nouns={"door": 7, "drawer": 8},
        verb_synonyms={"opening": "open"},
        noun_synonyms={"doors": "door"},
    )

    assert taxonomy.canonicalize_verb("opening") == "open"
    assert taxonomy.canonicalize_noun("doors") == "door"
    assert taxonomy.get_verb_class("opening") == 3
    assert taxonomy.get_noun_class("doors") == 7


def test_load_taxonomy_reads_yaml_file(tmp_path: Path) -> None:
    taxonomy_path = tmp_path / "taxonomy.yaml"
    taxonomy_path.write_text(
        "\n".join(
            [
                "verbs:",
                "  open: 3",
                "nouns:",
                "  door: 7",
                "verb_synonyms:",
                "  opening: open",
                "noun_synonyms:",
                "  doors: door",
            ]
        ),
        encoding="utf-8",
    )

    taxonomy = load_taxonomy(taxonomy_path)

    assert taxonomy.verbs["open"] == 3
    assert taxonomy.nouns["door"] == 7
    assert taxonomy.canonicalize_verb("opening") == "open"
    assert taxonomy.canonicalize_noun("doors") == "door"


def test_apply_dense_defaults_sets_expected_dense_values() -> None:
    config = BenchmarkConfig(
        name="dense-defaults",
        models=["model-a"],
        prompt_version="default",
        segmentation=SegmentationConfig(window_size_s=10.0),
        extraction=ExtractionConfig(num_frames=8),
        labeling_mode="dense",
        taxonomy=None,
    )

    apply_dense_defaults(config)

    assert config.segmentation.window_size_s == 3.0
    assert config.extraction.num_frames == 4
    assert config.prompt_version == "dense_action_label"
    assert config.taxonomy == "configs/taxonomy.yaml"


def test_apply_dense_defaults_preserves_explicit_overrides() -> None:
    config = BenchmarkConfig(
        name="dense-overrides",
        models=["model-a"],
        prompt_version="custom_dense_prompt",
        segmentation=SegmentationConfig(window_size_s=2.5),
        extraction=ExtractionConfig(num_frames=6),
        labeling_mode="dense",
        taxonomy="custom/taxonomy.yaml",
    )

    apply_dense_defaults(config)

    assert config.segmentation.window_size_s == 2.5
    assert config.extraction.num_frames == 6
    assert config.prompt_version == "custom_dense_prompt"
    assert config.taxonomy == "custom/taxonomy.yaml"
