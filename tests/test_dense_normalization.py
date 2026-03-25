from __future__ import annotations

from video_eval_harness.labeling.normalization import normalize_action_label
from video_eval_harness.schemas import VerbNounTaxonomy


def build_taxonomy() -> VerbNounTaxonomy:
    return VerbNounTaxonomy(
        verbs={"open": 3, "close": 4},
        nouns={"door": 7, "drawer": 8},
        verb_synonyms={"opening": "open"},
        noun_synonyms={"doors": "door"},
    )


def test_normalize_action_label_canonicalizes_synonyms() -> None:
    label = normalize_action_label(
        '{"verb": "opening", "noun": "doors", "confidence": 0.82}',
        build_taxonomy(),
    )

    assert label is not None
    assert label.verb == "open"
    assert label.noun == "door"
    assert label.verb_class == 3
    assert label.noun_class == 7
    assert label.action == "open door"
    assert label.confidence == 0.82


def test_normalize_action_label_returns_none_for_non_json_response() -> None:
    label = normalize_action_label("No structured label returned.", build_taxonomy())
    assert label is None


def test_normalize_action_label_leaves_unknown_classes_unresolved() -> None:
    label = normalize_action_label(
        '{"verb": "open", "noun": "window", "confidence": 0.55}',
        build_taxonomy(),
    )

    assert label is not None
    assert label.verb == "open"
    assert label.noun == "window"
    assert label.verb_class == 3
    assert label.noun_class is None
    assert label.action == "open window"
