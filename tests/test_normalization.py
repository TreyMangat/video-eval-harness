"""Tests for response parsing and normalization."""

from video_eval_harness.labeling.normalization import (
    extract_json_from_text,
    normalize_confidence,
    normalize_string_list,
    parse_model_response,
)


class TestExtractJson:
    def test_direct_json(self):
        text = '{"primary_action": "cooking", "confidence": 0.9}'
        result = extract_json_from_text(text)
        assert result is not None
        assert result["primary_action"] == "cooking"

    def test_markdown_code_block(self):
        text = '```json\n{"primary_action": "walking"}\n```'
        result = extract_json_from_text(text)
        assert result is not None
        assert result["primary_action"] == "walking"

    def test_json_with_preamble(self):
        text = 'Here is my analysis:\n{"primary_action": "reading"}'
        result = extract_json_from_text(text)
        assert result is not None
        assert result["primary_action"] == "reading"

    def test_no_json(self):
        text = "This is just plain text with no JSON."
        result = extract_json_from_text(text)
        assert result is None

    def test_empty_string(self):
        assert extract_json_from_text("") is None
        assert extract_json_from_text("   ") is None

    def test_invalid_json(self):
        text = '{"primary_action": "cooking", bad}'
        # Should fail gracefully
        extract_json_from_text(text)
        # Might or might not extract depending on regex match
        # The important thing is no exception


class TestNormalizeStringList:
    def test_none(self):
        assert normalize_string_list(None) == []

    def test_string(self):
        assert normalize_string_list("hello") == ["hello"]

    def test_empty_string(self):
        assert normalize_string_list("  ") == []

    def test_list(self):
        assert normalize_string_list(["a", "b"]) == ["a", "b"]

    def test_list_with_none(self):
        assert normalize_string_list(["a", None, "b"]) == ["a", "b"]


class TestNormalizeConfidence:
    def test_normal_float(self):
        assert normalize_confidence(0.85) == 0.85

    def test_percentage(self):
        assert normalize_confidence(85) == 0.85

    def test_none(self):
        assert normalize_confidence(None) is None

    def test_string(self):
        assert normalize_confidence("0.9") == 0.9

    def test_clamped(self):
        assert normalize_confidence(1.5) == 1.0
        assert normalize_confidence(-0.5) == 0.0


class TestParseModelResponse:
    def _base_kwargs(self):
        return dict(
            run_id="run_test",
            video_id="vid_test",
            segment_id="vid_test_seg0000",
            start_time_s=0.0,
            end_time_s=10.0,
            model_name="gpt4o",
            provider="openrouter",
            latency_ms=1000.0,
        )

    def test_successful_parse(self):
        raw = '{"primary_action": "cooking", "secondary_actions": ["stirring"], "description": "Cooking food", "objects": ["pan"], "environment_context": "kitchen", "confidence": 0.9, "reasoning_summary_or_notes": "Clear cooking action", "uncertainty_flags": []}'
        result = parse_model_response(raw, **self._base_kwargs())
        assert result.parsed_success is True
        assert result.primary_action == "cooking"
        assert result.confidence == 0.9

    def test_failed_parse(self):
        raw = "I cannot analyze this video segment."
        result = parse_model_response(raw, **self._base_kwargs())
        assert result.parsed_success is False
        assert result.parse_error is not None
        assert result.raw_response_text == raw

    def test_partial_json(self):
        raw = '{"primary_action": "walking"}'
        result = parse_model_response(raw, **self._base_kwargs())
        assert result.parsed_success is True
        assert result.primary_action == "walking"
        assert result.secondary_actions == []

    def test_metadata_preserved(self):
        raw = '{"primary_action": "test"}'
        result = parse_model_response(raw, **self._base_kwargs())
        assert result.run_id == "run_test"
        assert result.model_name == "gpt4o"
        assert result.latency_ms == 1000.0
