from __future__ import annotations

from video_eval_harness.labeling.normalization import extract_json_from_text, parse_model_response


def _base_kwargs() -> dict[str, object]:
    return {
        "run_id": "run_norm_test",
        "video_id": "vid_norm_test",
        "segment_id": "vid_norm_test_seg0000",
        "start_time_s": 0.0,
        "end_time_s": 10.0,
        "model_name": "model-a",
        "provider": "mock",
        "latency_ms": 123.0,
    }


def test_extract_json_from_markdown_wrapped_payload() -> None:
    raw = "Here is the result:\n\n```json\n{\"primary_action\": \"displaying test pattern\", \"confidence\": 0.97}\n```\n"
    parsed = extract_json_from_text(raw)
    assert parsed is not None
    assert parsed["primary_action"] == "displaying test pattern"


def test_extract_json_from_prose_embedded_payload() -> None:
    raw = (
        "After reviewing the frames I found the following object: "
        '{"primary_action": "walking forward", "description": "A person moves ahead."} '
        "This should be enough."
    )
    parsed = extract_json_from_text(raw)
    assert parsed is not None
    assert parsed["description"] == "A person moves ahead."


def test_parse_model_response_fails_gracefully_on_malformed_json() -> None:
    raw = "```json\n{\"primary_action\": \"broken\", \"confidence\": }\n```"
    result = parse_model_response(raw, **_base_kwargs())

    assert result.parsed_success is False
    assert result.parse_error == "Could not extract JSON from response"
    assert result.raw_response_text == raw
