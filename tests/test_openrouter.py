from __future__ import annotations

from pathlib import Path

import httpx

from video_eval_harness.providers.openrouter import OpenRouterProvider


class MockHttpxClient:
    def __init__(self, responses: list[httpx.Response]):
        self._responses = responses
        self.calls = 0
        self.requests: list[tuple[str, dict]] = []

    def post(self, url: str, json: dict) -> httpx.Response:
        self.calls += 1
        self.requests.append((url, json))
        return self._responses.pop(0)

    def close(self) -> None:
        return None


def _response(
    status_code: int,
    *,
    json_body: dict | None = None,
    headers: dict | None = None,
) -> httpx.Response:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return httpx.Response(status_code, json=json_body, headers=headers, request=request)


def _make_provider(
    monkeypatch,
    responses: list[httpx.Response],
) -> tuple[OpenRouterProvider, MockHttpxClient]:
    client = MockHttpxClient(responses)
    monkeypatch.setattr("video_eval_harness.providers.openrouter.httpx.Client", lambda *args, **kwargs: client)
    return OpenRouterProvider(api_key="test-key"), client


def test_complete_parses_successful_response(monkeypatch) -> None:
    provider, client = _make_provider(
        monkeypatch,
        [
            _response(
                200,
                json_body={
                    "model": "openai/test",
                    "choices": [{"message": {"content": '{"primary_action":"testing"}'}}],
                    "usage": {"prompt_tokens": 120, "completion_tokens": 24, "total_tokens": 144},
                },
            ),
        ],
    )

    result = provider.complete("openai/test", "hello")

    assert result.success is True
    assert result.text == '{"primary_action":"testing"}'
    assert result.model == "openai/test"
    assert result.prompt_tokens == 120
    assert result.completion_tokens == 24
    assert result.total_tokens == 144
    assert client.calls == 1
    assert client.requests[0][0] == "/chat/completions"


def test_send_request_retries_on_429(monkeypatch) -> None:
    provider, client = _make_provider(
        monkeypatch,
        [
            _response(429, json_body={"error": {"message": "Rate limited"}}, headers={"retry-after": "0"}),
            _response(200, json_body={"choices": [{"message": {"content": "{}"}}]}),
        ],
    )

    monkeypatch.setattr("video_eval_harness.providers.openrouter.time.sleep", lambda *_: None)
    provider._send_request.retry.wait = lambda *_: 0
    provider._send_request.retry.sleep = lambda *_: None

    result = provider.complete("openai/test", "hello")

    assert result.success is True
    assert client.calls == 2


def test_complete_returns_error_payload_message(monkeypatch) -> None:
    provider, _client = _make_provider(
        monkeypatch,
        [
            _response(200, json_body={"error": {"message": "Upstream validation failed"}}),
        ],
    )

    result = provider.complete("openai/test", "hello")

    assert result.success is False
    assert result.error == "Upstream validation failed"
    assert result.raw_response == {"error": {"message": "Upstream validation failed"}}


def test_complete_returns_failure_on_http_error_response(monkeypatch) -> None:
    provider, _client = _make_provider(
        monkeypatch,
        [
            _response(400, json_body={"error": {"message": "Bad request"}}),
        ],
    )

    result = provider.complete("openai/test", "hello")

    assert result.success is False
    assert result.error is not None


def test_complete_extracts_cost_from_usage_total_cost(monkeypatch) -> None:
    provider, _client = _make_provider(
        monkeypatch,
        [
            _response(
                200,
                json_body={
                    "model": "openai/test",
                    "choices": [{"message": {"content": '{"primary_action":"testing"}'}}],
                    "usage": {
                        "prompt_tokens": 120,
                        "completion_tokens": 24,
                        "total_tokens": 144,
                        "total_cost": 0.0042,
                    },
                },
            ),
        ],
    )

    result = provider.complete("openai/test", "hello")

    assert result.success is True
    assert result.prompt_tokens == 120
    assert result.completion_tokens == 24
    assert result.total_tokens == 144
    assert result.estimated_cost == 0.0042


def test_complete_extracts_cost_from_usage_cost_fallback(monkeypatch) -> None:
    provider, _client = _make_provider(
        monkeypatch,
        [
            _response(
                200,
                json_body={
                    "model": "openai/test",
                    "choices": [{"message": {"content": '{"primary_action":"testing"}'}}],
                    "usage": {
                        "prompt_tokens": 80,
                        "completion_tokens": 20,
                        "total_tokens": 100,
                        "cost": 0.0015,
                    },
                },
            ),
        ],
    )

    result = provider.complete("openai/test", "hello")

    assert result.success is True
    assert result.estimated_cost == 0.0015


def test_build_messages_returns_text_only_message_without_images(monkeypatch) -> None:
    provider, _client = _make_provider(monkeypatch, [])

    messages = provider._build_messages("describe the scene", None)

    assert messages == [{"role": "user", "content": "describe the scene"}]


def test_build_messages_skips_missing_images(monkeypatch, tmp_path) -> None:
    provider, _client = _make_provider(monkeypatch, [])
    image_path = Path(tmp_path / "frame.jpg")
    image_path.write_bytes(b"fake-image")

    messages = provider._build_messages(
        "describe the scene",
        [tmp_path / "missing.jpg", image_path],
    )

    assert len(messages) == 1
    content = messages[0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "image_url"
    assert content[1] == {"type": "text", "text": "describe the scene"}
