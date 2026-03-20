from __future__ import annotations

import httpx

from video_eval_harness.providers.openrouter import OpenRouterProvider


class SequenceClient:
    def __init__(self, responses: list[httpx.Response]):
        self._responses = responses
        self.calls = 0

    def post(self, url: str, json: dict) -> httpx.Response:
        self.calls += 1
        return self._responses.pop(0)

    def close(self) -> None:
        return None


def _response(status_code: int, *, json_body: dict | None = None, headers: dict | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    return httpx.Response(status_code, json=json_body, headers=headers, request=request)


def test_send_request_retries_on_429(monkeypatch) -> None:
    provider = OpenRouterProvider(api_key="test-key")
    provider._client = SequenceClient(
        [
            _response(429, json_body={"error": {"message": "Rate limited"}}, headers={"retry-after": "0"}),
            _response(200, json_body={"choices": [{"message": {"content": "{}"}}]}),
        ]
    )

    monkeypatch.setattr("video_eval_harness.providers.openrouter.time.sleep", lambda *_: None)
    provider._send_request.retry.wait = lambda *_: 0
    provider._send_request.retry.sleep = lambda *_: None

    response = provider._send_request({"model": "openai/test", "messages": []})

    assert response.status_code == 200
    assert provider._client.calls == 2


def test_complete_returns_error_payload_message() -> None:
    provider = OpenRouterProvider(api_key="test-key")
    provider._client = SequenceClient(
        [
            _response(200, json_body={"error": {"message": "Upstream validation failed"}}),
        ]
    )

    result = provider.complete("openai/test", "hello")

    assert result.success is False
    assert result.error == "Upstream validation failed"
    assert result.raw_response == {"error": {"message": "Upstream validation failed"}}


def test_complete_extracts_cost_from_usage_total_cost() -> None:
    provider = OpenRouterProvider(api_key="test-key")
    provider._client = SequenceClient(
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
        ]
    )

    result = provider.complete("openai/test", "hello")

    assert result.success is True
    assert result.prompt_tokens == 120
    assert result.completion_tokens == 24
    assert result.total_tokens == 144
    assert result.estimated_cost == 0.0042
