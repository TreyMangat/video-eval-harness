"""Native OpenAI provider implementation (direct API, not via OpenRouter)."""

from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..log import get_logger
from .base import BaseProvider, ProviderResponse

logger = get_logger(__name__)

OPENAI_API_BASE = "https://api.openai.com/v1"


class OpenAINativeProvider(BaseProvider):
    """Native OpenAI API provider for direct access to OpenAI models."""

    provider_name = "openai"

    def __init__(
        self,
        api_key: str,
        timeout_s: float = 120.0,
        max_retries: int = 3,
    ):
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var.")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client = httpx.Client(
            base_url=OPENAI_API_BASE,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(timeout_s),
        )

    def complete(
        self,
        model_id: str,
        prompt: str,
        image_paths: list[str | Path] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> ProviderResponse:
        """Send a chat completion request to the OpenAI API."""
        messages = self._build_messages(prompt, image_paths)

        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        start = time.perf_counter()
        try:
            response = self._send_request(payload)
            latency_ms = (time.perf_counter() - start) * 1000
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error(f"Request to {model_id} failed: {e}")
            return ProviderResponse(
                text="",
                model=model_id,
                provider=self.provider_name,
                latency_ms=latency_ms,
                error=str(e),
                success=False,
            )

        # Parse response
        data = response.json()

        if "error" in data:
            error_msg = data["error"].get("message", str(data["error"]))
            logger.error(f"API error from {model_id}: {error_msg}")
            return ProviderResponse(
                text="",
                model=model_id,
                provider=self.provider_name,
                latency_ms=latency_ms,
                error=error_msg,
                success=False,
                raw_response=data,
            )

        choice = data.get("choices", [{}])[0]
        text = choice.get("message", {}).get("content", "")
        usage = data.get("usage", {})

        # Extract cost from OpenAI usage response if available.
        # OpenAI may include cost information in newer API responses;
        # otherwise we leave it as None for the caller to compute.
        estimated_cost = None
        if usage.get("total_cost") is not None:
            estimated_cost = float(usage["total_cost"])

        return ProviderResponse(
            text=text,
            model=data.get("model", model_id),
            provider=self.provider_name,
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            estimated_cost=estimated_cost,
            raw_response=data,
            success=True,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    )
    def _send_request(self, payload: dict) -> httpx.Response:
        """Send request with retry logic and rate-limit handling."""
        resp = self._client.post("/chat/completions", json=payload)
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("retry-after", "5"))
            logger.warning(f"Rate limited by OpenAI, waiting {retry_after}s")
            time.sleep(retry_after)
            raise httpx.HTTPStatusError("Rate limited", request=resp.request, response=resp)
        if resp.status_code >= 500:
            logger.warning(f"OpenAI server error ({resp.status_code}), retrying")
            raise httpx.HTTPStatusError("Server error", request=resp.request, response=resp)
        resp.raise_for_status()
        return resp

    def _build_messages(
        self, prompt: str, image_paths: list[str | Path] | None
    ) -> list[dict]:
        """Build OpenAI-compatible messages with optional image content."""
        if not image_paths:
            return [{"role": "user", "content": prompt}]

        content_parts: list[dict] = []

        # Add images first
        for img_path in image_paths:
            img_path = Path(img_path)
            if not img_path.exists():
                logger.warning(f"Image not found, skipping: {img_path}")
                continue

            mime_type = mimetypes.guess_type(str(img_path))[0] or "image/jpeg"
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")

            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{b64}",
                },
            })

        # Add text prompt
        content_parts.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content_parts}]

    def list_models(self) -> list[dict]:
        """Fetch available models from the OpenAI API."""
        try:
            resp = self._client.get("/models")
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", [])
        except Exception as e:
            logger.error(f"Failed to list OpenAI models: {e}")
            return []

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()
