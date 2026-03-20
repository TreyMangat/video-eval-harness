"""Native Google Gemini provider implementation."""

from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..log import get_logger
from .base import BaseProvider, ProviderResponse

logger = get_logger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiNativeProvider(BaseProvider):
    """Native Google Gemini API provider.

    Uses the Gemini REST API directly with API key authentication
    passed as a query parameter.
    """

    provider_name = "gemini"

    def __init__(
        self,
        api_key: str,
        timeout_s: float = 120.0,
        max_retries: int = 3,
    ):
        if not api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY env var.")
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client = httpx.Client(
            base_url=GEMINI_API_BASE,
            headers={"Content-Type": "application/json"},
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
        """Send a generateContent request to Gemini."""
        parts = self._build_parts(prompt, image_paths)

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }

        endpoint = f"/models/{model_id}:generateContent"

        start = time.perf_counter()
        try:
            response = self._send_request(endpoint, payload)
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

        data = response.json()

        # Check for API-level errors
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

        # Extract text from Gemini response structure
        # Response: { "candidates": [{ "content": { "parts": [{ "text": "..." }] } }] }
        candidates = data.get("candidates", [])
        text = ""
        if candidates:
            candidate_parts = candidates[0].get("content", {}).get("parts", [])
            text_pieces = [p["text"] for p in candidate_parts if "text" in p]
            text = "".join(text_pieces)

        # Extract token usage from usageMetadata
        usage = data.get("usageMetadata", {})
        prompt_tokens = usage.get("promptTokenCount")
        completion_tokens = usage.get("candidatesTokenCount")
        total_tokens = usage.get("totalTokenCount")

        return ProviderResponse(
            text=text,
            model=model_id,
            provider=self.provider_name,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw_response=data,
            success=True,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError)),
    )
    def _send_request(self, endpoint: str, payload: dict) -> httpx.Response:
        """Send request with retry logic and rate-limit handling."""
        resp = self._client.post(
            endpoint,
            json=payload,
            params={"key": self.api_key},
        )
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("retry-after", "5"))
            logger.warning(f"Rate limited, waiting {retry_after}s")
            time.sleep(retry_after)
            raise httpx.HTTPStatusError("Rate limited", request=resp.request, response=resp)
        if resp.status_code >= 500:
            raise httpx.HTTPStatusError("Server error", request=resp.request, response=resp)
        resp.raise_for_status()
        return resp

    def _build_parts(
        self, prompt: str, image_paths: list[str | Path] | None
    ) -> list[dict]:
        """Build Gemini-format parts with optional inline image data."""
        parts: list[dict] = []

        # Add images first
        if image_paths:
            for img_path in image_paths:
                img_path = Path(img_path)
                if not img_path.exists():
                    logger.warning(f"Image not found, skipping: {img_path}")
                    continue

                mime_type = mimetypes.guess_type(str(img_path))[0] or "image/jpeg"
                with open(img_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")

                parts.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": b64,
                    }
                })

        # Add text prompt
        parts.append({"text": prompt})

        return parts

    def list_models(self) -> list[dict]:
        """Fetch available models from the Gemini API."""
        try:
            resp = self._client.get("/models", params={"key": self.api_key})
            resp.raise_for_status()
            data = resp.json()
            return data.get("models", [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def close(self) -> None:
        self._client.close()
