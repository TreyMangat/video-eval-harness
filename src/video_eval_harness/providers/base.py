"""Base provider abstraction for model inference."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ProviderResponse:
    """Standardized response from any provider."""

    text: str
    model: str
    provider: str
    latency_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    estimated_cost: Optional[float] = None
    raw_response: Optional[dict] = None
    error: Optional[str] = None
    success: bool = True


class BaseProvider(abc.ABC):
    """Abstract base class for model providers."""

    provider_name: str = "base"

    @abc.abstractmethod
    def complete(
        self,
        model_id: str,
        prompt: str,
        image_paths: list[str | Path] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> ProviderResponse:
        """Send a completion request with optional images.

        Args:
            model_id: Provider-specific model identifier.
            prompt: Text prompt to send.
            image_paths: Optional list of local image paths to include.
            max_tokens: Maximum tokens in the response.
            temperature: Sampling temperature.

        Returns:
            Standardized ProviderResponse.
        """
        ...

    @abc.abstractmethod
    def list_models(self) -> list[dict]:
        """List available models from this provider."""
        ...
