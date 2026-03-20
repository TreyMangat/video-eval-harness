"""Disk-based caching for API responses and computed results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from diskcache import Cache

from .config import get_artifacts_dir
from .log import get_logger

logger = get_logger(__name__)


class ResponseCache:
    """Cache for API responses keyed by (model, prompt_hash, input_hash)."""

    def __init__(self, cache_dir: Optional[str | Path] = None):
        if cache_dir is None:
            cache_dir = get_artifacts_dir() / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = Cache(str(self.cache_dir), size_limit=2 * 1024**3)  # 2GB

    def make_key(self, model: str, prompt_hash: str, input_hash: str, variant_id: str = "") -> str:
        """Create a cache key.

        When ``variant_id`` is non-empty (sweep runs), it becomes part of
        the key so the same (model, prompt, frames) tuple can have
        distinct cache entries per extraction variant.
        """
        if variant_id:
            return f"{model}:{prompt_hash}:{input_hash}:{variant_id}"
        return f"{model}:{prompt_hash}:{input_hash}"

    def get(self, key: str) -> Optional[str]:
        """Get cached response text."""
        val = self._cache.get(key)
        if val is not None:
            logger.debug(f"Cache hit: {key[:60]}...")
        return val

    def set(self, key: str, response_text: str) -> None:
        """Cache a response."""
        self._cache.set(key, response_text)

    def hash_content(self, content: str | bytes | dict | list) -> str:
        """Hash arbitrary content for cache key."""
        if isinstance(content, (dict, list)):
            content = json.dumps(content, sort_keys=True)
        if isinstance(content, str):
            content = content.encode("utf-8")
        return hashlib.sha256(content).hexdigest()[:16]

    def close(self) -> None:
        self._cache.close()
