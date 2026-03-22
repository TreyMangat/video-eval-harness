"""Disk-based caching for API responses and computed results."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Optional

from diskcache import Cache

from .config import get_artifacts_dir
from .log import get_logger

logger = get_logger(__name__)


class ResponseCache:
    """Cache for API responses keyed by model, prompt, input, and input mode."""

    def __init__(self, cache_dir: Optional[str | Path] = None):
        if cache_dir is None:
            cache_dir = get_artifacts_dir() / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache = Cache(str(self.cache_dir), size_limit=2 * 1024**3)  # 2GB

    def _normalize_input_mode(self, input_mode: object) -> str:
        if not isinstance(input_mode, str):
            return ""
        normalized = input_mode.strip().lower()
        if normalized in {"frames", "video"}:
            return normalized
        return ""

    def _infer_input_mode_from_caller(self) -> str:
        """Infer the effective input mode from a runner-style calling context."""
        frame = inspect.currentframe()
        make_key_frame = frame.f_back if frame is not None else None
        caller = make_key_frame.f_back if make_key_frame is not None else None
        try:
            if caller is None:
                return ""

            actual_input_mode = self._normalize_input_mode(caller.f_locals.get("actual_input_mode"))
            if actual_input_mode:
                return actual_input_mode

            explicit_input_mode = self._normalize_input_mode(caller.f_locals.get("input_mode"))
            if explicit_input_mode:
                return explicit_input_mode

            runner = caller.f_locals.get("self")
            requested_mode = self._normalize_input_mode(getattr(runner, "input_mode", ""))
            if requested_mode == "frames":
                return "frames"
            if requested_mode == "video":
                return "video"

            if getattr(runner, "input_mode", "").strip().lower() != "auto":
                return ""

            model_cfg = caller.f_locals.get("model_cfg")
            segment = caller.f_locals.get("segment")
            video_id = getattr(segment, "video_id", None)
            video_paths = getattr(runner, "video_paths", {})
            supports_video = bool(getattr(model_cfg, "supports_video", False))
            if supports_video and video_id in video_paths:
                return "video"
            return "frames"
        finally:
            del caller
            del make_key_frame
            del frame

    def make_key(
        self,
        model: str,
        prompt_hash: str,
        input_hash: str,
        variant_id: str = "",
        input_mode: str = "",
    ) -> str:
        """Create a cache key.

        When ``variant_id`` is non-empty (sweep runs), it becomes part of
        the key so the same (model, prompt, frames) tuple can have
        distinct cache entries per extraction variant.
        """
        effective_input_mode = self._normalize_input_mode(input_mode) or self._infer_input_mode_from_caller()
        parts = [model, prompt_hash, input_hash]
        if effective_input_mode:
            parts.append(effective_input_mode)
        if variant_id:
            parts.append(variant_id)
        return ":".join(parts)

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
