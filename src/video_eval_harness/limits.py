"""Server-side cost controls for public benchmark API mode."""

from __future__ import annotations

from .log import get_logger

logger = get_logger(__name__)

PUBLIC_LIMITS = {
    "max_clip_duration_s": 60,
    "max_file_size_mb": 100,
    "max_segments": 6,
    "max_models": 7,
    "allowed_models": [
        # Fast tier
        "gemini-3-flash",
        "gpt-5.4-mini",
        "qwen3.5-27b",
        # Frontier tier
        "gemini-3.1-pro",
        "gpt-5.4",
        "qwen3.5-vl",
        "llama-4-maverick",
    ],
    "max_frames_per_segment": 8,
    "extraction_methods": ["uniform"],  # no keyframe for public (needs OpenCV)
    "max_concurrent_jobs": 3,
}


def validate_public_request(
    file_size_bytes: int,
    duration_s: float,
    requested_models: list[str],
    num_frames: int = 8,
) -> tuple[bool, str | None]:
    """Validate an upload against public limits.

    Returns (is_valid, error_message_or_none).
    """
    if file_size_bytes > PUBLIC_LIMITS["max_file_size_mb"] * 1024 * 1024:
        return False, f"File too large. Max {PUBLIC_LIMITS['max_file_size_mb']}MB."
    if duration_s > PUBLIC_LIMITS["max_clip_duration_s"]:
        return False, f"Clip too long. Max {PUBLIC_LIMITS['max_clip_duration_s']}s."
    disallowed = [m for m in requested_models if m not in PUBLIC_LIMITS["allowed_models"]]
    if disallowed:
        return False, f"Models not available for public use: {disallowed}"
    if len(requested_models) > PUBLIC_LIMITS["max_models"]:
        return False, f"Max {PUBLIC_LIMITS['max_models']} models per run."
    if num_frames > PUBLIC_LIMITS["max_frames_per_segment"]:
        return False, f"Max {PUBLIC_LIMITS['max_frames_per_segment']} frames per segment."
    return True, None


def clamp_public_config(config: dict) -> dict:
    """Force public-safe values onto a config dict before running."""
    config["max_segments"] = min(config.get("max_segments", 6), PUBLIC_LIMITS["max_segments"])
    config["num_frames"] = min(config.get("num_frames", 8), PUBLIC_LIMITS["max_frames_per_segment"])
    config["methods"] = ["uniform"]
    return config
