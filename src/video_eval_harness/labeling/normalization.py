"""Parse and normalize model responses into structured labels."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from ..log import get_logger
from ..schemas import SegmentLabelResult

logger = get_logger(__name__)


def extract_json_from_text(text: str) -> Optional[dict]:
    """Extract JSON object from model response text.

    Handles responses wrapped in markdown code blocks, extra text before/after, etc.
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # Try direct JSON parse first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if md_match:
        try:
            obj = json.loads(md_match.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            obj = json.loads(brace_match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return None


def normalize_string_list(val: Any) -> list[str]:
    """Normalize a value into a list of strings."""
    if val is None:
        return []
    if isinstance(val, str):
        return [val] if val.strip() else []
    if isinstance(val, list):
        return [str(v) for v in val if v]
    return [str(val)]


def normalize_confidence(val: Any) -> Optional[float]:
    """Normalize confidence to 0-1 float."""
    if val is None:
        return None
    try:
        f = float(val)
        # Only treat as percentage if clearly looks like 0-100 scale (>= 10)
        if f >= 10.0 and f <= 100.0:
            f = f / 100.0
        return max(0.0, min(1.0, f))
    except (ValueError, TypeError):
        return None


def parse_model_response(
    raw_text: str,
    run_id: str,
    video_id: str,
    segment_id: str,
    start_time_s: float,
    end_time_s: float,
    model_name: str,
    provider: str,
    latency_ms: float,
    estimated_cost: Optional[float] = None,
    prompt_version: Optional[str] = None,
    error: Optional[str] = None,
    **extra_fields,
) -> SegmentLabelResult:
    """Parse a raw model response into a normalized SegmentLabelResult.

    Extra keyword arguments (e.g. sweep fields like extraction_variant_id,
    extraction_label, etc.) are forwarded to the SegmentLabelResult constructor.
    """
    common = dict(
        run_id=run_id,
        video_id=video_id,
        segment_id=segment_id,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        model_name=model_name,
        provider=provider,
        latency_ms=latency_ms,
        estimated_cost=estimated_cost,
        prompt_version=prompt_version,
        **extra_fields,
    )

    if error:
        return SegmentLabelResult(
            **common,
            raw_response_text=raw_text,
            parsed_success=False,
            parse_error=f"Provider error: {error}",
        )

    parsed = extract_json_from_text(raw_text)

    if parsed is None:
        return SegmentLabelResult(
            **common,
            raw_response_text=raw_text,
            parsed_success=False,
            parse_error="Could not extract JSON from response",
        )

    try:
        return SegmentLabelResult(
            **common,
            primary_action=parsed.get("primary_action"),
            secondary_actions=normalize_string_list(parsed.get("secondary_actions")),
            description=parsed.get("description"),
            objects=normalize_string_list(parsed.get("objects")),
            environment_context=parsed.get("environment_context"),
            confidence=normalize_confidence(parsed.get("confidence")),
            reasoning_summary_or_notes=parsed.get("reasoning_summary_or_notes"),
            uncertainty_flags=normalize_string_list(parsed.get("uncertainty_flags")),
            raw_response_text=raw_text,
            parsed_success=True,
        )
    except Exception as e:
        return SegmentLabelResult(
            **common,
            raw_response_text=raw_text,
            parsed_success=False,
            parse_error=f"Validation error: {e}",
        )
