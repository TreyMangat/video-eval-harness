"""ID generation utilities."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path


def generate_run_id() -> str:
    """Generate a unique run ID."""
    return f"run_{uuid.uuid4().hex[:12]}"


def generate_video_id(path: str | Path) -> str:
    """Generate a deterministic video ID from file path and size."""
    p = Path(path)
    stat = p.stat()
    content = f"{p.name}:{stat.st_size}:{stat.st_mtime}"
    h = hashlib.sha256(content.encode()).hexdigest()[:12]
    stem = p.stem[:40].replace(" ", "_")
    return f"vid_{stem}_{h}"


def generate_segment_id(video_id: str, segment_index: int) -> str:
    """Generate a deterministic segment ID."""
    return f"{video_id}_seg{segment_index:04d}"
