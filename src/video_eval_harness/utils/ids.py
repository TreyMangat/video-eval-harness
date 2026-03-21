"""ID generation utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path


def generate_run_id(
    video_names: list[str] | None = None,
    is_sweep: bool = False,
    name: str | None = None,
) -> str:
    """Generate a human-readable run ID.

    Format: ``run_{date}_{hint}_{short_hash}``

    Examples::

        run_20260320_cooking30s_a7f4        (single video)
        run_20260320_6videos_b8e2           (directory with multiple videos)
        run_20260320_factory_sweep_c3d1     (sweep run)
        run_20260320_my-experiment_e5a9     (custom --name)

    The old no-args signature still works for backwards compatibility,
    producing ``run_{date}_benchmark_{hash}``.
    """
    import re
    import time
    from datetime import datetime, timezone

    date = datetime.now(timezone.utc).strftime("%Y%m%d")

    # Hint: --name override, or auto-generated from video names
    if name:
        hint = re.sub(r"[^a-z0-9_-]", "", name.lower().replace(" ", "-"))[:40]
    elif video_names and len(video_names) == 1:
        stem = Path(video_names[0]).stem[:20].lower().replace(" ", "_")
        hint = re.sub(r"[^a-z0-9_]", "", stem)
    elif video_names and len(video_names) > 1:
        hint = f"{len(video_names)}videos"
    else:
        hint = "benchmark"

    if is_sweep:
        hint = f"{hint}_sweep"

    short_hash = hashlib.sha256(f"{time.time_ns()}".encode()).hexdigest()[:4]
    return f"run_{date}_{hint}_{short_hash}"


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
