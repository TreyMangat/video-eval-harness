"""FFmpeg wrapper utilities for video metadata and frame extraction."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..log import get_logger

logger = get_logger(__name__)


@dataclass
class VideoInfo:
    """Raw video information from ffprobe."""

    duration_s: float
    width: int
    height: int
    fps: float
    codec: Optional[str]
    file_size_bytes: int


_ffmpeg_resolved: Optional[str] = None


def get_ffmpeg_path() -> str:
    """Get ffmpeg path from env, PATH, or auto-discover on Windows."""
    import os
    import shutil

    global _ffmpeg_resolved
    if _ffmpeg_resolved is not None:
        return _ffmpeg_resolved

    # 1. Explicit env var
    env_path = os.environ.get("FFMPEG_PATH")
    if env_path and Path(env_path).exists():
        _ffmpeg_resolved = env_path
        return env_path

    # 2. On PATH
    found = shutil.which("ffmpeg")
    if found:
        _ffmpeg_resolved = found
        return found

    # 3. Auto-discover common Windows install locations
    if os.name == "nt":
        search_dirs = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages",
            Path("C:/ffmpeg/bin"),
            Path("C:/Program Files/ffmpeg/bin"),
            Path(os.environ.get("PROGRAMDATA", "")) / "chocolatey" / "lib" / "ffmpeg" / "tools" / "ffmpeg" / "bin",
        ]
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            # Search recursively for ffmpeg.exe
            for candidate in search_dir.rglob("ffmpeg.exe"):
                _ffmpeg_resolved = str(candidate)
                logger.info(f"Auto-discovered ffmpeg at: {_ffmpeg_resolved}")
                return _ffmpeg_resolved

    _ffmpeg_resolved = "ffmpeg"  # fallback
    return "ffmpeg"


def get_ffprobe_path() -> str:
    """Get ffprobe path (derived from ffmpeg path)."""
    ffmpeg = get_ffmpeg_path()
    # Replace the last occurrence of 'ffmpeg' with 'ffprobe' to handle paths
    if "ffmpeg" in Path(ffmpeg).name:
        return str(Path(ffmpeg).parent / Path(ffmpeg).name.replace("ffmpeg", "ffprobe"))
    return "ffprobe"


def probe_video(video_path: str | Path) -> VideoInfo:
    """Extract video metadata using ffprobe."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cmd = [
        get_ffprobe_path(),
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
    except FileNotFoundError:
        raise RuntimeError(
            "ffprobe not found. Install ffmpeg: https://ffmpeg.org/download.html"
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffprobe timed out on {path}")

    data = json.loads(result.stdout)

    # Find the video stream
    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise ValueError(f"No video stream found in {path}")

    # Parse fps from r_frame_rate (e.g., "30/1")
    fps_str = video_stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) > 0 else 30.0
    else:
        fps = float(fps_str)

    # Duration: prefer format duration, fall back to stream
    duration = float(data.get("format", {}).get("duration", 0))
    if duration == 0:
        duration = float(video_stream.get("duration", 0))

    return VideoInfo(
        duration_s=duration,
        width=int(video_stream.get("width", 0)),
        height=int(video_stream.get("height", 0)),
        fps=fps,
        codec=video_stream.get("codec_name"),
        file_size_bytes=path.stat().st_size,
    )


def extract_frame_at_time(
    video_path: str | Path,
    timestamp_s: float,
    output_path: str | Path,
    max_dimension: int = 1280,
    quality: int = 85,
) -> Path:
    """Extract a single frame from a video at the given timestamp."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Build scale filter that caps max dimension while preserving aspect ratio
    scale_filter = (
        f"scale='if(gt(iw,ih),min({max_dimension},iw),-2)'"
        f":'{f'if(gt(ih,iw),min({max_dimension},ih),-2)'}'"
    )
    # Simpler approach: scale down if larger
    scale_filter = f"scale=min({max_dimension}\\,iw):min({max_dimension}\\,ih):force_original_aspect_ratio=decrease"

    cmd = [
        get_ffmpeg_path(),
        "-ss", str(timestamp_s),
        "-i", str(video_path),
        "-vframes", "1",
        "-vf", scale_filter,
        "-q:v", str(max(1, min(31, (100 - quality) * 31 // 100))),
        "-y",
        str(output),
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=30)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Install ffmpeg: https://ffmpeg.org/download.html")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Frame extraction failed at t={timestamp_s}s: {e.stderr[:200] if e.stderr else ''}")
        raise

    return output
