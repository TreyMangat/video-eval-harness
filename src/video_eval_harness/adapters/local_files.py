"""Adapter for individual local video files."""

from __future__ import annotations

from pathlib import Path

from .dataset_base import BaseAdapter, VideoEntry


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


class LocalFileAdapter(BaseAdapter):
    """Adapter for a single local video file or list of files."""

    def __init__(self, paths: list[str | Path]):
        self.paths = [Path(p) for p in paths]

    def list_videos(self) -> list[VideoEntry]:
        entries = []
        for p in self.paths:
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                entries.append(VideoEntry(path=p))
            elif not p.exists():
                raise FileNotFoundError(f"Video file not found: {p}")
        return entries

    def name(self) -> str:
        return "local_files"
