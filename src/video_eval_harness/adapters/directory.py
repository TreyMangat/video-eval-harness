"""Adapter for a directory of video files."""

from __future__ import annotations

from pathlib import Path

from .dataset_base import BaseAdapter, VideoEntry
from .local_files import VIDEO_EXTENSIONS


class DirectoryAdapter(BaseAdapter):
    """Adapter that scans a directory for video files."""

    def __init__(self, directory: str | Path, recursive: bool = True):
        self.directory = Path(directory)
        self.recursive = recursive
        if not self.directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {self.directory}")

    def list_videos(self) -> list[VideoEntry]:
        pattern = "**/*" if self.recursive else "*"
        entries = []
        for p in sorted(self.directory.glob(pattern)):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                entries.append(VideoEntry(path=p))
        return entries

    def name(self) -> str:
        return "directory"
