"""Base adapter interface for video data sources."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VideoEntry:
    """A video entry from a data source."""

    path: Path
    video_id: Optional[str] = None
    metadata: Optional[dict] = None


class BaseAdapter(abc.ABC):
    """Abstract base class for dataset/source adapters."""

    @abc.abstractmethod
    def list_videos(self) -> list[VideoEntry]:
        """Return a list of available video entries."""
        ...

    @abc.abstractmethod
    def name(self) -> str:
        """Return the adapter name."""
        ...
