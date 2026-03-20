"""Base segmenter abstraction."""

from __future__ import annotations

import abc

from ..schemas import Segment, VideoMetadata


class BaseSegmenter(abc.ABC):
    """Abstract base class for video segmenters."""

    @abc.abstractmethod
    def segment(self, video: VideoMetadata) -> list[Segment]:
        """Divide a video into temporal segments.

        Args:
            video: Metadata of the video to segment.

        Returns:
            List of Segment objects.
        """
        ...
