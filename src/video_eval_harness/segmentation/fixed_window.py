"""Fixed-window segmentation strategy."""

from __future__ import annotations

from ..config import SegmentationConfig
from ..schemas import Segment, SegmentationMode, VideoMetadata
from ..utils.ids import generate_segment_id
from .base import BaseSegmenter


class FixedWindowSegmenter(BaseSegmenter):
    """Segment video into fixed-duration windows with optional overlap."""

    def __init__(self, config: SegmentationConfig):
        self.window_size_s = config.window_size_s
        self.stride_s = config.stride_s if config.stride_s is not None else config.window_size_s
        self.min_segment_s = config.min_segment_s

    def segment(self, video: VideoMetadata) -> list[Segment]:
        """Create fixed-window segments over the video duration."""
        segments: list[Segment] = []
        duration = video.duration_s

        if duration <= 0:
            return segments

        start = 0.0
        idx = 0
        while start < duration:
            end = min(start + self.window_size_s, duration)
            seg_duration = end - start

            # Skip very short final segments
            if seg_duration < self.min_segment_s and idx > 0:
                break

            segment_id = generate_segment_id(video.video_id, idx)
            segments.append(
                Segment(
                    segment_id=segment_id,
                    video_id=video.video_id,
                    segment_index=idx,
                    start_time_s=round(start, 3),
                    end_time_s=round(end, 3),
                    duration_s=round(seg_duration, 3),
                    segmentation_mode=SegmentationMode.FIXED_WINDOW,
                    segmentation_config={
                        "window_size_s": self.window_size_s,
                        "stride_s": self.stride_s,
                        "min_segment_s": self.min_segment_s,
                    },
                )
            )

            start += self.stride_s
            idx += 1

        return segments
