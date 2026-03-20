"""Scene/shot-boundary heuristic segmentation using frame difference detection."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..config import SegmentationConfig
from ..log import get_logger
from ..schemas import Segment, SegmentationMode, VideoMetadata
from ..utils.ids import generate_segment_id
from .base import BaseSegmenter

logger = get_logger(__name__)


class SceneHeuristicSegmenter(BaseSegmenter):
    """Segment video at scene/shot boundaries using frame-difference heuristics.

    Uses histogram difference between consecutive frames to detect scene cuts.
    Falls back to fixed-window segmentation if OpenCV cannot read the video.
    """

    def __init__(
        self,
        config: SegmentationConfig,
        threshold: float = 0.4,
        min_scene_s: float = 2.0,
        max_scene_s: float = 60.0,
        sample_fps: float = 2.0,
    ):
        self.config = config
        self.threshold = threshold
        self.min_scene_s = min_scene_s
        self.max_scene_s = max_scene_s
        self.sample_fps = sample_fps

    def segment(self, video: VideoMetadata) -> list[Segment]:
        """Detect scene boundaries and create segments."""
        path = Path(video.source_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")

        try:
            boundaries = self._detect_boundaries(str(path), video.duration_s, video.fps)
        except Exception as e:
            logger.warning(f"Scene detection failed for {video.video_id}, falling back to fixed window: {e}")
            from .fixed_window import FixedWindowSegmenter
            return FixedWindowSegmenter(self.config).segment(video)

        return self._boundaries_to_segments(video, boundaries)

    def _detect_boundaries(
        self, video_path: str, duration_s: float, fps: float
    ) -> list[float]:
        """Detect scene boundaries by comparing frame histograms.

        Returns list of boundary timestamps (in seconds).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            # Sample at reduced FPS for efficiency
            frame_interval = max(1, int(fps / self.sample_fps))
            boundaries: list[float] = [0.0]  # Always start at 0
            prev_hist: Optional[np.ndarray] = None
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    timestamp_s = frame_idx / fps

                    # Compute histogram for the frame (grayscale)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
                    cv2.normalize(hist, hist)

                    if prev_hist is not None:
                        # Compare histograms using correlation
                        similarity = cv2.compareHist(
                            prev_hist, hist, cv2.HISTCMP_CORREL
                        )
                        diff = 1.0 - similarity

                        if diff > self.threshold:
                            # Check minimum scene length
                            time_since_last = timestamp_s - boundaries[-1]
                            if time_since_last >= self.min_scene_s:
                                boundaries.append(timestamp_s)

                    prev_hist = hist

                frame_idx += 1

            # Add final boundary at video end
            boundaries.append(duration_s)

            # Enforce max scene length by splitting long scenes
            boundaries = self._enforce_max_length(boundaries)

            return boundaries

        finally:
            cap.release()

    def _enforce_max_length(self, boundaries: list[float]) -> list[float]:
        """Split scenes that exceed max_scene_s."""
        result: list[float] = [boundaries[0]]
        for i in range(1, len(boundaries)):
            scene_len = boundaries[i] - result[-1]
            if scene_len > self.max_scene_s:
                # Split into roughly equal sub-segments
                n_splits = int(scene_len / self.max_scene_s) + 1
                sub_len = scene_len / n_splits
                for j in range(1, n_splits):
                    result.append(result[-1] + sub_len)
            result.append(boundaries[i])
        return result

    def _boundaries_to_segments(
        self, video: VideoMetadata, boundaries: list[float]
    ) -> list[Segment]:
        """Convert boundary timestamps to Segment objects."""
        segments: list[Segment] = []
        for idx in range(len(boundaries) - 1):
            start = round(boundaries[idx], 3)
            end = round(boundaries[idx + 1], 3)
            duration = round(end - start, 3)

            # Skip very short segments
            if duration < self.config.min_segment_s and idx > 0:
                continue

            segments.append(
                Segment(
                    segment_id=generate_segment_id(video.video_id, idx),
                    video_id=video.video_id,
                    segment_index=idx,
                    start_time_s=start,
                    end_time_s=end,
                    duration_s=duration,
                    segmentation_mode=SegmentationMode.SCENE_HEURISTIC,
                    segmentation_config={
                        "threshold": self.threshold,
                        "min_scene_s": self.min_scene_s,
                        "max_scene_s": self.max_scene_s,
                        "sample_fps": self.sample_fps,
                    },
                )
            )

        return segments
