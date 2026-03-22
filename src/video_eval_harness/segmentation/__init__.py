from .base import BaseSegmenter
from .fixed_window import FixedWindowSegmenter

# SceneHeuristicSegmenter requires cv2 which is an optional dependency.
# Lazy-import to avoid breaking the CLI when OpenCV isn't installed.
try:
    from .scene_heuristic import SceneHeuristicSegmenter
except ImportError:
    SceneHeuristicSegmenter = None  # type: ignore[assignment,misc]

__all__ = ["BaseSegmenter", "FixedWindowSegmenter", "SceneHeuristicSegmenter"]
