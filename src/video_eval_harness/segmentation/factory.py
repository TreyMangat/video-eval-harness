"""Helpers for constructing segmenters from config."""

from __future__ import annotations

from ..config import SegmentationConfig
from .base import BaseSegmenter
from .fixed_window import FixedWindowSegmenter


def build_segmenter(config: SegmentationConfig) -> BaseSegmenter:
    """Create the configured segmenter instance."""
    mode = (config.mode or "fixed_window").strip().lower()

    if mode == "fixed_window":
        return FixedWindowSegmenter(config)

    if mode == "scene_heuristic":
        from .scene_heuristic import SceneHeuristicSegmenter

        return SceneHeuristicSegmenter(config)

    raise ValueError(
        f"Unknown segmentation mode '{config.mode}'. Expected one of: fixed_window, scene_heuristic"
    )
