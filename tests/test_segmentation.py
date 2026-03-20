"""Tests for segmentation logic."""

import pytest
from video_eval_harness.config import SegmentationConfig
from video_eval_harness.schemas import SegmentationMode, VideoMetadata
from video_eval_harness.segmentation import FixedWindowSegmenter


def _make_video(duration: float) -> VideoMetadata:
    return VideoMetadata(
        video_id="vid_test_abc123",
        source_path="/test/video.mp4",
        filename="video.mp4",
        duration_s=duration,
        width=1920,
        height=1080,
        fps=30.0,
    )


def test_basic_segmentation():
    """10s window on 30s video = 3 segments."""
    cfg = SegmentationConfig(window_size_s=10.0)
    seg = FixedWindowSegmenter(cfg)
    video = _make_video(30.0)
    segments = seg.segment(video)

    assert len(segments) == 3
    assert segments[0].start_time_s == 0.0
    assert segments[0].end_time_s == 10.0
    assert segments[1].start_time_s == 10.0
    assert segments[2].end_time_s == 30.0


def test_segmentation_mode():
    cfg = SegmentationConfig(window_size_s=10.0)
    seg = FixedWindowSegmenter(cfg)
    video = _make_video(20.0)
    segments = seg.segment(video)
    assert all(s.segmentation_mode == SegmentationMode.FIXED_WINDOW for s in segments)


def test_partial_last_segment():
    """25s video with 10s window: last segment is 5s."""
    cfg = SegmentationConfig(window_size_s=10.0, min_segment_s=2.0)
    seg = FixedWindowSegmenter(cfg)
    video = _make_video(25.0)
    segments = seg.segment(video)

    assert len(segments) == 3
    assert segments[-1].duration_s == 5.0


def test_short_final_segment_dropped():
    """If final segment is shorter than min_segment_s, it should be dropped."""
    cfg = SegmentationConfig(window_size_s=10.0, min_segment_s=3.0)
    seg = FixedWindowSegmenter(cfg)
    video = _make_video(21.0)
    segments = seg.segment(video)

    # 0-10, 10-20, 20-21 (1s < 3s min → dropped)
    assert len(segments) == 2


def test_overlap_segmentation():
    """With 50% overlap: stride = 5, window = 10 on 20s video."""
    cfg = SegmentationConfig(window_size_s=10.0, stride_s=5.0)
    seg = FixedWindowSegmenter(cfg)
    video = _make_video(20.0)
    segments = seg.segment(video)

    # 0-10, 5-15, 10-20, 15-20
    assert len(segments) >= 3
    assert segments[0].start_time_s == 0.0
    assert segments[1].start_time_s == 5.0


def test_segment_ids_unique():
    cfg = SegmentationConfig(window_size_s=5.0)
    seg = FixedWindowSegmenter(cfg)
    video = _make_video(30.0)
    segments = seg.segment(video)

    ids = [s.segment_id for s in segments]
    assert len(ids) == len(set(ids))


def test_segment_indices_sequential():
    cfg = SegmentationConfig(window_size_s=5.0)
    seg = FixedWindowSegmenter(cfg)
    video = _make_video(30.0)
    segments = seg.segment(video)

    for i, s in enumerate(segments):
        assert s.segment_index == i


def test_segment_config_preserved():
    cfg = SegmentationConfig(window_size_s=10.0, stride_s=5.0)
    seg = FixedWindowSegmenter(cfg)
    video = _make_video(30.0)
    segments = seg.segment(video)

    for s in segments:
        assert s.segmentation_config["window_size_s"] == 10.0
        assert s.segmentation_config["stride_s"] == 5.0


def test_zero_duration_video():
    cfg = SegmentationConfig(window_size_s=10.0)
    seg = FixedWindowSegmenter(cfg)
    video = _make_video(0.0)
    segments = seg.segment(video)
    assert len(segments) == 0


def test_very_short_video():
    """Video shorter than one window."""
    cfg = SegmentationConfig(window_size_s=10.0, min_segment_s=1.0)
    seg = FixedWindowSegmenter(cfg)
    video = _make_video(3.0)
    segments = seg.segment(video)
    assert len(segments) == 1
    assert segments[0].duration_s == 3.0
