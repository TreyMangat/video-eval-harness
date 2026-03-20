"""Tests for extraction helpers and variant-aware frame caching."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from video_eval_harness.config import ExtractionConfig
from video_eval_harness.extraction.frames import FrameExtractor
from video_eval_harness.schemas import Segment, SegmentationMode
from video_eval_harness.storage import Storage


def make_segment() -> Segment:
    return Segment(
        segment_id="vid_test_seg0000",
        video_id="vid_test",
        segment_index=0,
        start_time_s=10.0,
        end_time_s=20.0,
        duration_s=10.0,
        segmentation_mode=SegmentationMode.FIXED_WINDOW,
        segmentation_config={},
    )


def test_uniform_timestamps_count_and_range(tmp_path):
    extractor = FrameExtractor(ExtractionConfig(num_frames=4), Storage(str(tmp_path / "artifacts")))
    timestamps = extractor._uniform_timestamps(make_segment(), 4)

    assert len(timestamps) == 4
    assert timestamps == sorted(timestamps)
    assert all(10.0 <= ts <= 20.0 for ts in timestamps)
    assert timestamps[0] > 10.0
    assert timestamps[-1] < 20.0


def test_default_variant_id_format(tmp_path):
    extractor = FrameExtractor(ExtractionConfig(num_frames=4), Storage(str(tmp_path / "artifacts")))

    assert extractor._default_variant_id(16, "keyframe") == "keyframe_16f"


def test_frame_cache_per_variant(tmp_path):
    storage = Storage(str(tmp_path / "artifacts"))
    extractor = FrameExtractor(ExtractionConfig(num_frames=2), storage)
    segment = make_segment()

    def fake_extract_frame_at_time(video_path, timestamp_s, output_path, max_dimension, quality):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake-frame")
        return Path(output_path)

    with patch(
        "video_eval_harness.extraction.frames.extract_frame_at_time",
        side_effect=fake_extract_frame_at_time,
    ) as mocked_extract:
        first = extractor.extract_frames(segment, "video.mp4", num_frames=2, method="uniform")
        second = extractor.extract_frames(segment, "video.mp4", num_frames=2, method="uniform")
        third = extractor.extract_frames(segment, "video.mp4", num_frames=4, method="uniform")

    assert mocked_extract.call_count == 6
    assert "/uniform_2f/" in first.frame_paths[0].replace("\\", "/")
    assert second.frame_paths == first.frame_paths
    assert "/uniform_4f/" in third.frame_paths[0].replace("\\", "/")
