"""Tests for Pydantic schemas."""

import pytest
from video_eval_harness.schemas import (
    ExtractedFrames,
    GroundTruthLabel,
    ModelRunSummary,
    RunConfig,
    Segment,
    SegmentLabelResult,
    SegmentationMode,
    VideoMetadata,
)


def test_video_metadata_creation():
    meta = VideoMetadata(
        video_id="vid_test_abc123",
        source_path="/path/to/video.mp4",
        filename="video.mp4",
        duration_s=120.5,
        width=1920,
        height=1080,
        fps=30.0,
    )
    assert meta.video_id == "vid_test_abc123"
    assert meta.duration_s == 120.5
    assert meta.ingested_at is not None


def test_segment_creation():
    seg = Segment(
        segment_id="vid_test_seg0000",
        video_id="vid_test",
        segment_index=0,
        start_time_s=0.0,
        end_time_s=10.0,
        duration_s=10.0,
        segmentation_mode=SegmentationMode.FIXED_WINDOW,
    )
    assert seg.segment_index == 0
    assert seg.duration_s == 10.0
    assert seg.segmentation_mode == SegmentationMode.FIXED_WINDOW


def test_segment_label_result_defaults():
    result = SegmentLabelResult(
        run_id="run_abc",
        video_id="vid_test",
        segment_id="vid_test_seg0000",
        start_time_s=0.0,
        end_time_s=10.0,
        model_name="gpt4o",
        provider="openrouter",
    )
    assert result.parsed_success is False
    assert result.secondary_actions == []
    assert result.objects == []
    assert result.uncertainty_flags == []
    assert result.timestamp is not None


def test_segment_label_result_full():
    result = SegmentLabelResult(
        run_id="run_abc",
        video_id="vid_test",
        segment_id="vid_test_seg0000",
        start_time_s=0.0,
        end_time_s=10.0,
        model_name="gpt4o",
        provider="openrouter",
        primary_action="cooking",
        secondary_actions=["stirring", "seasoning"],
        description="Person is cooking in a kitchen",
        objects=["pan", "spatula"],
        environment_context="indoor kitchen",
        confidence=0.92,
        parsed_success=True,
        latency_ms=1500.0,
    )
    assert result.primary_action == "cooking"
    assert len(result.secondary_actions) == 2
    assert result.confidence == 0.92


def test_ground_truth_label():
    gt = GroundTruthLabel(
        video_id="vid_test",
        segment_id="vid_test_seg0000",
        start_time_s=0.0,
        end_time_s=10.0,
        primary_action="cooking",
        source="manual",
    )
    assert gt.primary_action == "cooking"


def test_run_config():
    rc = RunConfig(
        run_id="run_abc",
        models=["gpt4o", "gemini"],
        prompt_version="concise",
    )
    assert len(rc.models) == 2
    assert rc.created_at is not None


def test_model_run_summary():
    s = ModelRunSummary(
        model_name="gpt4o",
        total_segments=100,
        successful_parses=95,
        failed_parses=5,
        parse_success_rate=0.95,
        avg_latency_ms=1200.0,
    )
    assert s.parse_success_rate == 0.95


def test_extracted_frames():
    ef = ExtractedFrames(
        segment_id="vid_test_seg0000",
        video_id="vid_test",
        frame_paths=["/a/1.jpg", "/a/2.jpg"],
        frame_timestamps_s=[1.0, 5.0],
        num_frames=2,
    )
    assert ef.num_frames == 2
    assert len(ef.frame_paths) == 2
