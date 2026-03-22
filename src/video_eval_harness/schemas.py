"""Core Pydantic schemas for the video evaluation harness."""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class _NoModelNs(BaseModel):
    """Base with relaxed protected namespaces for fields like model_name."""
    model_config = ConfigDict(protected_namespaces=())


class SegmentationMode(str, Enum):
    FIXED_WINDOW = "fixed_window"
    SCENE_HEURISTIC = "scene_heuristic"


class VideoMetadata(BaseModel):
    """Metadata extracted from a video file."""

    video_id: str
    source_path: str
    filename: str
    duration_s: float
    width: int
    height: int
    fps: float
    codec: Optional[str] = None
    file_size_bytes: Optional[int] = None
    ingested_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


class Segment(BaseModel):
    """A temporal segment of a video."""

    segment_id: str
    video_id: str
    segment_index: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    segmentation_mode: SegmentationMode
    segmentation_config: dict = Field(default_factory=dict)


class ExtractedFrames(BaseModel):
    """References to frames extracted from a segment."""

    segment_id: str
    video_id: str
    frame_paths: list[str] = Field(default_factory=list)
    frame_timestamps_s: list[float] = Field(default_factory=list)
    contact_sheet_path: Optional[str] = None
    num_frames: int = 0


class SegmentLabelResult(_NoModelNs):
    """Normalized label result from a model for a single segment."""

    run_id: str
    video_id: str
    segment_id: str
    start_time_s: float
    end_time_s: float
    model_name: str
    provider: str

    # Label fields
    primary_action: Optional[str] = None
    secondary_actions: list[str] = Field(default_factory=list)
    description: Optional[str] = None
    objects: list[str] = Field(default_factory=list)
    environment_context: Optional[str] = None
    confidence: Optional[float] = None
    reasoning_summary_or_notes: Optional[str] = None
    uncertainty_flags: list[str] = Field(default_factory=list)

    # Extraction sweep fields (optional, empty for non-sweep runs)
    extraction_variant_id: str = ""
    extraction_label: str = ""
    num_frames_used: int = 0
    sampling_method_used: str = ""
    sweep_id: str = ""

    # Input mode: "frames" (extracted images) or "video" (raw video segment)
    input_mode: str = "frames"

    # Meta
    raw_response_text: Optional[str] = None
    parsed_success: bool = False
    parse_error: Optional[str] = None
    latency_ms: Optional[float] = None
    estimated_cost: Optional[float] = None
    prompt_version: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())


class GroundTruthLabel(BaseModel):
    """Ground truth label for a segment, if available."""

    video_id: str
    segment_id: str
    start_time_s: float
    end_time_s: float
    primary_action: str
    secondary_actions: list[str] = Field(default_factory=list)
    description: Optional[str] = None
    source: Optional[str] = None


class RunConfig(BaseModel):
    """Snapshot of configuration for a benchmark run."""

    run_id: str
    created_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    models: list[str] = Field(default_factory=list)
    prompt_version: str = "default"
    segmentation_mode: SegmentationMode = SegmentationMode.FIXED_WINDOW
    segmentation_config: dict = Field(default_factory=dict)
    extraction_config: dict = Field(default_factory=dict)
    video_ids: list[str] = Field(default_factory=list)
    notes: Optional[str] = None
    display_name: Optional[str] = None


class ModelRunSummary(_NoModelNs):
    """Summary statistics for a single model within a run."""

    model_name: str
    total_segments: int = 0
    successful_parses: int = 0
    failed_parses: int = 0
    parse_success_rate: float = 0.0
    avg_latency_ms: Optional[float] = None
    median_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    total_estimated_cost: Optional[float] = None
    avg_confidence: Optional[float] = None
