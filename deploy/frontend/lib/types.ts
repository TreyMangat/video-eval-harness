export type ModelSummary = {
  model_name: string;
  total_segments: number;
  successful_parses: number;
  failed_parses: number;
  parse_success_rate: number;
  avg_latency_ms: number | null;
  median_latency_ms: number | null;
  p95_latency_ms: number | null;
  total_estimated_cost: number | null;
  avg_confidence: number | null;
  consensus_alignment_rate: number | null;
};

export type LabelResult = {
  run_id: string;
  video_id: string;
  segment_id: string;
  start_time_s: number;
  end_time_s: number;
  model_name: string;
  provider: string;
  primary_action: string | null;
  secondary_actions: string[];
  description: string | null;
  objects: string[];
  environment_context: string | null;
  confidence: number | null;
  reasoning_summary_or_notes: string | null;
  uncertainty_flags: string[];
  parsed_success: boolean;
  parse_error: string | null;
  latency_ms: number | null;
  estimated_cost: number | null;
  prompt_version: string | null;
};

export type SegmentSummary = {
  segment_id: string;
  video_id: string;
  video_filename: string | null;
  segment_index: number;
  start_time_s: number;
  end_time_s: number;
  duration_s: number;
  segmentation_mode: string;
  frame_count: number;
  frame_timestamps_s: number[];
  has_contact_sheet: boolean;
};

export type RunPayload = {
  run_id: string;
  config: {
    models: string[];
    prompt_version: string;
    segmentation_mode: string;
    segmentation_config: Record<string, unknown>;
    extraction_config: Record<string, unknown>;
    model_configs: Record<string, { role?: string; notes?: string }>;
    video_ids: string[];
    created_at: string;
  };
  models: string[];
  videos: Array<{
    filename?: string;
    video_id?: string;
    source_path?: string;
    duration_s?: number;
    width?: number;
    height?: number;
    fps?: number;
    codec?: string;
    file_size_bytes?: number;
    ingested_at?: string;
  }>;
  summaries: Record<string, ModelSummary>;
  agreement: Record<string, Record<string, number>>;
  segments: SegmentSummary[];
  results: LabelResult[];
};

export type FramePreview = {
  timestamp_s: number;
  data_url: string | null;
};

export type SegmentMedia = {
  run_id: string;
  segment_id: string;
  start_time_s: number;
  end_time_s: number;
  frame_timestamps_s: number[];
  contact_sheet_data_url: string | null;
  frames: FramePreview[];
};

export type RunListItem = {
  run_id: string;
  created_at: string;
  models: string[];
  prompt_version: string;
  video_ids: string[];
};

export type ModelCatalogItem = {
  name: string;
  model_id: string;
  provider: string;
  supports_images: boolean;
  role: string;
  notes: string;
};
