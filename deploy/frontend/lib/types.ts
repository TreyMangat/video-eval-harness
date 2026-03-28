export type ModelSummary = {
  model_name: string;
  total_segments: number;
  successful_parses: number;
  failed_parses: number;
  parse_success_rate: number;
  accuracy?: number | null;
  exact_match_rate?: number | null;
  fuzzy_match_rate?: number | null;
  llm_accuracy?: number | null;
  avg_latency_ms: number | null;
  median_latency_ms: number | null;
  p95_latency_ms: number | null;
  total_estimated_cost: number | null;
  avg_confidence: number | null;
  consensus_alignment_rate: number | null;
  input_mode?: string | null;
};

export type AccuracyMetric = {
  accuracy?: number | null;
  exact_match_rate?: number | null;
  fuzzy_match_rate?: number | null;
  llm_accuracy?: number | null;
  mean_similarity?: number | null;
  evaluated_segments?: number | null;
};

export type ActionLabel = {
  verb: string;
  noun: string;
  verb_class?: number | null;
  noun_class?: number | null;
  action: string;
  confidence?: number | null;
};

export type GroundTruthEntry = {
  video_id?: string;
  segment_id?: string;
  segment_index?: number;
  start_time_s?: number;
  end_time_s?: number;
  primary_action?: string | null;
  label?: string | null;
  description?: string | null;
  source?: string | null;
};

export type ConsensusEntry = {
  segment_id?: string | null;
  video_id?: string | null;
  start_s?: number | null;
  end_s?: number | null;
  start_time_s?: number | null;
  end_time_s?: number | null;
  consensus_action?: string | null;
  label?: string | null;
  method?: "unanimous" | "majority" | "highest_confidence" | string | null;
  agreement?: number | null;
  agreement_ratio?: number | null;
  votes?: Record<string, number> | null;
  action_verb?: string | null;
  action_noun?: string | null;
};

export type ConsensusSummary = {
  total_segments?: number | null;
  unanimous_count?: number | null;
  majority_count?: number | null;
  mean_agreement?: number | null;
  unanimous_rate?: number | null;
  majority_rate?: number | null;
};

export type EnsembleSegmentResult = {
  segment_id: string;
  ensemble_label: string;
  ground_truth: string;
  correct: boolean;
};

export type EnsembleResult = {
  ensemble_accuracy: number;
  best_single_model: string;
  best_single_accuracy: number;
  improvement: number;
  model_weights_used: Record<string, number>;
  per_segment: EnsembleSegmentResult[];
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
  extraction_variant_id?: string;
  extraction_label?: string;
  num_frames_used?: number;
  sampling_method_used?: string;
  sweep_id?: string;
  action_label?: ActionLabel | null;
  labeling_mode?: string | null;
  timestamp?: string | null;
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
  consensus?: ConsensusEntry | null;
};

export type RunPayload = {
  run_id: string;
  run_type?: "comparison" | "accuracy_test" | null;
  has_ensemble?: boolean;
  config: {
    models: string[];
    prompt_version: string;
    segmentation_mode: string;
    segmentation_config: Record<string, unknown>;
    extraction_config: Record<string, unknown>;
    model_configs?: Record<string, { role?: string; notes?: string }>;
    video_ids: string[];
    created_at: string;
    display_name?: string | null;
    notes?: string | null;
    labeling_mode?: string | null;
    taxonomy_path?: string | null;
  };
  labeling_mode?: string | null;
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
  llm_agreement?: Record<string, Record<string, number>> | null;
  llm_accuracy?: Record<string, AccuracyMetric> | null;
  judge_stats?: Record<string, unknown> | null;
  accuracy_by_model?: Record<string, AccuracyMetric> | null;
  segments: SegmentSummary[];
  results: LabelResult[];
  ground_truth?: GroundTruthEntry[] | null;
  consensus?: ConsensusEntry[] | null;
  consensus_summary?: ConsensusSummary | null;
  sweep?: SweepMetrics;
  taxonomy_agreement?: Record<string, unknown> | null;
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
  variant_id?: string | null;
  variant_label?: string | null;
};

export type RunListItem = {
  run_id: string;
  created_at: string;
  models: string[];
  prompt_version: string;
  video_ids: string[];
  run_type?: "comparison" | "accuracy_test" | null;
  has_ensemble?: boolean;
};

export type ModelCatalogItem = {
  name: string;
  display_name?: string;
  model_id: string;
  provider: string;
  supports_images: boolean;
  description?: string;
  tier?: "fast" | "frontier";
  estimated_cost_per_segment?: number;
  role?: string;
  notes: string;
};

export type SweepCell = {
  model_name: string;
  variant_label: string;
  variant_id: string;
  total_segments: number;
  successful_parses: number;
  parse_success_rate: number;
  avg_latency_ms: number | null;
  median_latency_ms: number | null;
  p95_latency_ms: number | null;
  avg_confidence: number | null;
  total_estimated_cost: number | null;
};

export type ModelStabilityScore = {
  model_name: string;
  self_agreement: number;
  rank_positions: number[];
  rank_stability: number;
};

export type SweepMetrics = {
  has_sweep: boolean;
  variants: string[];
  cells: SweepCell[];
  stability: ModelStabilityScore[];
  agreement_by_variant: Record<string, Record<string, Record<string, number>>>;
  parse_success_matrix: Record<string, Record<string, number>>;
  variant_id_by_label: Record<string, string>;
};
