import { computeAgreementMatrix, computeSweepMetrics } from "./run-metrics";
import type {
  LabelResult,
  ModelSummary,
  RunPayload,
  SegmentSummary,
  SweepMetrics,
} from "./types";

export type SegmentComparisonSample = {
  segment: SegmentSummary;
  results: LabelResult[];
  mean_agreement: number | null;
  variant_label: string | null;
};

export type ModelDeltaRow = {
  model_name: string;
  left_parse_rate: number | null;
  right_parse_rate: number | null;
  parse_rate_delta: number | null;
  left_latency_ms: number | null;
  right_latency_ms: number | null;
  latency_delta_ms: number | null;
  left_agreement: number | null;
  right_agreement: number | null;
  agreement_delta: number | null;
};

export type ParseSuccessMatrix = Record<string, Record<string, number | null>>;

export type CoreComparisonRow = {
  model_name: string;
  parse_rate: number | null;
  agreement: number | null;
  accuracy: number | null;
  llm_accuracy: number | null;
  fuzzy_accuracy: number | null;
  confidence: number | null;
  avg_latency_ms: number | null;
  total_cost: number | null;
  stability: number | null;
  input_mode: string;
};

export type CostByModelRow = {
  model_name: string;
  total_cost: number;
};

export type CostByVariantRow = {
  variant_label: string;
  total_cost: number;
};

export type CostBySegmentRow = {
  segment_id: string;
  video_id: string;
  start_time_s: number;
  total_cost: number;
};

export type CostBreakdown = {
  total_cost: number;
  by_model: CostByModelRow[];
  by_variant: CostByVariantRow[];
  by_segment: CostBySegmentRow[];
};

const MODEL_COLOR_MAP: Array<{ match: string; color: string }> = [
  { match: "mistral-large", color: "#D85A30" },
  { match: "mistral-small-4-2503", color: "#F0997B" },
  { match: "mistral-small-4", color: "#F0997B" },
  { match: "mistral-small-3.1", color: "#F5C4B3" },
  { match: "grok-4.1", color: "#D4537E" },
  { match: "grok-4-fast", color: "#D4537E" },
  { match: "grok-4", color: "#993556" },
  { match: "nemotron", color: "#1D9E75" },
  { match: "qwen3.5-122b", color: "#AFA9EC" },
  { match: "qwen3.5-9b", color: "#CECBF6" },
  { match: "gemini", color: "#4c9aff" },
  { match: "gpt", color: "#22c55e" },
  { match: "qwen", color: "#a855f7" },
  { match: "llama", color: "#f59e0b" },
];

export type ModelTier = "fast" | "frontier" | "free" | "unknown";

const FAST_TIER_MATCHES = [
  "gemini-3-flash",
  "gpt-5.4-mini",
  "qwen3.5-27b",
  "mistral-small-4",
  "mistral-small-4-2503",
  "qwen3.5-9b",
];

const FREE_TIER_MATCHES = ["nemotron-nano", "mistral-small-3.1"];

const FRONTIER_TIER_MATCHES = [
  "gemini-3.1-pro",
  "gpt-5.4",
  "qwen3.5-vl",
  "llama-4-maverick",
  "mistral-large",
  "grok-4",
  "qwen3.5-122b",
];

function average(values: number[]): number | null {
  if (values.length === 0) {
    return null;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function safeModels(run: RunPayload): string[] {
  return Array.isArray(run.models) ? run.models : [];
}

function safeVideos(run: RunPayload): RunPayload["videos"] {
  return Array.isArray(run.videos) ? run.videos : [];
}

function safeVideoIds(run: RunPayload): string[] {
  return Array.isArray(run.config?.video_ids) ? run.config.video_ids : [];
}

function safeSegments(run: RunPayload): SegmentSummary[] {
  return Array.isArray(run.segments) ? run.segments : [];
}

function safeResults(run: RunPayload): LabelResult[] {
  return Array.isArray(run.results) ? run.results : [];
}

function safeSummaries(run: RunPayload): Record<string, ModelSummary> {
  return run.summaries && typeof run.summaries === "object" ? run.summaries : {};
}

function safeAgreement(run: RunPayload): Record<string, Record<string, number>> {
  return run.agreement && typeof run.agreement === "object" ? run.agreement : {};
}

function titleCaseToken(value: string): string {
  if (!value) {
    return value;
  }
  return value[0].toUpperCase() + value.slice(1);
}

function formatVersionToken(value: string): string {
  const match = /^v(\d{2,4})$/i.exec(value);
  if (!match) {
    return titleCaseToken(value);
  }
  return `v${match[1].split("").join(".")}`;
}

function humanizeRunSlug(value: string): string {
  return value
    .split(/[_-]+/)
    .filter(Boolean)
    .map((token) => formatVersionToken(token))
    .join(" ");
}

function formatRunTimestamp(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Run";
  }

  const stamp = new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
  return `Run ${stamp}`;
}

function sortByAgreementThenParse(
  left: CoreComparisonRow,
  right: CoreComparisonRow
): number {
  return (
    compareNullableNumbers(left.agreement, right.agreement, true) ||
    compareNullableNumbers(left.parse_rate, right.parse_rate, true) ||
    left.model_name.localeCompare(right.model_name)
  );
}

export function formatPercent(value: number | null | undefined): string {
  if (value == null) {
    return "-";
  }
  return `${Math.round(value * 100)}%`;
}

export function formatSignedPercentDelta(value: number | null | undefined): string {
  if (value == null) {
    return "-";
  }
  const percent = (value * 100).toFixed(1);
  return `${value > 0 ? "+" : ""}${percent}%`;
}

export function formatMoney(value: number | null | undefined): string {
  if (value == null) {
    return "-";
  }
  return `$${value.toFixed(4)}`;
}

export function formatLatency(value: number | null | undefined): string {
  if (value == null) {
    return "-";
  }
  return `${Math.round(value)} ms`;
}

export function formatSignedLatency(value: number | null | undefined): string {
  if (value == null) {
    return "-";
  }
  return `${value > 0 ? "+" : ""}${Math.round(value)} ms`;
}

export function formatDateTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

export function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${mins}:${secs}`;
}

export function displayRunName(runId: string, createdAt?: string): string {
  const raw = runId.trim();
  const withoutPrefix = raw.replace(/^run_/, "");
  const legacyHashOnly = /^[a-f0-9]{8,}$/i.test(withoutPrefix);
  let cleaned = withoutPrefix.replace(/^\d{8}_/, "");
  cleaned = cleaned.replace(/_[a-f0-9]{4,16}$/i, "");

  let isSweep = false;
  if (cleaned.endsWith("_sweep")) {
    cleaned = cleaned.slice(0, -"_sweep".length);
    isSweep = true;
  }

  if (!cleaned || legacyHashOnly || /^[a-f0-9]{8,}$/i.test(cleaned)) {
    return createdAt ? formatRunTimestamp(createdAt) : "Benchmark Run";
  }

  const label = humanizeRunSlug(cleaned);
  return isSweep ? `${label} (sweep)` : label;
}

export function displayVideoName(videoId: string): string {
  return videoId.trim().replace(/^vid_/, "").replace(/_[a-f0-9]{8,16}$/i, "");
}

export function displaySegmentName(segment: Pick<SegmentSummary, "video_id" | "start_time_s" | "end_time_s">): string {
  return `${displayVideoName(segment.video_id)} ${formatTime(segment.start_time_s)}-${formatTime(segment.end_time_s)}`;
}

export function modelColor(modelName: string): string {
  const normalized = modelName.toLowerCase();
  for (const entry of MODEL_COLOR_MAP) {
    if (normalized.includes(entry.match)) {
      return entry.color;
    }
  }
  return "#38bdf8";
}

export function getModelTier(modelName: string): ModelTier {
  const normalized = modelName.toLowerCase();

  if (FREE_TIER_MATCHES.some((match) => normalized.includes(match))) {
    return "free";
  }
  if (FAST_TIER_MATCHES.some((match) => normalized.includes(match))) {
    return "fast";
  }
  if (FRONTIER_TIER_MATCHES.some((match) => normalized.includes(match))) {
    return "frontier";
  }
  return "unknown";
}

export function overallModelScore(row: CoreComparisonRow): number | null {
  const primaryScore = row.accuracy ?? row.agreement;
  if (row.parse_rate == null && primaryScore == null) {
    return null;
  }
  return ((row.parse_rate ?? 0) + (primaryScore ?? 0)) / 2;
}

export function agreementPerDollarScore(
  row: CoreComparisonRow,
  segmentCount: number
): number | null {
  const primaryScore = row.accuracy ?? row.agreement;
  if (primaryScore == null || row.total_cost == null || segmentCount <= 0) {
    return null;
  }
  if (row.total_cost === 0) {
    return Number.POSITIVE_INFINITY;
  }
  if (row.total_cost < 0) {
    return null;
  }
  return primaryScore / (row.total_cost / segmentCount);
}

export function resultVariantLabel(result: LabelResult): string {
  if (result.extraction_variant_id?.trim()) {
    return result.extraction_label?.trim() || result.extraction_variant_id;
  }
  return "default";
}

export function getRunVideoLabel(run: RunPayload): string {
  const videoIds = safeVideoIds(run);
  const videos = safeVideos(run);
  const explicitVideoCount = videoIds.length;
  if (explicitVideoCount > 1) {
    return `${explicitVideoCount} videos`;
  }
  return (
    videos[0]?.filename ||
    (videos[0]?.video_id ? displayVideoName(videos[0].video_id) : null) ||
    (videoIds[0] ? displayVideoName(videoIds[0]) : null) ||
    "Unknown video"
  );
}

export function runBreadcrumb(run: RunPayload): string {
  const models = safeModels(run);
  const segments = safeSegments(run);
  const videos = safeVideos(run);
  const videoCount = safeVideoIds(run).length || videos.length || 0;
  return [
    displayRunName(run.run_id, run.config.created_at),
    `${models.length} ${models.length === 1 ? "model" : "models"}`,
    `${videoCount} ${videoCount === 1 ? "video" : "videos"}`,
    `${segments.length} ${segments.length === 1 ? "segment" : "segments"}`,
  ].join(" \u00b7 ");
}

export function getSweepData(run: RunPayload): SweepMetrics | null {
  if (run.sweep?.has_sweep) {
    return run.sweep;
  }
  const computed = computeSweepMetrics(safeResults(run));
  return computed.has_sweep ? computed : null;
}

function averageOffDiagonal(
  matrix: Record<string, Record<string, number>>,
  modelName?: string
): number | null {
  const rows = modelName ? [modelName] : Object.keys(matrix);
  const values: number[] = [];

  for (const row of rows) {
    for (const [column, value] of Object.entries(matrix[row] ?? {})) {
      if (row === column) {
        continue;
      }
      values.push(value);
    }
  }

  return average(values);
}

export function meanModelAgreement(
  matrix: Record<string, Record<string, number>>,
  modelName: string
): number | null {
  return averageOffDiagonal(matrix, modelName);
}

function compareNullableNumbers(
  left: number | null | undefined,
  right: number | null | undefined,
  descending = true
): number {
  const leftValue = left ?? Number.NEGATIVE_INFINITY;
  const rightValue = right ?? Number.NEGATIVE_INFINITY;
  return descending ? rightValue - leftValue : leftValue - rightValue;
}

function resolveLlmAccuracy(summary: ModelSummary | undefined): number | null {
  if (!summary) {
    return null;
  }
  if (typeof summary.llm_accuracy === "number") {
    return summary.llm_accuracy;
  }
  if (typeof summary.accuracy === "number") {
    return summary.accuracy;
  }
  return null;
}

function resolveInputMode(summary: ModelSummary | undefined): string {
  const raw = typeof summary?.input_mode === "string" ? summary.input_mode.trim().toLowerCase() : "";
  if (!raw) {
    return "frames";
  }
  if (raw.includes("video")) {
    return "video";
  }
  return "frames";
}

export function buildCoreComparisonRows(
  run: RunPayload,
  sweepData: SweepMetrics | null
): CoreComparisonRow[] {
  const models = safeModels(run);
  const summaries = safeSummaries(run);
  const agreement = safeAgreement(run);
  const stabilityByModel = new Map(
    (sweepData?.stability ?? []).map((entry) => [entry.model_name, entry.rank_stability])
  );

  return models
    .map((modelName) => {
      const summary = summaries[modelName];
      return {
        model_name: modelName,
        parse_rate: summary?.parse_success_rate ?? null,
        agreement: meanModelAgreement(agreement, modelName),
        accuracy: summary?.exact_match_rate ?? null,
        llm_accuracy: resolveLlmAccuracy(summary),
        fuzzy_accuracy: summary?.fuzzy_match_rate ?? null,
        confidence: summary?.avg_confidence ?? null,
        avg_latency_ms: summary?.avg_latency_ms ?? null,
        total_cost: summary?.total_estimated_cost ?? null,
        stability: stabilityByModel.get(modelName) ?? null,
        input_mode: resolveInputMode(summary),
      };
    })
    .sort(
      (left, right) =>
        compareNullableNumbers(left.agreement, right.agreement, true) ||
        compareNullableNumbers(left.parse_rate, right.parse_rate, true) ||
        compareNullableNumbers(left.confidence, right.confidence, true) ||
        left.model_name.localeCompare(right.model_name)
    );
}

export function hasAccuracy(rows: CoreComparisonRow[]): boolean {
  return rows.some((row) => row.accuracy != null);
}

export function bestAgreementModel(rows: CoreComparisonRow[]): CoreComparisonRow | null {
  return rows.find((row) => row.agreement != null) ?? null;
}

export function bestOverallModel(rows: CoreComparisonRow[]): CoreComparisonRow | null {
  return [...rows]
    .filter((row) => overallModelScore(row) != null)
    .sort(
      (left, right) =>
        compareNullableNumbers(overallModelScore(left), overallModelScore(right), true) ||
        compareNullableNumbers(left.agreement, right.agreement, true) ||
        compareNullableNumbers(left.parse_rate, right.parse_rate, true) ||
        left.model_name.localeCompare(right.model_name)
    )[0] ?? null;
}

export function bestValueModel(
  rows: CoreComparisonRow[],
  segmentCount: number
): CoreComparisonRow | null {
  const scoredRows = [...rows]
    .filter((row) => agreementPerDollarScore(row, segmentCount) != null)
    .sort(
      (left, right) =>
        compareNullableNumbers(
          agreementPerDollarScore(left, segmentCount),
          agreementPerDollarScore(right, segmentCount),
          true
        ) ||
        sortByAgreementThenParse(left, right) ||
        compareNullableNumbers(left.total_cost, right.total_cost, false) ||
        left.model_name.localeCompare(right.model_name)
    );

  if (scoredRows.length > 0) {
    return scoredRows[0] ?? null;
  }

  return [...rows]
    .filter((row) => row.agreement != null)
    .sort(sortByAgreementThenParse)[0] ?? null;
}

export function lowestCostModel(rows: CoreComparisonRow[]): CoreComparisonRow | null {
  return [...rows]
    .filter((row) => row.total_cost != null)
    .sort(
      (left, right) =>
        compareNullableNumbers(left.total_cost, right.total_cost, false) ||
        left.model_name.localeCompare(right.model_name)
    )[0] ?? null;
}

export function fastestModel(rows: CoreComparisonRow[]): CoreComparisonRow | null {
  const rowsWithLatency = [...rows]
    .filter((row) => row.avg_latency_ms != null && row.avg_latency_ms > 0)
    .sort(
      (left, right) =>
        compareNullableNumbers(left.avg_latency_ms, right.avg_latency_ms, false) ||
        compareNullableNumbers(left.parse_rate, right.parse_rate, true) ||
        left.model_name.localeCompare(right.model_name)
    );

  if (rowsWithLatency.length > 0) {
    return rowsWithLatency[0] ?? null;
  }

  return [...rows]
    .filter((row) => row.parse_rate != null || row.agreement != null)
    .sort(sortByAgreementThenParse)[0] ?? null;
}

export function selectFeaturedVariant(sweep: SweepMetrics | null): string | null {
  if (!sweep?.variants.length) {
    return null;
  }

  const rankedVariants = sweep.variants
    .map((variant) => {
      const cells = sweep.cells.filter((cell) => cell.variant_label === variant);
      return {
        variant,
        parse_rate: average(cells.map((cell) => cell.parse_success_rate)) ?? 0,
        latency_ms:
          average(
            cells
              .map((cell) => cell.avg_latency_ms)
              .filter((value): value is number => value != null)
          ) ?? Number.POSITIVE_INFINITY,
      };
    })
    .sort(
      (left, right) =>
        right.parse_rate - left.parse_rate ||
        left.latency_ms - right.latency_ms ||
        left.variant.localeCompare(right.variant)
    );

  return rankedVariants[0]?.variant ?? null;
}

export function filterResultsByVariant(
  results: LabelResult[],
  variantLabel: string | null
): LabelResult[] {
  if (!variantLabel) {
    return results;
  }
  return results.filter((result) => resultVariantLabel(result) === variantLabel);
}

export function selectSampleSegments(
  run: RunPayload,
  variantLabel: string | null,
  limit = 3
): SegmentComparisonSample[] {
  const segments = safeSegments(run);
  const filteredResults = filterResultsByVariant(safeResults(run), variantLabel);
  const agreement = safeAgreement(run);
  const preferredAgreementMatrix =
    (variantLabel ? run.sweep?.agreement_by_variant?.[variantLabel] : null) ??
    (Object.keys(agreement).length > 0 ? agreement : null);

  return segments
    .map((segment) => {
      const segmentResults = filteredResults
        .filter((result) => result.segment_id === segment.segment_id)
        .sort(
          (left, right) =>
            left.model_name.localeCompare(right.model_name) ||
            resultVariantLabel(left).localeCompare(resultVariantLabel(right))
        );
      if (segmentResults.length === 0) {
        return null;
      }
      return {
        segment,
        results: segmentResults,
        mean_agreement: averageOffDiagonal(
          preferredAgreementMatrix ?? computeAgreementMatrix(segmentResults)
        ),
        variant_label: variantLabel,
      };
    })
    .filter((sample): sample is SegmentComparisonSample => sample != null)
    .sort(
      (left, right) =>
        (left.mean_agreement ?? 1) - (right.mean_agreement ?? 1) ||
        left.segment.start_time_s - right.segment.start_time_s
    )
    .slice(0, limit);
}

export function buildModelDeltaRows(
  leftRun: RunPayload,
  rightRun: RunPayload
): ModelDeltaRow[] {
  const models = [...new Set([...safeModels(leftRun), ...safeModels(rightRun)])].sort();
  const leftSummaries = safeSummaries(leftRun);
  const rightSummaries = safeSummaries(rightRun);
  const leftAgreementMatrix = safeAgreement(leftRun);
  const rightAgreementMatrix = safeAgreement(rightRun);

  return models.map((modelName) => {
    const leftSummary = leftSummaries[modelName];
    const rightSummary = rightSummaries[modelName];
    const leftAgreement = meanModelAgreement(leftAgreementMatrix, modelName);
    const rightAgreement = meanModelAgreement(rightAgreementMatrix, modelName);

    return {
      model_name: modelName,
      left_parse_rate: leftSummary?.parse_success_rate ?? null,
      right_parse_rate: rightSummary?.parse_success_rate ?? null,
      parse_rate_delta:
        leftSummary && rightSummary
          ? rightSummary.parse_success_rate - leftSummary.parse_success_rate
          : null,
      left_latency_ms: leftSummary?.avg_latency_ms ?? null,
      right_latency_ms: rightSummary?.avg_latency_ms ?? null,
      latency_delta_ms:
        leftSummary?.avg_latency_ms != null && rightSummary?.avg_latency_ms != null
          ? rightSummary.avg_latency_ms - leftSummary.avg_latency_ms
          : null,
      left_agreement: leftAgreement,
      right_agreement: rightAgreement,
      agreement_delta:
        leftAgreement != null && rightAgreement != null
          ? rightAgreement - leftAgreement
          : null,
    };
  });
}

export function buildCostBreakdown(
  run: RunPayload,
  sweepData: SweepMetrics | null
): CostBreakdown {
  const results = safeResults(run);
  const models = safeModels(run);
  const segments = safeSegments(run);
  const totalCost = results.reduce((sum, result) => sum + (result.estimated_cost ?? 0), 0);

  const byModel = models
    .map((modelName) => ({
      model_name: modelName,
      total_cost: results
        .filter((result) => result.model_name === modelName)
        .reduce((sum, result) => sum + (result.estimated_cost ?? 0), 0),
    }))
    .sort((left, right) => right.total_cost - left.total_cost);

  const variantTotals = new Map<string, number>();
  for (const result of results) {
    const variantLabel = resultVariantLabel(result);
    variantTotals.set(
      variantLabel,
      (variantTotals.get(variantLabel) ?? 0) + (result.estimated_cost ?? 0)
    );
  }
  const preferredVariantOrder = sweepData?.variants ?? [];
  const byVariant = [...variantTotals.entries()]
    .map(([variant_label, total_cost]) => ({ variant_label, total_cost }))
    .sort((left, right) => {
      const leftIndex = preferredVariantOrder.indexOf(left.variant_label);
      const rightIndex = preferredVariantOrder.indexOf(right.variant_label);
      if (leftIndex !== -1 || rightIndex !== -1) {
        return (
          (leftIndex === -1 ? Number.MAX_SAFE_INTEGER : leftIndex) -
          (rightIndex === -1 ? Number.MAX_SAFE_INTEGER : rightIndex)
        );
      }
      return right.total_cost - left.total_cost;
    });

  const segmentLookup = new Map(segments.map((segment) => [segment.segment_id, segment] as const));
  const segmentTotals = new Map<string, CostBySegmentRow>();
  for (const result of results) {
    const segment = segmentLookup.get(result.segment_id);
    const current = segmentTotals.get(result.segment_id) ?? {
      segment_id: result.segment_id,
      video_id: result.video_id,
      start_time_s: segment?.start_time_s ?? result.start_time_s,
      total_cost: 0,
    };
    current.total_cost += result.estimated_cost ?? 0;
    segmentTotals.set(result.segment_id, current);
  }
  const bySegment = [...segmentTotals.values()].sort(
    (left, right) =>
      left.video_id.localeCompare(right.video_id) ||
      left.start_time_s - right.start_time_s
  );

  return {
    total_cost: totalCost,
    by_model: byModel,
    by_variant: byVariant,
    by_segment: bySegment,
  };
}

export function groupSegmentsByVideo(run: RunPayload): Record<string, SegmentSummary[]> {
  const segments = safeSegments(run);
  return Object.fromEntries(
    [...new Set(segments.map((segment) => segment.video_id))]
      .sort()
      .map((videoId) => [
        videoId,
        segments
          .filter((segment) => segment.video_id === videoId)
          .sort((left, right) => left.start_time_s - right.start_time_s),
      ])
  );
}

export function buildParseSuccessMatrix(
  sweep: SweepMetrics | null,
  models: string[],
  variants: string[]
): ParseSuccessMatrix {
  const matrix: ParseSuccessMatrix = {};

  for (const modelName of models) {
    matrix[modelName] = {};
    for (const variant of variants) {
      matrix[modelName][variant] = sweep?.parse_success_matrix[modelName]?.[variant] ?? null;
    }
  }

  return matrix;
}

export function buildSweepDeltaMatrix(
  leftSweep: SweepMetrics | null,
  rightSweep: SweepMetrics | null,
  leftModels: string[],
  rightModels: string[]
): {
  models: string[];
  variants: string[];
  left_matrix: ParseSuccessMatrix;
  right_matrix: ParseSuccessMatrix;
  delta_matrix: ParseSuccessMatrix;
} | null {
  if (!leftSweep || !rightSweep) {
    return null;
  }

  const models = [...new Set([...leftModels, ...rightModels])].sort();
  const variants = [...new Set([...leftSweep.variants, ...rightSweep.variants])].sort();
  const leftMatrix = buildParseSuccessMatrix(leftSweep, models, variants);
  const rightMatrix = buildParseSuccessMatrix(rightSweep, models, variants);
  const deltaMatrix: ParseSuccessMatrix = {};

  for (const modelName of models) {
    deltaMatrix[modelName] = {};
    for (const variant of variants) {
      const leftValue = leftMatrix[modelName][variant];
      const rightValue = rightMatrix[modelName][variant];
      deltaMatrix[modelName][variant] =
        leftValue != null && rightValue != null ? rightValue - leftValue : null;
    }
  }

  return {
    models,
    variants,
    left_matrix: leftMatrix,
    right_matrix: rightMatrix,
    delta_matrix: deltaMatrix,
  };
}
