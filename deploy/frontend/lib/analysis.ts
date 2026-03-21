import { computeAgreementMatrix, computeSweepMetrics } from "./run-metrics";
import type {
  LabelResult,
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

function average(values: number[]): number | null {
  if (values.length === 0) {
    return null;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
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

export function resultVariantLabel(result: LabelResult): string {
  if (result.extraction_variant_id?.trim()) {
    return result.extraction_label?.trim() || result.extraction_variant_id;
  }
  return "default";
}

export function getSweepData(run: RunPayload): SweepMetrics | null {
  if (run.sweep?.has_sweep) {
    return run.sweep;
  }
  const computed = computeSweepMetrics(run.results);
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
  const filteredResults = filterResultsByVariant(run.results, variantLabel);

  return run.segments
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
        mean_agreement: averageOffDiagonal(computeAgreementMatrix(segmentResults)),
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
  const models = [...new Set([...leftRun.models, ...rightRun.models])].sort();

  return models.map((modelName) => {
    const leftSummary = leftRun.summaries[modelName];
    const rightSummary = rightRun.summaries[modelName];
    const leftAgreement = meanModelAgreement(leftRun.agreement, modelName);
    const rightAgreement = meanModelAgreement(rightRun.agreement, modelName);

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
