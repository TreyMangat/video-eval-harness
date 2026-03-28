export type ResolvedRunType = "accuracy_test" | "comparison" | "benchmark";

type SummaryLike = {
  accuracy?: number | null;
  llm_accuracy?: number | null;
};

export type RunTypeLike = {
  run_id?: string;
  run_type?: "comparison" | "accuracy_test" | null | string;
  has_ensemble?: boolean;
  accuracy_by_model?: unknown;
  llm_accuracy?: unknown;
  ground_truth?: unknown;
  has_accuracy?: boolean;
  has_ground_truth?: boolean;
  has_dense?: boolean;
  labeling_mode?: string | null;
  results?: Array<{ action_label?: unknown; labeling_mode?: string | null }> | null;
  summaries?: Record<string, SummaryLike> | null;
  config?: Record<string, unknown> | null;
};

function hasObjectEntries(value: unknown): boolean {
  return !!value && typeof value === "object" && Object.keys(value).length > 0;
}

function hasAccuracySignal(run: RunTypeLike): boolean {
  if (run.has_accuracy) {
    return true;
  }
  if (hasObjectEntries(run.accuracy_by_model) || hasObjectEntries(run.llm_accuracy)) {
    return true;
  }
  if (run.summaries && typeof run.summaries === "object") {
    return Object.values(run.summaries).some(
      (summary) => summary?.accuracy != null || summary?.llm_accuracy != null
    );
  }
  return false;
}

function hasGroundTruthSignal(run: RunTypeLike): boolean {
  if (run.has_ground_truth) {
    return true;
  }
  if (run.ground_truth != null) {
    return true;
  }
  return Boolean(run.config && "ground_truth" in run.config && run.config.ground_truth);
}

export function isDenseRun(run: RunTypeLike): boolean {
  if (run.has_dense) {
    return true;
  }
  if (typeof run.labeling_mode === "string" && run.labeling_mode.trim().toLowerCase() === "dense") {
    return true;
  }

  const configLabelingMode = run.config?.labeling_mode;
  if (
    typeof configLabelingMode === "string" &&
    configLabelingMode.trim().toLowerCase() === "dense"
  ) {
    return true;
  }

  return (
    Array.isArray(run.results) &&
    run.results.some(
      (result) =>
        result?.action_label != null ||
        (typeof result?.labeling_mode === "string" &&
          result.labeling_mode.trim().toLowerCase() === "dense")
    )
  );
}

export function getRunType(run: RunTypeLike): ResolvedRunType {
  if (run.run_type === "accuracy_test") {
    return "accuracy_test";
  }
  if (run.run_type === "comparison") {
    return "comparison";
  }
  if (hasAccuracySignal(run) || hasGroundTruthSignal(run)) {
    return "benchmark";
  }
  return "comparison";
}

export function getRunTypeLabel(runType: ResolvedRunType): string {
  if (runType === "accuracy_test") {
    return "Accuracy";
  }
  if (runType === "benchmark") {
    return "Benchmark";
  }
  return "Comparison";
}

export function getRunTypeBadgeClass(runType: ResolvedRunType): string {
  if (runType === "accuracy_test") {
    return "badge-accuracy";
  }
  if (runType === "benchmark") {
    return "badge-benchmark";
  }
  return "badge-comparison";
}

export function isAccuracyTestRun(run: RunTypeLike): boolean {
  return getRunType(run) === "accuracy_test";
}

export function isComparisonRun(run: RunTypeLike): boolean {
  return getRunType(run) === "comparison";
}

export function isBenchmarkRun(run: RunTypeLike): boolean {
  return getRunType(run) === "benchmark";
}
