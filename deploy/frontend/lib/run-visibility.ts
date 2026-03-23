type RunVisibilityLike = {
  run_id?: string | null;
  name?: string | null;
  display_name?: string | null;
  models?: unknown;
  summaries?: Record<string, unknown> | null;
  config?: Record<string, unknown> | null;
};

const HIDDEN_RUN_PATTERNS = [
  "debug-fast",
  "debug-test",
  "run-type-accuracy-test",
  "run-type-comparison-test",
  "accuracy-e2e-test",
  "e2e-test",
  "e2e-final-test",
  "e2e-smoke",
  "stage-e2e",
  "cors-test",
  "volume-fix",
  "preview-fix",
  "warmup-test",
  "fix-test",
  "report-accuracy-live-test",
  "node-test",
];

function normalizedRunName(run: RunVisibilityLike): string {
  const configDisplayName =
    run.config && typeof run.config.display_name === "string"
      ? run.config.display_name
      : null;

  return [run.name, run.display_name, configDisplayName, run.run_id]
    .filter((value): value is string => typeof value === "string" && value.trim().length > 0)
    .join(" ")
    .toLowerCase();
}

function runModelCount(run: RunVisibilityLike): number {
  if (run.summaries && typeof run.summaries === "object") {
    return Object.keys(run.summaries).length;
  }

  if (Array.isArray(run.models)) {
    return run.models.filter(
      (entry): entry is string => typeof entry === "string" && entry.trim().length > 0
    ).length;
  }

  return 0;
}

export function isVisibleRun(run: RunVisibilityLike): boolean {
  const normalized = normalizedRunName(run);
  if (HIDDEN_RUN_PATTERNS.some((pattern) => normalized.includes(pattern))) {
    return false;
  }

  return runModelCount(run) > 0;
}
