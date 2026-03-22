import { notFound } from "next/navigation";

import {
  AgreementMatrixCard,
  RunMetadataCard,
  SegmentComparisonSamplesCard,
  StabilityTableCard,
  VariantHeatmapCard,
} from "../../../components/analysis-panels";
import { TopNav } from "../../../components/navigation";
import { RunTypeBadge } from "../../../components/run-type-badge";
import {
  bestOverallModel,
  bestValueModel,
  buildCoreComparisonRows,
  buildParseSuccessMatrix,
  displayRunName,
  displaySegmentName,
  fastestModel,
  formatLatency,
  formatMoney,
  formatPercent,
  formatTime,
  getRunVideoLabel,
  getSweepData,
  modelColor,
  runBreadcrumb,
  selectFeaturedVariant,
  selectSampleSegments,
} from "../../../lib/analysis";
import { getRunType } from "../../../lib/run-type";
import { loadRun } from "../../../lib/run-source";
import type {
  AccuracyMetric,
  GroundTruthEntry,
  LabelResult,
  RunPayload,
  SegmentSummary,
} from "../../../lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type AccuracyEntry = {
  model_name: string;
  score: number | null;
  source_label: string;
  exact_match_rate: number | null;
  fuzzy_match_rate: number | null;
  llm_accuracy: number | null;
  mean_similarity: number | null;
  evaluated_segments: number | null;
  parse_rate: number | null;
  total_cost: number | null;
  avg_latency_ms: number | null;
};

function readFirst(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

function toNullableNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function accuracyMetricForModel(
  payload: Record<string, AccuracyMetric> | null | undefined,
  modelName: string
): AccuracyMetric | null {
  if (!payload) {
    return null;
  }
  const metric = payload[modelName];
  return metric && typeof metric === "object" ? metric : null;
}

function resolveAccuracyEntry(run: RunPayload, modelName: string): AccuracyEntry {
  const summaries = run.summaries && typeof run.summaries === "object" ? run.summaries : {};
  const summary = summaries[modelName];
  const rawAccuracy = accuracyMetricForModel(run.accuracy_by_model, modelName);
  const rawLlmAccuracy = accuracyMetricForModel(run.llm_accuracy, modelName);

  const llmAccuracy =
    toNullableNumber(rawLlmAccuracy?.llm_accuracy) ??
    toNullableNumber(rawAccuracy?.llm_accuracy) ??
    toNullableNumber(summary?.llm_accuracy);
  const directAccuracy =
    toNullableNumber(rawAccuracy?.accuracy) ?? toNullableNumber(summary?.accuracy);
  const fuzzyMatch =
    toNullableNumber(rawAccuracy?.fuzzy_match_rate) ?? toNullableNumber(summary?.fuzzy_match_rate);
  const exactMatch =
    toNullableNumber(rawAccuracy?.exact_match_rate) ?? toNullableNumber(summary?.exact_match_rate);
  const meanSimilarity = toNullableNumber(rawAccuracy?.mean_similarity);

  if (llmAccuracy != null) {
    return {
      model_name: modelName,
      score: llmAccuracy,
      source_label: "LLM-judged accuracy",
      exact_match_rate: exactMatch,
      fuzzy_match_rate: fuzzyMatch,
      llm_accuracy: llmAccuracy,
      mean_similarity: meanSimilarity,
      evaluated_segments: toNullableNumber(rawAccuracy?.evaluated_segments),
      parse_rate: toNullableNumber(summary?.parse_success_rate),
      total_cost: toNullableNumber(summary?.total_estimated_cost),
      avg_latency_ms: toNullableNumber(summary?.avg_latency_ms),
    };
  }

  if (directAccuracy != null) {
    return {
      model_name: modelName,
      score: directAccuracy,
      source_label: "Accuracy",
      exact_match_rate: exactMatch,
      fuzzy_match_rate: fuzzyMatch,
      llm_accuracy: llmAccuracy,
      mean_similarity: meanSimilarity,
      evaluated_segments: toNullableNumber(rawAccuracy?.evaluated_segments),
      parse_rate: toNullableNumber(summary?.parse_success_rate),
      total_cost: toNullableNumber(summary?.total_estimated_cost),
      avg_latency_ms: toNullableNumber(summary?.avg_latency_ms),
    };
  }

  if (fuzzyMatch != null) {
    return {
      model_name: modelName,
      score: fuzzyMatch,
      source_label: "Semantic match rate",
      exact_match_rate: exactMatch,
      fuzzy_match_rate: fuzzyMatch,
      llm_accuracy: llmAccuracy,
      mean_similarity: meanSimilarity,
      evaluated_segments: toNullableNumber(rawAccuracy?.evaluated_segments),
      parse_rate: toNullableNumber(summary?.parse_success_rate),
      total_cost: toNullableNumber(summary?.total_estimated_cost),
      avg_latency_ms: toNullableNumber(summary?.avg_latency_ms),
    };
  }

  return {
    model_name: modelName,
    score: exactMatch ?? meanSimilarity ?? null,
    source_label: exactMatch != null ? "Exact match rate" : "Mean similarity",
    exact_match_rate: exactMatch,
    fuzzy_match_rate: fuzzyMatch,
    llm_accuracy: llmAccuracy,
    mean_similarity: meanSimilarity,
    evaluated_segments: toNullableNumber(rawAccuracy?.evaluated_segments),
    parse_rate: toNullableNumber(summary?.parse_success_rate),
    total_cost: toNullableNumber(summary?.total_estimated_cost),
    avg_latency_ms: toNullableNumber(summary?.avg_latency_ms),
  };
}

function buildAccuracyEntries(run: RunPayload): AccuracyEntry[] {
  const models = Array.isArray(run.models) ? run.models : [];
  return [...models]
    .map((modelName) => resolveAccuracyEntry(run, modelName))
    .sort((left, right) => {
      const leftScore = left.score ?? Number.NEGATIVE_INFINITY;
      const rightScore = right.score ?? Number.NEGATIVE_INFINITY;
      if (leftScore !== rightScore) {
        return rightScore - leftScore;
      }

      const leftParse = left.parse_rate ?? Number.NEGATIVE_INFINITY;
      const rightParse = right.parse_rate ?? Number.NEGATIVE_INFINITY;
      if (leftParse !== rightParse) {
        return rightParse - leftParse;
      }

      return left.model_name.localeCompare(right.model_name);
    });
}

function bestAccuracyEntry(entries: AccuracyEntry[]): AccuracyEntry | null {
  return entries.find((entry) => entry.score != null) ?? null;
}

function accuracyPerDollar(entry: AccuracyEntry): number | null {
  if (entry.score == null) {
    return null;
  }
  if (entry.total_cost == null) {
    return null;
  }
  if (entry.total_cost === 0) {
    return Number.POSITIVE_INFINITY;
  }
  return entry.score / entry.total_cost;
}

function bestAccuracyValueEntry(entries: AccuracyEntry[]): AccuracyEntry | null {
  return [...entries]
    .filter((entry) => accuracyPerDollar(entry) != null)
    .sort((left, right) => {
      const leftRatio = accuracyPerDollar(left) ?? Number.NEGATIVE_INFINITY;
      const rightRatio = accuracyPerDollar(right) ?? Number.NEGATIVE_INFINITY;
      if (leftRatio !== rightRatio) {
        return rightRatio - leftRatio;
      }

      const leftScore = left.score ?? Number.NEGATIVE_INFINITY;
      const rightScore = right.score ?? Number.NEGATIVE_INFINITY;
      if (leftScore !== rightScore) {
        return rightScore - leftScore;
      }

      return left.model_name.localeCompare(right.model_name);
    })[0] ?? null;
}

function verdictSentence(
  runLabel: string,
  winner: ReturnType<typeof bestOverallModel>,
  rows: ReturnType<typeof buildCoreComparisonRows>,
  sweepData: ReturnType<typeof getSweepData>
): string {
  if (!winner) {
    return `No clear winner emerged from ${runLabel} because the exported run is missing summary metrics.`;
  }

  const stability = sweepData?.stability.find((entry) => entry.model_name === winner.model_name);
  const agreementLeader = rows[0]?.model_name === winner.model_name;
  const stabilityClause = stability
    ? `${formatPercent(stability.self_agreement)} self-agreement`
    : `${formatPercent(winner.parse_rate)} parse success`;
  const agreementClause = agreementLeader
    ? `the highest cross-model agreement at ${formatPercent(winner.agreement)}`
    : `strong cross-model agreement at ${formatPercent(winner.agreement)}`;

  return `In ${runLabel}, ${winner.model_name} is the most reliable model, with ${stabilityClause} and ${agreementClause}.`;
}

function accuracyVerdictSentence(
  runLabel: string,
  bestAccuracy: AccuracyEntry | null,
  modelCount: number,
  hasGroundTruth: boolean
): string {
  if (!bestAccuracy || bestAccuracy.score == null) {
    return `In ${runLabel}, accuracy scoring was not available for this run yet, so the report focuses on model agreement below.`;
  }

  return `In ${runLabel}, ${bestAccuracy.model_name} scored ${formatPercent(bestAccuracy.score)} accuracy against ${hasGroundTruth ? "your labels" : "the available ground truth"}${modelCount > 1 ? `, out of ${modelCount} models tested` : ""}.`;
}

function normalizeLabelText(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function labelTokens(value: string): string[] {
  return normalizeLabelText(value)
    .split(" ")
    .filter((token) => token.length > 2 && !["the", "and", "with", "from"].includes(token));
}

function labelSimilarity(left: string | null | undefined, right: string | null | undefined): number | null {
  if (!left || !right) {
    return null;
  }

  const normalizedLeft = normalizeLabelText(left);
  const normalizedRight = normalizeLabelText(right);
  if (!normalizedLeft || !normalizedRight) {
    return null;
  }
  if (normalizedLeft === normalizedRight) {
    return 1;
  }
  if (normalizedLeft.includes(normalizedRight) || normalizedRight.includes(normalizedLeft)) {
    return 0.85;
  }

  const leftTokens = new Set(labelTokens(left));
  const rightTokens = new Set(labelTokens(right));
  if (leftTokens.size === 0 || rightTokens.size === 0) {
    return null;
  }

  const intersection = [...leftTokens].filter((token) => rightTokens.has(token)).length;
  if (intersection === 0) {
    return 0;
  }

  const union = new Set([...leftTokens, ...rightTokens]).size;
  const jaccard = intersection / union;
  const overlap = intersection / Math.max(leftTokens.size, rightTokens.size);
  return Math.max(jaccard, overlap);
}

function isLikelyAccuracyMatch(
  groundTruthLabel: string | null,
  result: LabelResult | null
): { isMatch: boolean | null; score: number | null } {
  if (!groundTruthLabel || !result?.primary_action) {
    return { isMatch: null, score: null };
  }

  const score = labelSimilarity(groundTruthLabel, result.primary_action);
  if (score == null) {
    return { isMatch: null, score: null };
  }

  return { isMatch: score >= 0.6, score };
}

function getGroundTruthEntry(
  segment: SegmentSummary,
  groundTruth: GroundTruthEntry[] | null | undefined
): GroundTruthEntry | null {
  if (!groundTruth || groundTruth.length === 0) {
    return null;
  }

  return (
    groundTruth.find(
      (entry) =>
        entry.segment_id === segment.segment_id ||
        entry.segment_index === segment.segment_index
    ) ?? null
  );
}

function getGroundTruthLabel(
  segment: SegmentSummary,
  groundTruth: GroundTruthEntry[] | null | undefined
): string | null {
  const match = getGroundTruthEntry(segment, groundTruth);
  return (match?.primary_action ?? match?.label ?? null) || null;
}

function getModelResult(
  segment: SegmentSummary,
  modelName: string,
  results: LabelResult[]
): LabelResult | null {
  return (
    results.find(
      (result) =>
        result.model_name === modelName &&
        result.segment_id === segment.segment_id
    ) ?? null
  );
}

function accuracyScoreClass(score: number | null): string {
  if (score == null) {
    return "is-none";
  }
  if (score >= 0.8) {
    return "is-high";
  }
  if (score >= 0.5) {
    return "is-mid";
  }
  return "is-low";
}

function HeroSummaryCard({
  label,
  modelName,
  accentColor,
  heroValue,
  secondary,
}: {
  label: string;
  modelName: string;
  accentColor: string;
  heroValue: string;
  secondary: string;
}) {
  return (
    <article className="hero-card" style={{ borderTopColor: accentColor }}>
      <p className="hero-card-label">{label}</p>
      <p className="hero-card-model" style={{ color: accentColor }}>
        {modelName}
      </p>
      <p className="hero-card-number">{heroValue}</p>
      <p className="hero-card-secondary">{secondary}</p>
    </article>
  );
}

export default async function RunReportPage({
  params,
  searchParams,
}: {
  params: Promise<{ runId: string }>;
  searchParams: Promise<{ dataDir?: string | string[] }>;
}) {
  const { runId } = await params;
  const { dataDir: rawDataDir } = await searchParams;
  const dataDir = readFirst(rawDataDir);
  const run = await loadRun(runId, dataDir);

  if (!run) {
    notFound();
  }

  const sweepData = getSweepData(run);
  const models = Array.isArray(run.models) ? run.models : [];
  const segments = Array.isArray(run.segments) ? run.segments : [];
  const results = Array.isArray(run.results) ? run.results : [];
  const agreement = run.agreement && typeof run.agreement === "object" ? run.agreement : {};
  const groundTruth = Array.isArray(run.ground_truth) ? run.ground_truth : null;
  const rows = buildCoreComparisonRows(run, sweepData);
  const comparisonWinner = bestOverallModel(rows);
  const comparisonBestValue = bestValueModel(rows, segments.length || 1);
  const fastest = fastestModel(rows);
  const featuredVariant = selectFeaturedVariant(sweepData);
  const samples = selectSampleSegments(run, featuredVariant, 3);
  const runLabel = displayRunName(run.run_id, run.config.created_at);
  const reportName = run.config.display_name?.trim() || runLabel;
  const runType = getRunType(run);
  const parseSuccessMatrix = sweepData
    ? buildParseSuccessMatrix(sweepData, models, sweepData.variants)
    : null;
  const accuracyEntries = buildAccuracyEntries(run);
  const bestAccuracy = bestAccuracyEntry(accuracyEntries);
  const bestAccuracyValue = bestAccuracyValueEntry(accuracyEntries);
  const hasGroundTruth = Boolean(groundTruth?.length);
  const hasAccuracyMetrics = accuracyEntries.some((entry) => entry.score != null);
  const isAccuracyReport = runType === "accuracy_test" || hasGroundTruth || hasAccuracyMetrics;

  return (
    <main className="analysis-shell report-page">
      <div className="no-print">
        <TopNav active="runs" />
      </div>

      <section className="visual-card report-verdict-card">
        <p className="section-eyebrow">Printable Summary</p>
        <h1 className="report-verdict">
          {isAccuracyReport
            ? accuracyVerdictSentence(reportName, bestAccuracy, models.length, hasGroundTruth)
            : verdictSentence(runLabel, comparisonWinner, rows, sweepData)}
        </h1>
        <div className="report-subhead-row">
          <p className="report-subhead">
            {reportName}
            {featuredVariant ? ` \u00b7 sample segments from ${featuredVariant}` : ""}
          </p>
          <RunTypeBadge run={run} />
        </div>
        {isAccuracyReport ? (
          <p className="report-accuracy-note">
            {hasGroundTruth
              ? "Accuracy was measured by comparing each model's labels against your saved ground truth descriptions."
              : "This run includes accuracy metrics in addition to agreement, but segment-level ground truth labels were not included in the payload."}
          </p>
        ) : null}
      </section>

      <section className="hero-grid report-hero-grid">
        {isAccuracyReport ? (
          <>
            <HeroSummaryCard
              label="Most Accurate"
              modelName={bestAccuracy?.model_name ?? "\u2014"}
              accentColor={bestAccuracy ? modelColor(bestAccuracy.model_name) : "#97c459"}
              heroValue={bestAccuracy ? formatPercent(bestAccuracy.score) : "\u2014"}
              secondary={
                bestAccuracy
                  ? `${bestAccuracy.source_label} \u00b7 ${formatPercent(bestAccuracy.parse_rate)} parse`
                  : "Accuracy metrics unavailable"
              }
            />
            <HeroSummaryCard
              label="Best Value"
              modelName={bestAccuracyValue?.model_name ?? "\u2014"}
              accentColor={bestAccuracyValue ? modelColor(bestAccuracyValue.model_name) : "#1d9e75"}
              heroValue={
                bestAccuracyValue
                  ? accuracyPerDollar(bestAccuracyValue) === Number.POSITIVE_INFINITY
                    ? "\u221e"
                    : `${(accuracyPerDollar(bestAccuracyValue) ?? 0).toFixed(1)}x`
                  : "\u2014"
              }
              secondary={
                bestAccuracyValue
                  ? `${formatPercent(bestAccuracyValue.score)} at ${formatMoney(bestAccuracyValue.total_cost)}`
                  : "Highest accuracy per dollar"
              }
            />
            <HeroSummaryCard
              label="Fastest"
              modelName={fastest?.model_name ?? "\u2014"}
              accentColor={fastest ? modelColor(fastest.model_name) : "#ef9f27"}
              heroValue={
                fastest?.avg_latency_ms != null ? formatLatency(fastest.avg_latency_ms) : "\u2014"
              }
              secondary={
                fastest
                  ? `${formatPercent(fastest.parse_rate)} parse \u00b7 ${formatMoney(fastest.total_cost)}`
                  : "Latency unavailable"
              }
            />
          </>
        ) : (
          <>
            {comparisonWinner ? (
              <HeroSummaryCard
                label="Winner"
                modelName={comparisonWinner.model_name}
                accentColor={modelColor(comparisonWinner.model_name)}
                heroValue={formatPercent(comparisonWinner.agreement)}
                secondary={`${formatPercent(comparisonWinner.parse_rate)} parse \u00b7 ${formatMoney(comparisonWinner.total_cost)}`}
              />
            ) : null}
            {comparisonBestValue ? (
              <HeroSummaryCard
                label="Best Value"
                modelName={comparisonBestValue.model_name}
                accentColor={modelColor(comparisonBestValue.model_name)}
                heroValue={
                  comparisonBestValue.total_cost === 0
                    ? "\u221e"
                    : comparisonBestValue.total_cost && comparisonBestValue.total_cost > 0
                      ? `${((comparisonBestValue.agreement ?? 0) / (comparisonBestValue.total_cost / Math.max(segments.length, 1))).toFixed(1)}x`
                      : "\u2014"
                }
                secondary={`${formatMoney(comparisonBestValue.total_cost)} total \u00b7 ${formatPercent(comparisonBestValue.agreement)} agreement`}
              />
            ) : null}
            {fastest ? (
              <HeroSummaryCard
                label="Fastest"
                modelName={fastest.model_name}
                accentColor={modelColor(fastest.model_name)}
                heroValue={fastest.avg_latency_ms != null ? formatLatency(fastest.avg_latency_ms) : "\u2014"}
                secondary={`${formatPercent(fastest.parse_rate)} parse \u00b7 ${formatMoney(fastest.total_cost)}`}
              />
            ) : null}
          </>
        )}
      </section>

      {isAccuracyReport ? (
        <section className="visual-card accuracy-report-section">
          <div className="section-heading">
            <p className="section-eyebrow">Accuracy Results</p>
            <h3>How accurately did each model match the ground truth?</h3>
            <p className="chart-desc">
              {hasGroundTruth
                ? "These scores reflect how closely each model matched your saved labels."
                : "These scores reflect the run's available ground-truth accuracy metrics."}
            </p>
          </div>

          {hasAccuracyMetrics ? (
            <div className="accuracy-model-cards">
              {accuracyEntries.map((entry) => (
                <article key={entry.model_name} className="accuracy-model-card">
                  <span className="model-name">{entry.model_name}</span>
                  <span className={`accuracy-score ${accuracyScoreClass(entry.score)}`}>
                    {formatPercent(entry.score)}
                  </span>
                  <span className="accuracy-label">{entry.source_label}</span>
                </article>
              ))}
            </div>
          ) : (
            <div className="accuracy-pending">
              <p>
                Accuracy scoring is still processing or was not available for this run. The
                agreement views below still show where the models aligned with each other.
              </p>
            </div>
          )}

          {hasGroundTruth ? (
            <div className="segment-accuracy-table">
              <h3>Segment-by-segment breakdown</h3>
              <p className="chart-desc">
                Your labels are shown next to each model answer. Match marks use lightweight label
                similarity because per-segment judge verdicts are not included in the exported run
                payload.
              </p>
              <div className="table-scroll">
                <table className="accuracy-table">
                  <thead>
                    <tr>
                      <th>Segment</th>
                      <th>Your label</th>
                      {models.map((modelName) => (
                        <th key={modelName}>{modelName}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {segments.map((segment) => {
                      const groundTruthLabel = getGroundTruthLabel(segment, groundTruth);
                      return (
                        <tr key={segment.segment_id}>
                          <td className="seg-time">
                            {formatTime(segment.start_time_s)} - {formatTime(segment.end_time_s)}
                          </td>
                          <td className="gt-label">{groundTruthLabel ?? "\u2014"}</td>
                          {models.map((modelName) => {
                            const result = getModelResult(segment, modelName, results);
                            const { isMatch } = isLikelyAccuracyMatch(groundTruthLabel, result);
                            return (
                              <td
                                key={`${segment.segment_id}-${modelName}`}
                                className={`model-answer ${isMatch === true ? "correct" : isMatch === false ? "incorrect" : ""}`}
                              >
                                {result?.primary_action || "\u2014"}
                                {isMatch != null ? (
                                  <span
                                    className={`match-indicator ${isMatch ? "match" : "mismatch"}`}
                                  >
                                    {isMatch ? "\u2713" : "\u2717"}
                                  </span>
                                ) : null}
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}
        </section>
      ) : null}

      <AgreementMatrixCard
        title={
          isAccuracyReport
            ? "Did the models at least agree with each other?"
            : "How often do models agree with each other?"
        }
        description={
          isAccuracyReport
            ? "Even when models miss the ground truth, they may still agree with each other. That is consensus without correctness."
            : "This is the one visual to carry into a discussion: it shows how tightly the models cluster."
        }
        matrix={agreement}
      />

      <SegmentComparisonSamplesCard
        title={isAccuracyReport ? "Model Agreement by Segment" : "Sample Segment Comparisons"}
        description={
          isAccuracyReport
            ? "These samples keep the comparison view visible below the accuracy summary so you can separate correctness from consensus."
            : featuredVariant
              ? `Three representative segments from ${featuredVariant}, shown with the model outputs side by side.`
              : "Three representative segments from this run, shown with the model outputs side by side."
        }
        samples={samples}
        formatSampleTitle={(sample) => displaySegmentName(sample.segment)}
        formatSampleMeta={(sample) =>
          sample.variant_label ? sample.variant_label : "Representative comparison"
        }
      />

      <details className="visual-card full-details-card">
        <summary className="sweep-summary">
          <div>
            <p className="section-eyebrow">Appendix</p>
            <h3>Full details</h3>
            <p>Context and sweep diagnostics for anyone who wants the full benchmark picture.</p>
          </div>
        </summary>

        <RunMetadataCard
          title="Run Context"
          runId={run.run_id}
          createdAt={run.config.created_at}
          videoLabel={getRunVideoLabel(run)}
          promptVersion={run.config.prompt_version}
          models={models}
          segments={segments.length}
          compact
          compactText={runBreadcrumb(run)}
        />

        {sweepData && parseSuccessMatrix ? (
          <VariantHeatmapCard
            title="Model x Variant Parse Success"
            description="Parse success heatmap from the run's exported sweep summary."
            models={models}
            variants={sweepData.variants}
            matrix={parseSuccessMatrix}
          />
        ) : null}

        {sweepData ? (
          <StabilityTableCard
            title="Stability Scores"
            description="Self-agreement and rank stability across the extraction variants in this run."
            stability={sweepData.stability}
          />
        ) : null}
      </details>
    </main>
  );
}
