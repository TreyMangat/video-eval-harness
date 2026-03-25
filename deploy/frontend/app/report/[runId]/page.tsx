import { notFound } from "next/navigation";

import {
  AgreementMatrixCard,
  RunMetadataCard,
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
  displayVideoName,
  filterResultsByVariant,
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

type ReportSegment = {
  segment_id: string;
  video_id: string;
  video_name: string;
  segment_index: number | null;
  start_s: number;
  end_s: number;
};

function readFirst(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

function toNullableNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function average(values: number[]): number | null {
  if (values.length === 0) {
    return null;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function safeNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function formatConfidence(value: number | null | undefined): string | null {
  if (value == null || !Number.isFinite(value)) {
    return null;
  }
  const normalized = value > 1 ? value / 100 : value;
  return `${Math.round(Math.max(0, Math.min(1, normalized)) * 100)}%`;
}

function normalizeReportSegments(segments: SegmentSummary[]): ReportSegment[] {
  return (segments || []).map((segment) => ({
      segment_id: segment.segment_id,
      video_id: segment.video_id,
      video_name: displayVideoName(segment.video_id),
      segment_index: segment.segment_index,
      start_s: segment.start_time_s,
      end_s: segment.end_time_s,
    }))
    .sort(
      (left, right) =>
        left.video_id.localeCompare(right.video_id) || left.start_s - right.start_s
    );
}

function getUniqueSegments(results: LabelResult[]): ReportSegment[] {
  const seen = new Map<string, ReportSegment>();

  for (const result of results) {
    const key =
      result.segment_id?.trim() || `${result.video_id}_${safeNumber(result.start_time_s)}`;

    if (!seen.has(key)) {
      seen.set(key, {
        segment_id: result.segment_id,
        video_id: result.video_id,
        video_name: displayVideoName(result.video_id),
        segment_index: null,
        start_s: safeNumber(result.start_time_s),
        end_s: safeNumber(result.end_time_s),
      });
    }
  }

  return [...seen.values()].sort(
    (left, right) =>
      left.video_id.localeCompare(right.video_id) || left.start_s - right.start_s
  );
}

function collectAgreementValues(value: unknown, metricKeys: string[]): number[] {
  if (Array.isArray(value)) {
    return value.flatMap((entry) => collectAgreementValues(entry, metricKeys));
  }

  if (!value || typeof value !== "object") {
    return [];
  }

  const record = value as Record<string, unknown>;
  const directValues = (metricKeys || []).map((key) => toNullableNumber(record[key])).filter(
    (entry): entry is number => entry != null && entry >= 0 && entry <= 1
  );
  const nestedValues = Object.entries(record || {}).filter(([key]) => !metricKeys.includes(key)).flatMap(
    ([, entry]) => collectAgreementValues(entry, metricKeys)
  );

  return [...directValues, ...nestedValues];
}

function summarizeTaxonomyAgreement(value: unknown): {
  verb: number | null;
  noun: number | null;
  action: number | null;
} | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const verb = average(
    collectAgreementValues(value, ["verb_agreement", "avg_verb_agreement", "mean_verb_agreement"])
  );
  const noun = average(
    collectAgreementValues(value, ["noun_agreement", "avg_noun_agreement", "mean_noun_agreement"])
  );
  const action = average(
    collectAgreementValues(value, [
      "action_agreement",
      "avg_action_agreement",
      "mean_action_agreement",
    ])
  );

  if (verb == null && noun == null && action == null) {
    return null;
  }

  return { verb, noun, action };
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
  return (models || []).map((modelName) => resolveAccuracyEntry(run, modelName))
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
  return (entries || []).filter((entry) => accuracyPerDollar(entry) != null)
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

function getGroundTruthEntry(
  segment: ReportSegment,
  groundTruth: GroundTruthEntry[] | null | undefined
): GroundTruthEntry | null {
  if (!groundTruth || groundTruth.length === 0) {
    return null;
  }

  return (
    groundTruth.find(
      (entry) =>
        entry.segment_id === segment.segment_id ||
        entry.segment_index === segment.segment_index ||
        (entry.video_id === segment.video_id &&
          Math.abs((entry.start_time_s ?? 0) - segment.start_s) < 1 &&
          Math.abs((entry.end_time_s ?? 0) - segment.end_s) < 1)
    ) ?? null
  );
}

function getGroundTruthLabel(
  segment: ReportSegment,
  groundTruth: GroundTruthEntry[] | null | undefined
): string | null {
  const match = getGroundTruthEntry(segment, groundTruth);
  return (match?.primary_action ?? match?.label ?? null) || null;
}

function getModelResult(
  segment: ReportSegment,
  modelName: string,
  results: LabelResult[]
): LabelResult | null {
  return (
    results.find(
      (result) =>
        result.model_name === modelName &&
        (result.segment_id === segment.segment_id ||
          (result.video_id === segment.video_id &&
            Math.abs((result.start_time_s ?? 0) - segment.start_s) < 1))
    ) ?? null
  );
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
  const featuredVariant = selectFeaturedVariant(sweepData);
  const displayedResults = filterResultsByVariant(results, featuredVariant);
  const allSegments =
    segments.length > 0 ? normalizeReportSegments(segments) : getUniqueSegments(displayedResults);
  const agreement = run.agreement && typeof run.agreement === "object" ? run.agreement : {};
  const groundTruth = Array.isArray(run.ground_truth) ? run.ground_truth : null;
  const rows = buildCoreComparisonRows(run, sweepData);
  const comparisonWinner = bestOverallModel(rows);
  const comparisonBestValue = bestValueModel(rows, segments.length || 1);
  const fastest = fastestModel(rows);
  const taxonomyAgreement = summarizeTaxonomyAgreement(run.taxonomy_agreement);
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
  const isDenseMode =
    run.labeling_mode === "dense" ||
    run.config?.labeling_mode === "dense" ||
    results.some(
      (result) =>
        result.action_label != null || result.labeling_mode?.trim().toLowerCase() === "dense"
    );

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
            {featuredVariant ? ` \u00b7 segment view using ${featuredVariant}` : ""}
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

      <section className="visual-card all-segments-section">
        <div className="section-heading">
          <p className="section-eyebrow">Segment Breakdown</p>
          <h2>What each model said, segment by segment</h2>
          <p className="chart-desc">
            {featuredVariant
              ? `Every segment from this benchmark, using ${featuredVariant} so each model appears side by side.`
              : "Every segment from this benchmark, with each model's label shown side by side."}
            {hasGroundTruth
              ? " Your saved label appears on each segment when ground truth is available."
              : ""}
          </p>
        </div>

        {allSegments.length > 0 ? (
          <div className="segment-list">
            {(allSegments || []).map((segment) => {
              const groundTruthLabel = getGroundTruthLabel(segment, groundTruth);
              return (
                <div key={segment.segment_id} className="segment-row">
                  <div className="segment-header">
                    <span className="segment-time">
                      {segment.video_name || segment.video_id} · {formatTime(segment.start_s)} -{" "}
                      {formatTime(segment.end_s)}
                    </span>
                    {groundTruthLabel ? (
                      <span className="gt-badge">Your label: {groundTruthLabel}</span>
                    ) : null}
                  </div>

                  <div className="model-labels">
                    {(models || []).map((modelName) => {
                      const result = getModelResult(segment, modelName, displayedResults);
                      const parsed = result ? result.parsed_success !== false : true;
                      const confidenceLabel = formatConfidence(
                        result?.action_label?.confidence ?? result?.confidence ?? null
                      );

                      return (
                        <div
                          key={`${segment.segment_id}-${modelName}`}
                          className={`model-label-card ${result && !parsed ? "parse-failed" : ""}`}
                          style={{ borderLeftColor: modelColor(modelName) }}
                        >
                          <span className="model-label-name">{modelName}</span>
                          {result && !parsed ? (
                            <span className="model-label-error">Failed to parse</span>
                          ) : isDenseMode && result?.action_label ? (
                            <div className="dense-label">
                              <span className="verb-tag">{result.action_label.verb}</span>
                              <span className="noun-tag">{result.action_label.noun}</span>
                            </div>
                          ) : (
                            <span className="model-label-text">
                              {result?.primary_action || "\u2014"}
                            </span>
                          )}
                          {confidenceLabel ? (
                            <span className="model-label-confidence">{confidenceLabel}</span>
                          ) : null}
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="empty-state">No segment comparisons are available for this run.</p>
        )}
      </section>

      {isDenseMode && taxonomyAgreement ? (
        <section className="visual-card dense-metrics-section">
          <div className="section-heading">
            <p className="section-eyebrow">Taxonomy Agreement</p>
            <h2>Structured verb+noun matching</h2>
            <p className="chart-desc">
              These averages summarize how often the models matched on the verb, noun, and full
              action taxonomy labels.
            </p>
          </div>
          <div className="dense-metric-cards">
            <article className="metric-card">
              <span className="metric-label">Verb agreement</span>
              <span className="metric-value">{formatPercent(taxonomyAgreement.verb)}</span>
            </article>
            <article className="metric-card">
              <span className="metric-label">Noun agreement</span>
              <span className="metric-value">{formatPercent(taxonomyAgreement.noun)}</span>
            </article>
            <article className="metric-card">
              <span className="metric-label">Action agreement</span>
              <span className="metric-value">{formatPercent(taxonomyAgreement.action)}</span>
            </article>
          </div>
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
