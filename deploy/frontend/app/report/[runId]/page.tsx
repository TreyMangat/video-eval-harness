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
  getRunVideoLabel,
  getSweepData,
  modelColor,
  runBreadcrumb,
  selectFeaturedVariant,
  selectSampleSegments,
} from "../../../lib/analysis";
import { getRunType } from "../../../lib/run-type";
import { loadRun } from "../../../lib/run-source";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function readFirst(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
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
  const rows = buildCoreComparisonRows(run, sweepData);
  const winner = bestOverallModel(rows);
  const bestValue = bestValueModel(rows, run.segments.length || 1);
  const fastest = fastestModel(rows);
  const featuredVariant = selectFeaturedVariant(sweepData);
  const samples = selectSampleSegments(run, featuredVariant, 3);
  const runLabel = displayRunName(run.run_id, run.config.created_at);
  const runType = getRunType(run);
  const parseSuccessMatrix = sweepData
    ? buildParseSuccessMatrix(sweepData, run.models, sweepData.variants)
    : null;

  return (
    <main className="analysis-shell report-page">
      <div className="no-print">
        <TopNav active="runs" />
      </div>

      <section className="visual-card report-verdict-card">
        <p className="section-eyebrow">Printable Summary</p>
        <h1 className="report-verdict">{verdictSentence(runLabel, winner, rows, sweepData)}</h1>
        <div className="report-subhead-row">
          <p className="report-subhead">
            {runLabel}
            {featuredVariant ? ` \u00b7 sample segments from ${featuredVariant}` : ""}
          </p>
          <RunTypeBadge run={run} />
        </div>
        {runType === "accuracy_test" ? (
          <p className="report-accuracy-note">
            This run was scored against user-provided ground truth labels.
          </p>
        ) : null}
      </section>

      <section className="hero-grid report-hero-grid">
        {winner ? (
          <HeroSummaryCard
            label="Winner"
            modelName={winner.model_name}
            accentColor={modelColor(winner.model_name)}
            heroValue={formatPercent(winner.agreement)}
            secondary={`${formatPercent(winner.parse_rate)} parse \u00b7 ${formatMoney(winner.total_cost)}`}
          />
        ) : null}
        {bestValue ? (
          <HeroSummaryCard
            label="Best Value"
            modelName={bestValue.model_name}
            accentColor={modelColor(bestValue.model_name)}
            heroValue={
              bestValue.total_cost === 0
                ? "\u221e"
                : bestValue.total_cost && bestValue.total_cost > 0
                  ? `${((bestValue.agreement ?? 0) / (bestValue.total_cost / Math.max(run.segments.length, 1))).toFixed(1)}x`
                  : "\u2014"
            }
            secondary={`${formatMoney(bestValue.total_cost)} total \u00b7 ${formatPercent(bestValue.agreement)} agreement`}
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
      </section>

      <AgreementMatrixCard
        title="How often do models agree with each other?"
        description="This is the one visual to carry into a discussion: it shows how tightly the models cluster."
        matrix={run.agreement}
      />

      <SegmentComparisonSamplesCard
        title="Sample Segment Comparisons"
        description={
          featuredVariant
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
          models={run.models}
          segments={run.segments.length}
          compact
          compactText={runBreadcrumb(run)}
        />

        {sweepData && parseSuccessMatrix ? (
          <VariantHeatmapCard
            title="Model x Variant Parse Success"
            description="Parse success heatmap from the run's exported sweep summary."
            models={run.models}
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
