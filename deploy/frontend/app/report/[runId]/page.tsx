import Link from "next/link";
import { notFound } from "next/navigation";

import {
  AgreementMatrixCard,
  RunMetadataCard,
  SegmentComparisonSamplesCard,
  StabilityTableCard,
  VariantHeatmapCard,
} from "../../../components/analysis-panels";
import {
  buildParseSuccessMatrix,
  getSweepData,
  selectFeaturedVariant,
  selectSampleSegments,
} from "../../../lib/analysis";
import { loadArtifactRun } from "../../../lib/local-runs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function readFirst(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
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
  const run = await loadArtifactRun(runId, dataDir);

  if (!run) {
    notFound();
  }

  const sweepData = getSweepData(run);
  const featuredVariant = selectFeaturedVariant(sweepData);
  const samples = selectSampleSegments(run, featuredVariant, 3);
  const videoLabel =
    run.videos[0]?.filename || run.videos[0]?.video_id || run.config.video_ids[0] || "Unknown video";
  const parseSuccessMatrix = sweepData
    ? buildParseSuccessMatrix(sweepData, run.models, sweepData.variants)
    : null;

  return (
    <main className="analysis-shell report-page">
      <div className="analysis-topbar no-print">
        <div>
          <p className="eyebrow">Printable Run Summary</p>
          <h1 className="analysis-title">VBench Report</h1>
          <p className="helper-copy">
            Static summary view for sharing benchmark findings or saving to PDF.
          </p>
        </div>
        <div className="analysis-actions">
          <Link href="/" className="ghost-btn">
            Back to Dashboard
          </Link>
          <Link href={`/compare?runA=${run.run_id}`} className="ghost-btn">
            Compare This Run
          </Link>
        </div>
      </div>

      <section className="report-heading">
        <div>
          <p className="eyebrow">Run Snapshot</p>
          <h1 className="analysis-title">{run.run_id}</h1>
          <p className="report-copy">
            Shareable benchmark summary built from exported run artifacts.
            {featuredVariant ? ` Sample comparisons use the featured variant ${featuredVariant}.` : ""}
          </p>
        </div>
      </section>

      <div className="analysis-grid two-up">
        <RunMetadataCard
          title="Run Metadata"
          runId={run.run_id}
          createdAt={run.config.created_at}
          videoLabel={videoLabel}
          promptVersion={run.config.prompt_version}
          models={run.models}
          segments={run.segments.length}
          extraRows={
            sweepData
              ? [
                  { label: "Sweep Variants", value: String(sweepData.variants.length) },
                  { label: "Featured Variant", value: featuredVariant ?? "-" },
                ]
              : [{ label: "Sweep Variants", value: "No sweep data" }]
          }
        />

        {sweepData && parseSuccessMatrix ? (
          <VariantHeatmapCard
            title="Model x Variant Parse Success"
            description="Parse success heatmap from the run's exported sweep summary."
            models={run.models}
            variants={sweepData.variants}
            matrix={parseSuccessMatrix}
          />
        ) : (
          <section className="agreement-section">
            <h3>Model x Variant Parse Success</h3>
            <p className="chart-desc">
              This run does not include sweep data, so there is no variant heatmap to export.
            </p>
          </section>
        )}
      </div>

      {sweepData ? (
        <StabilityTableCard
          title="Stability Scores"
          description="Self-agreement and rank stability across the extraction variants in this run."
          stability={sweepData.stability}
        />
      ) : null}

      <AgreementMatrixCard
        title="Top-Level Agreement Matrix"
        description="Agreement across the run's primary-action labels."
        matrix={run.agreement}
      />

      <SegmentComparisonSamplesCard
        title="Sample Segment Comparisons"
        description={
          featuredVariant
            ? `Three representative segment comparisons from ${featuredVariant}.`
            : "Three representative segment comparisons from this run."
        }
        samples={samples}
      />
    </main>
  );
}
