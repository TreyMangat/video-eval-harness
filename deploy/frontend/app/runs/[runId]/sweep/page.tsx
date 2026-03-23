import { notFound } from "next/navigation";

import {
  AgreementMatrixCard,
  RunMetadataCard,
  StabilityTableCard,
  VariantHeatmapCard,
} from "../../../../components/analysis-panels";
import { TopNav } from "../../../../components/navigation";
import {
  buildParseSuccessMatrix,
  formatPercent,
  getSweepData,
  runBreadcrumb,
} from "../../../../lib/analysis";
import { loadRun } from "../../../../lib/run-source";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function readFirst(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

export default async function RunSweepPage({
  params,
  searchParams,
}: {
  params: Promise<{ runId: string }>;
  searchParams: Promise<{ dataDir?: string | string[] }>;
}) {
  const { runId } = await params;
  const resolvedSearchParams = await searchParams;
  const dataDir = readFirst(resolvedSearchParams.dataDir) ?? process.env.VBENCH_RUNS_DIR;
  const run = await loadRun(runId, dataDir);

  if (!run) {
    notFound();
  }

  const sweepData = getSweepData(run);
  const runModels = Array.isArray(run.models) ? run.models : [];
  const runSegments = Array.isArray(run.segments) ? run.segments : [];
  const sweepVariants = Array.isArray(sweepData?.variants) ? sweepData.variants : [];
  const agreementByVariant =
    sweepData?.agreement_by_variant && typeof sweepData.agreement_by_variant === "object"
      ? sweepData.agreement_by_variant
      : {};
  const stableLeader =
    (sweepData?.stability ?? [])
      .slice()
      .sort(
        (left, right) =>
          right.self_agreement - left.self_agreement ||
          left.model_name.localeCompare(right.model_name)
      )[0] ?? null;

  return (
    <main className="analysis-shell">
      <TopNav active="runs" />

      <section className="visual-card">
        <div className="section-heading">
          <p className="section-eyebrow">Sweep</p>
          <h1 className="run-title">Does changing frame count change the winner?</h1>
          <p className="chart-desc">
            Compare parse success, stability, and pairwise agreement across every extraction
            variant in the run.
          </p>
        </div>
        <RunMetadataCard
          title="Run Context"
          runId={run.run_id}
          createdAt={run.config?.created_at ?? ""}
          videoLabel=""
          promptVersion={run.config?.prompt_version ?? ""}
          models={runModels}
          segments={runSegments.length}
          compact
          compactText={runBreadcrumb(run)}
        />
      </section>

      {sweepData ? (
        <>
          <VariantHeatmapCard
            title="Which variants parse most reliably?"
            description="Parse-success matrix across all extraction variants."
            models={runModels}
            variants={sweepVariants}
            matrix={buildParseSuccessMatrix(sweepData, runModels, sweepVariants)}
          />

          {stableLeader ? (
            <p className="page-summary">
              {stableLeader.model_name} is the most stable model, giving the same answer{" "}
              {formatPercent(stableLeader.self_agreement)} of the time regardless of frame count.
            </p>
          ) : null}

          <StabilityTableCard
            title="Which model stays most consistent across variants?"
            description="Self-agreement and ranking stability across the sweep."
            stability={sweepData.stability ?? []}
          />

          {Object.entries(agreementByVariant)
            .sort(([left], [right]) => left.localeCompare(right))
            .map(([variant, matrix]) => (
              <AgreementMatrixCard
                key={variant}
                title={`How much do models agree in ${variant}?`}
                matrix={matrix}
              />
            ))}
        </>
      ) : (
        <section className="visual-card">
          <div className="section-heading">
            <h3>No sweep data</h3>
            <p className="chart-desc">
              This run does not include a sweep summary, so there is nothing to compare yet.
            </p>
          </div>
        </section>
      )}
    </main>
  );
}
