import { notFound } from "next/navigation";

import { RunMetadataCard } from "../../../../components/analysis-panels";
import { TopNav } from "../../../../components/navigation";
import {
  buildCostBreakdown,
  displaySegmentName,
  formatMoney,
  getSweepData,
  runBreadcrumb,
} from "../../../../lib/analysis";
import { loadRun } from "../../../../lib/run-source";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function readFirst(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

export default async function RunCostPage({
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
  const costs = buildCostBreakdown(run, sweepData);
  const hasCapturedCost = run.results.some((result) => (result.estimated_cost ?? 0) > 0);
  const segmentLookup = new Map(run.segments.map((segment) => [segment.segment_id, segment] as const));

  return (
    <main className="analysis-shell">
      <TopNav active="runs" />

      <section className="visual-card">
        <div className="section-heading">
          <p className="section-eyebrow">Cost</p>
          <h1 className="run-title">Where did the money go?</h1>
          <p className="chart-desc">
            Start with the run total, then break the spend down by model, variant, and segment.
          </p>
        </div>
        <RunMetadataCard
          title="Run Context"
          runId={run.run_id}
          createdAt={run.config.created_at}
          videoLabel=""
          promptVersion={run.config.prompt_version}
          models={run.models}
          segments={run.segments.length}
          compact
          compactText={runBreadcrumb(run)}
        />
      </section>

      {!hasCapturedCost ? (
        <section className="visual-card cost-zero-banner">
          <p>Cost data was not captured for this run. Ensure the provider returns cost estimates.</p>
        </section>
      ) : (
        <>
          <section className="summary-grid">
            <article className="summary-card">
              <p className="card-label">Run Total</p>
              <p className="card-value">{formatMoney(costs.total_cost)}</p>
              <span className="card-sublabel">Summed from result-level estimated cost</span>
            </article>
            <article className="summary-card">
              <p className="card-label">Cost / Segment</p>
              <p className="card-value">
                {run.segments.length > 0 ? formatMoney(costs.total_cost / run.segments.length) : "-"}
              </p>
              <span className="card-sublabel">Average across {run.segments.length} segments</span>
            </article>
            <article className="summary-card">
              <p className="card-label">Costed Rows</p>
              <p className="card-value">
                {run.results.filter((result) => (result.estimated_cost ?? 0) > 0).length}
              </p>
              <span className="card-sublabel">Results with captured cost data</span>
            </article>
          </section>

          <div className="analysis-grid three-up">
            <section className="visual-card">
              <div className="section-heading">
                <h3>Cost by Model</h3>
              </div>
              <div className="table-scroll">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Total Cost</th>
                    </tr>
                  </thead>
                  <tbody>
                    {costs.by_model.map((row) => (
                      <tr key={row.model_name}>
                        <td>{row.model_name}</td>
                        <td>{formatMoney(row.total_cost)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="visual-card">
              <div className="section-heading">
                <h3>Cost by Variant</h3>
              </div>
              <div className="table-scroll">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Variant</th>
                      <th>Total Cost</th>
                    </tr>
                  </thead>
                  <tbody>
                    {costs.by_variant.map((row) => (
                      <tr key={row.variant_label}>
                        <td>{row.variant_label}</td>
                        <td>{formatMoney(row.total_cost)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="visual-card">
              <div className="section-heading">
                <h3>Cost by Segment</h3>
              </div>
              <div className="table-scroll">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Segment</th>
                      <th>Cost</th>
                    </tr>
                  </thead>
                  <tbody>
                    {costs.by_segment.map((row) => (
                      <tr key={row.segment_id}>
                        <td>
                          {segmentLookup.get(row.segment_id)
                            ? displaySegmentName(segmentLookup.get(row.segment_id)!)
                            : row.segment_id}
                        </td>
                        <td>{formatMoney(row.total_cost)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        </>
      )}
    </main>
  );
}
