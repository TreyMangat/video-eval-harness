import Link from "next/link";

import {
  AgreementMatrixCard,
  MetricDeltaTableCard,
  RunMetadataCard,
  VariantHeatmapCard,
} from "../../components/analysis-panels";
import {
  buildModelDeltaRows,
  buildSweepDeltaMatrix,
  getSweepData,
} from "../../lib/analysis";
import { listArtifactRuns, loadArtifactRun } from "../../lib/local-runs";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function readFirst(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

function defaultCompareRunIds(runIds: string[]): { runA?: string; runB?: string } {
  return {
    runA: runIds[0],
    runB: runIds.find((runId) => runId !== runIds[0]),
  };
}

export default async function CompareRunsPage({
  searchParams,
}: {
  searchParams: Promise<{
    runA?: string | string[];
    runB?: string | string[];
    dataDir?: string | string[];
  }>;
}) {
  const resolvedSearchParams = await searchParams;
  const dataDir = readFirst(resolvedSearchParams.dataDir);
  const availableRuns = await listArtifactRuns(dataDir);
  const defaultRuns = defaultCompareRunIds(availableRuns.map((run) => run.run_id));
  const runAId = readFirst(resolvedSearchParams.runA) ?? defaultRuns.runA;
  const runBId = readFirst(resolvedSearchParams.runB) ?? defaultRuns.runB;

  const leftRun = runAId ? await loadArtifactRun(runAId, dataDir) : null;
  const rightRun = runBId ? await loadArtifactRun(runBId, dataDir) : null;
  const leftSweep = leftRun ? getSweepData(leftRun) : null;
  const rightSweep = rightRun ? getSweepData(rightRun) : null;
  const deltaRows =
    leftRun && rightRun ? buildModelDeltaRows(leftRun, rightRun) : [];
  const sweepDelta =
    leftRun && rightRun
      ? buildSweepDeltaMatrix(leftSweep, rightSweep, leftRun.models, rightRun.models)
      : null;

  return (
    <main className="analysis-shell">
      <div className="analysis-topbar">
        <div>
          <p className="eyebrow">Run Comparison</p>
          <h1 className="analysis-title">VBench Compare</h1>
          <p className="helper-copy">
            Side-by-side view of two exported runs, including sweep deltas when both runs have
            variant data.
          </p>
        </div>
        <div className="analysis-actions">
          <Link href="/" className="ghost-btn">
            Back to Dashboard
          </Link>
          {leftRun ? (
            <Link href={`/report/${leftRun.run_id}`} className="ghost-btn">
              Report A
            </Link>
          ) : null}
          {rightRun ? (
            <Link href={`/report/${rightRun.run_id}`} className="ghost-btn">
              Report B
            </Link>
          ) : null}
        </div>
      </div>

      <section className="raw-section">
        <h3>Select Runs</h3>
        <p className="chart-desc">
          Compare two exported runs using the same JSON artifact loading path as the dashboard.
        </p>
        <form method="get" className="compare-form">
          <label className="field">
            <span>Run A</span>
            <select name="runA" defaultValue={runAId}>
              {availableRuns.map((run) => (
                <option key={`left-${run.run_id}`} value={run.run_id}>
                  {run.run_id}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Run B</span>
            <select name="runB" defaultValue={runBId}>
              {availableRuns.map((run) => (
                <option key={`right-${run.run_id}`} value={run.run_id}>
                  {run.run_id}
                </option>
              ))}
            </select>
          </label>
          {dataDir ? <input type="hidden" name="dataDir" value={dataDir} /> : null}
          <button type="submit" className="primary-btn">
            Compare Runs
          </button>
        </form>
      </section>

      {!leftRun || !rightRun ? (
        <section className="empty-hero">
          <h2>Pick two runs to compare</h2>
          <p>Once both runs are selected, the page will render metric deltas and sweep changes.</p>
        </section>
      ) : (
        <>
          <div className="analysis-grid two-up">
            <RunMetadataCard
              title="Run A"
              runId={leftRun.run_id}
              createdAt={leftRun.config.created_at}
              videoLabel={
                leftRun.videos[0]?.filename ||
                leftRun.videos[0]?.video_id ||
                leftRun.config.video_ids[0] ||
                "Unknown video"
              }
              promptVersion={leftRun.config.prompt_version}
              models={leftRun.models}
              segments={leftRun.segments.length}
              extraRows={[
                {
                  label: "Sweep Variants",
                  value: leftSweep ? String(leftSweep.variants.length) : "No sweep data",
                },
              ]}
            />
            <RunMetadataCard
              title="Run B"
              runId={rightRun.run_id}
              createdAt={rightRun.config.created_at}
              videoLabel={
                rightRun.videos[0]?.filename ||
                rightRun.videos[0]?.video_id ||
                rightRun.config.video_ids[0] ||
                "Unknown video"
              }
              promptVersion={rightRun.config.prompt_version}
              models={rightRun.models}
              segments={rightRun.segments.length}
              extraRows={[
                {
                  label: "Sweep Variants",
                  value: rightSweep ? String(rightSweep.variants.length) : "No sweep data",
                },
              ]}
            />
          </div>

          <MetricDeltaTableCard
            title="Per-Model Metric Deltas"
            description="Green means the right-hand run improved; latency improvements are green when they go down."
            rows={deltaRows}
            leftLabel="Run A"
            rightLabel="Run B"
          />

          <div className="analysis-grid two-up">
            <AgreementMatrixCard
              title={`Agreement Matrix: ${leftRun.run_id}`}
              matrix={leftRun.agreement}
            />
            <AgreementMatrixCard
              title={`Agreement Matrix: ${rightRun.run_id}`}
              matrix={rightRun.agreement}
            />
          </div>

          {sweepDelta ? (
            <div className="analysis-grid three-up">
              <VariantHeatmapCard
                title={`Sweep Heatmap: ${leftRun.run_id}`}
                description="Model x variant parse success for Run A."
                models={sweepDelta.models}
                variants={sweepDelta.variants}
                matrix={sweepDelta.left_matrix}
              />
              <VariantHeatmapCard
                title={`Sweep Heatmap: ${rightRun.run_id}`}
                description="Model x variant parse success for Run B."
                models={sweepDelta.models}
                variants={sweepDelta.variants}
                matrix={sweepDelta.right_matrix}
              />
              <VariantHeatmapCard
                title="Sweep Change"
                description="Run B minus Run A."
                models={sweepDelta.models}
                variants={sweepDelta.variants}
                matrix={sweepDelta.delta_matrix}
                deltaMode
              />
            </div>
          ) : (
            <section className="agreement-section">
              <h3>Sweep Comparison</h3>
              <p className="chart-desc">
                One or both selected runs do not have sweep exports, so the heatmap change view is
                unavailable.
              </p>
            </section>
          )}
        </>
      )}
    </main>
  );
}
