import {
  AgreementMatrixCard,
  MetricDeltaTableCard,
} from "../../components/analysis-panels";
import { TopNav } from "../../components/navigation";
import {
  buildModelDeltaRows,
  displayRunName,
} from "../../lib/analysis";
import { listRuns, loadRun } from "../../lib/run-source";

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
  const dataDir = readFirst(resolvedSearchParams.dataDir) ?? process.env.VBENCH_RUNS_DIR;
  const availableRuns = await listRuns(dataDir);
  const defaultRuns = defaultCompareRunIds(availableRuns.map((run) => run.run_id));
  const runAId = readFirst(resolvedSearchParams.runA) ?? defaultRuns.runA;
  const runBId = readFirst(resolvedSearchParams.runB) ?? defaultRuns.runB;

  const leftRun = runAId ? await loadRun(runAId, dataDir) : null;
  const rightRun = runBId ? await loadRun(runBId, dataDir) : null;
  const deltaRows = leftRun && rightRun ? buildModelDeltaRows(leftRun, rightRun) : [];
  const leftLabel = leftRun
    ? displayRunName(leftRun.run_id, leftRun.config.created_at)
    : "Left run";
  const rightLabel = rightRun
    ? displayRunName(rightRun.run_id, rightRun.config.created_at)
    : "Right run";

  return (
    <main className="analysis-shell">
      <TopNav active="compare" />

      <section className="visual-card">
        <div className="section-heading">
          <p className="section-eyebrow">Compare</p>
          <h1 className="run-title">What changed between these runs?</h1>
          <p className="chart-desc">
            Pick a baseline and a comparison run to see who improved, who regressed, and whether
            the agreement pattern changed.
          </p>
        </div>

        <form method="get" className="compare-form">
          <label className="field">
            <span>Baseline run</span>
            <select name="runA" defaultValue={runAId}>
              {availableRuns.map((run) => (
                <option key={`left-${run.run_id}`} value={run.run_id}>
                  {displayRunName(run.run_id, run.created_at)}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>Comparison run</span>
            <select name="runB" defaultValue={runBId}>
              {availableRuns.map((run) => (
                <option key={`right-${run.run_id}`} value={run.run_id}>
                  {displayRunName(run.run_id, run.created_at)}
                </option>
              ))}
            </select>
          </label>
          {dataDir ? <input type="hidden" name="dataDir" value={dataDir} /> : null}
          <div className="compare-submit-row">
            <button type="submit" className="primary-btn">
              Update comparison
            </button>
          </div>
        </form>
      </section>

      {!leftRun || !rightRun ? (
        <section className="visual-card empty-hero">
          <h2>Pick two runs to compare</h2>
          <p>Once both runs are selected, this page will show the metric deltas and agreement matrices.</p>
        </section>
      ) : (
        <>
          <MetricDeltaTableCard
            title="How did each model change?"
            description="Green marks an improvement. For latency, green means the model got faster."
            rows={deltaRows}
            leftLabel={leftLabel}
            rightLabel={rightLabel}
          />

          <div className="analysis-grid two-up">
            <AgreementMatrixCard
              title={leftLabel}
              description="Pairwise agreement for the baseline run."
              matrix={leftRun.agreement}
            />
            <AgreementMatrixCard
              title={rightLabel}
              description="Pairwise agreement for the comparison run."
              matrix={rightRun.agreement}
            />
          </div>
        </>
      )}
    </main>
  );
}
