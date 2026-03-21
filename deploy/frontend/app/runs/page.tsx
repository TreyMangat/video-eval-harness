import { TopNav } from "../../components/navigation";
import { RunsTable } from "../../components/runs-table";
import {
  buildCoreComparisonRows,
  displayRunName,
  displayVideoName,
  getSweepData,
} from "../../lib/analysis";
import { listRuns, loadRun } from "../../lib/run-source";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function readFirst(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

export default async function RunsPage({
  searchParams,
}: {
  searchParams: Promise<{ dataDir?: string | string[] }>;
}) {
  const resolvedSearchParams = await searchParams;
  const dataDir = readFirst(resolvedSearchParams.dataDir) ?? process.env.VBENCH_RUNS_DIR;
  const runs = await listRuns(dataDir);
  const rows = (
    await Promise.all(
      runs.map(async (run) => {
        const payload = await loadRun(run.run_id, dataDir);
        const comparisonRows = payload
          ? buildCoreComparisonRows(payload, getSweepData(payload))
          : [];
        const bestRow = comparisonRows[0] ?? null;

        return {
          run_id: run.run_id,
          display_name: displayRunName(run.run_id, run.created_at),
          created_at: run.created_at,
          models: run.models,
          video_names: run.video_ids.map((videoId) => displayVideoName(videoId)),
          best_agreement: bestRow?.agreement ?? null,
          best_model_name: bestRow?.model_name ?? null,
          data_dir: dataDir,
        };
      })
    )
  ).sort(
    (left, right) =>
      new Date(right.created_at).getTime() - new Date(left.created_at).getTime()
  );

  return (
    <main className="analysis-shell">
      <TopNav active="runs" />
      {rows.length === 0 ? (
        <section className="visual-card">
          <div className="section-heading">
            <p className="section-eyebrow">Run Index</p>
            <h2 className="run-title">Which run do you want to inspect?</h2>
          </div>
          <p className="empty-state">No exported runs were found.</p>
        </section>
      ) : (
        <RunsTable rows={rows} />
      )}
    </main>
  );
}
