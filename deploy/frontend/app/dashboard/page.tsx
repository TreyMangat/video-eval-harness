import { DashboardSummary } from "../../components/dashboard-summary";
import { listRuns, loadRun } from "../../lib/run-source";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function readFirst(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

export default async function DashboardPage({
  searchParams,
}: {
  searchParams: Promise<{ run?: string | string[]; dataDir?: string | string[] }>;
}) {
  const resolvedSearchParams = await searchParams;
  const dataDir = readFirst(resolvedSearchParams.dataDir) ?? process.env.VBENCH_RUNS_DIR;
  const runs = await listRuns(dataDir);
  const selectedRunId = readFirst(resolvedSearchParams.run) ?? runs[0]?.run_id;
  const selectedRun = selectedRunId ? await loadRun(selectedRunId, dataDir) : null;
  const run = selectedRun ?? (runs[0] ? await loadRun(runs[0].run_id, dataDir) : null);

  return <DashboardSummary runs={runs} run={run} dataDir={dataDir} basePath="/dashboard" />;
}
