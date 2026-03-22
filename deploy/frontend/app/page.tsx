import { redirect } from "next/navigation";

import { AggregateDashboard } from "../components/aggregate-dashboard";
import { listRuns, loadRun } from "../lib/run-source";
import type { RunPayload } from "../lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

function readFirst(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

function buildHref(pathname: string, query: Record<string, string | undefined>): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value) {
      params.set(key, value);
    }
  }
  const suffix = params.toString();
  return suffix ? `${pathname}?${suffix}` : pathname;
}

async function loadAggregateRuns(runIds: string[], dataDir?: string): Promise<RunPayload[]> {
  const settledRuns = await Promise.allSettled(
    runIds.map(async (runId) => loadRun(runId, dataDir))
  );

  return settledRuns.flatMap((result, index) => {
    if (result.status === "fulfilled" && result.value) {
      return [result.value];
    }

    const failedRunId = runIds[index];
    console.error(`Failed to load aggregate run ${failedRunId}:`, result.status === "rejected" ? result.reason : "Run not found");
    return [];
  });
}

export default async function HomePage({
  searchParams,
}: {
  searchParams: Promise<{
    run?: string | string[];
    dataDir?: string | string[];
  }>;
}) {
  const resolvedSearchParams = await searchParams;
  const dataDir = readFirst(resolvedSearchParams.dataDir) ?? process.env.VBENCH_RUNS_DIR;
  const legacyRunId = readFirst(resolvedSearchParams.run);

  if (legacyRunId) {
    redirect(buildHref(`/report/${legacyRunId}`, { dataDir }));
  }

  const runList = await listRuns(dataDir);
  const loadedRuns = await loadAggregateRuns(
    runList.map((run) => run.run_id),
    dataDir
  );

  return (
    <AggregateDashboard
      runs={loadedRuns}
      runList={runList}
      dataDir={dataDir}
    />
  );
}
