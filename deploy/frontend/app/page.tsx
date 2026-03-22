import { redirect } from "next/navigation";

import { AggregateDashboard } from "../components/aggregate-dashboard";
import { listRuns, loadRun } from "../lib/run-source";
import type { RunPayload } from "../lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const AGGREGATE_HISTORY_STEP = 10;
const AGGREGATE_THRESHOLD = 20;

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

function parsePositiveInteger(value: string | undefined): number | null {
  if (!value) {
    return null;
  }
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return null;
  }
  return parsed;
}

function resolveLoadLimit(totalRuns: number, requestedLimit: number | null): number {
  if (totalRuns === 0) {
    return 0;
  }
  if (totalRuns <= AGGREGATE_THRESHOLD) {
    return totalRuns;
  }
  const baseline = requestedLimit ?? AGGREGATE_HISTORY_STEP;
  return Math.min(totalRuns, Math.max(AGGREGATE_HISTORY_STEP, baseline));
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
    limit?: string | string[];
  }>;
}) {
  const resolvedSearchParams = await searchParams;
  const dataDir = readFirst(resolvedSearchParams.dataDir) ?? process.env.VBENCH_RUNS_DIR;
  const legacyRunId = readFirst(resolvedSearchParams.run);

  if (legacyRunId) {
    redirect(buildHref(`/report/${legacyRunId}`, { dataDir }));
  }

  const runList = await listRuns(dataDir);
  const requestedLimit = parsePositiveInteger(readFirst(resolvedSearchParams.limit));
  const loadLimit = resolveLoadLimit(runList.length, requestedLimit);
  const loadedRuns = await loadAggregateRuns(
    runList.slice(0, loadLimit).map((run) => run.run_id),
    dataDir
  );
  const nextLoadCount =
    loadLimit < runList.length ? Math.min(runList.length, loadLimit + AGGREGATE_HISTORY_STEP) : null;

  return (
    <AggregateDashboard
      runs={loadedRuns}
      runList={runList}
      dataDir={dataDir}
      basePath="/"
      loadedRunCount={loadedRuns.length}
      nextLoadCount={nextLoadCount}
    />
  );
}
