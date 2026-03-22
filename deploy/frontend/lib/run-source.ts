import { fetchRun, fetchRuns, fetchSegmentMedia, isInteractiveMode } from "./api-client";
import {
  listArtifactRuns,
  loadArtifactRun,
  loadArtifactSegmentMedia,
} from "./local-runs";
import type { RunListItem, RunPayload, SegmentMedia } from "./types";

const LIVE_FETCH_ATTEMPTS = 10;
const LIVE_FETCH_DELAY_MS = 2_000;

async function wait(ms: number): Promise<void> {
  await new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

async function retryLiveFetch<T>(load: () => Promise<T>): Promise<T> {
  let lastError: unknown;

  for (let attempt = 0; attempt < LIVE_FETCH_ATTEMPTS; attempt += 1) {
    try {
      return await load();
    } catch (error) {
      lastError = error;
      if (attempt < LIVE_FETCH_ATTEMPTS - 1) {
        await wait(LIVE_FETCH_DELAY_MS);
      }
    }
  }

  throw lastError;
}

function sortRunsByDate(runs: RunListItem[]): RunListItem[] {
  return [...runs].sort(
    (left, right) =>
      new Date(right.created_at).getTime() - new Date(left.created_at).getTime()
  );
}

function mergeRuns(primary: RunListItem[], secondary: RunListItem[]): RunListItem[] {
  const merged = new Map<string, RunListItem>();
  for (const run of [...primary, ...secondary]) {
    if (!merged.has(run.run_id)) {
      merged.set(run.run_id, run);
    }
  }
  return sortRunsByDate([...merged.values()]);
}

export async function listRuns(dataDir?: string): Promise<RunListItem[]> {
  if (isInteractiveMode()) {
    const [liveRuns, staticRuns] = await Promise.allSettled([
      fetchRuns(),
      listArtifactRuns(dataDir),
    ]);

    const resolvedLiveRuns = liveRuns.status === "fulfilled" ? liveRuns.value : [];
    const resolvedStaticRuns = staticRuns.status === "fulfilled" ? staticRuns.value : [];

    if (liveRuns.status === "rejected") {
      console.error("Failed to load live runs:", liveRuns.reason);
    }
    if (staticRuns.status === "rejected") {
      console.error("Failed to load static runs:", staticRuns.reason);
    }

    return mergeRuns(resolvedLiveRuns, resolvedStaticRuns);
  }

  try {
    return await listArtifactRuns(dataDir);
  } catch (error) {
    console.error("Failed to load local runs:", error);
    return [];
  }
}

export async function loadRun(
  runId: string,
  dataDir?: string
): Promise<RunPayload | null> {
  if (isInteractiveMode()) {
    try {
      const localRun = await loadArtifactRun(runId, dataDir);
      if (localRun) {
        return localRun;
      }
    } catch (localError) {
      console.error(`Failed to load local run ${runId}:`, localError);
    }

    try {
      return await retryLiveFetch(() => fetchRun(runId));
    } catch (liveError) {
      console.error(`Failed to load live run ${runId}:`, liveError);
      return null;
    }
  }

  try {
    return await loadArtifactRun(runId, dataDir);
  } catch (error) {
    console.error(`Failed to load local run ${runId}:`, error);
    return null;
  }
}

export async function loadSegmentMedia(
  runId: string,
  segmentId: string,
  variantId?: string | null,
  dataDir?: string
): Promise<SegmentMedia | null> {
  if (isInteractiveMode()) {
    try {
      const localMedia = await loadArtifactSegmentMedia(runId, segmentId, variantId, dataDir);
      if (localMedia) {
        return localMedia;
      }
    } catch (localError) {
      console.error(`Failed to load local segment media for ${runId}/${segmentId}:`, localError);
    }

    try {
      return await retryLiveFetch(() => fetchSegmentMedia(runId, segmentId, variantId));
    } catch (liveError) {
      console.error(`Failed to load live segment media for ${runId}/${segmentId}:`, liveError);
      return null;
    }
  }

  try {
    return await loadArtifactSegmentMedia(runId, segmentId, variantId, dataDir);
  } catch (error) {
    console.error(`Failed to load segment media for ${runId}/${segmentId}:`, error);
    return null;
  }
}
