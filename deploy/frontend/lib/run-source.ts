import { fetchRun, fetchRuns, fetchSegmentMedia, isInteractiveMode } from "./api-client";
import {
  listArtifactRuns,
  loadArtifactRun,
  loadArtifactSegmentMedia,
} from "./local-runs";
import type { RunListItem, RunPayload, SegmentMedia } from "./types";

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
      return await fetchRun(runId);
    } catch (liveError) {
      console.error(`Failed to load live run ${runId}:`, liveError);
      try {
        return await loadArtifactRun(runId, dataDir);
      } catch (localError) {
        console.error(`Failed to load local run ${runId}:`, localError);
        return null;
      }
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
      return await fetchSegmentMedia(runId, segmentId, variantId);
    } catch (liveError) {
      console.error(`Failed to load live segment media for ${runId}/${segmentId}:`, liveError);
      try {
        return await loadArtifactSegmentMedia(runId, segmentId, variantId, dataDir);
      } catch (localError) {
        console.error(`Failed to load local segment media for ${runId}/${segmentId}:`, localError);
        return null;
      }
    }
  }

  try {
    return await loadArtifactSegmentMedia(runId, segmentId, variantId, dataDir);
  } catch (error) {
    console.error(`Failed to load segment media for ${runId}/${segmentId}:`, error);
    return null;
  }
}
