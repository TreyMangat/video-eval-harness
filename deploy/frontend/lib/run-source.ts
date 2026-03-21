import { fetchRun, fetchRuns, fetchSegmentMedia, isInteractiveMode } from "./api-client";
import {
  listArtifactRuns,
  loadArtifactRun,
  loadArtifactSegmentMedia,
} from "./local-runs";
import type { RunListItem, RunPayload, SegmentMedia } from "./types";

export async function listRuns(dataDir?: string): Promise<RunListItem[]> {
  if (isInteractiveMode()) {
    try {
      return await fetchRuns();
    } catch {
      return [];
    }
  }

  return listArtifactRuns(dataDir);
}

export async function loadRun(
  runId: string,
  dataDir?: string
): Promise<RunPayload | null> {
  if (isInteractiveMode()) {
    try {
      return await fetchRun(runId);
    } catch {
      return null;
    }
  }

  return loadArtifactRun(runId, dataDir);
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
    } catch {
      return null;
    }
  }

  return loadArtifactSegmentMedia(runId, segmentId, variantId, dataDir);
}
