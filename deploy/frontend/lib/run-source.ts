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
    } catch {
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
      return await fetchSegmentMedia(runId, segmentId, variantId);
    } catch {
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
