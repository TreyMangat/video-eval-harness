import { NextResponse } from "next/server";

import { fetchBackend, readBackendJson } from "../../../../../../../lib/backend";
import { loadArtifactSegmentMedia } from "../../../../../../../lib/local-runs";

export const runtime = "nodejs";

export async function GET(
  request: Request,
  context: { params: Promise<{ runId: string; segmentId: string }> }
) {
  const url = new URL(request.url);
  const dataDir = url.searchParams.get("dataDir") ?? undefined;

  try {
    const { runId, segmentId } = await context.params;
    const variantId = url.searchParams.get("variantId");
    const localMedia = await loadArtifactSegmentMedia(runId, segmentId, variantId, dataDir);
    if (localMedia) {
      return NextResponse.json(localMedia, { status: 200 });
    }
  } catch {
    // Fall through to backend mode.
  }

  try {
    const { runId, segmentId } = await context.params;
    const response = await fetchBackend(`runs/${runId}/segments/${segmentId}/media`);
    const data = await readBackendJson(response);
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to load segment media" },
      { status: 500 }
    );
  }
}
