import { NextResponse } from "next/server";

import { fetchBackend, readBackendJson } from "../../../../lib/backend";
import { loadArtifactRun } from "../../../../lib/local-runs";

export const runtime = "nodejs";

export async function GET(
  request: Request,
  context: { params: Promise<{ runId: string }> }
) {
  try {
    const { runId } = await context.params;
    const localRun = await loadArtifactRun(runId);
    if (localRun) {
      return NextResponse.json(localRun, { status: 200 });
    }
  } catch {
    // Fall through to backend mode.
  }

  try {
    const { runId } = await context.params;
    const response = await fetchBackend(`runs/${runId}`);
    const data = await readBackendJson(response);
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to load run" },
      { status: 500 }
    );
  }
}
