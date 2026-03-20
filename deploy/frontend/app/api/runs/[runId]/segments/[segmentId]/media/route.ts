import { NextResponse } from "next/server";

import { fetchBackend, readBackendJson } from "../../../../../../../lib/backend";

export async function GET(
  request: Request,
  context: { params: Promise<{ runId: string; segmentId: string }> }
) {
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
