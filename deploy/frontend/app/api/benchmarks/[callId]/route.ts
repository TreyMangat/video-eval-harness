import { NextResponse } from "next/server";

import { fetchBackend, readBackendJson } from "../../../../lib/backend";

export const runtime = "nodejs";

export async function GET(
  request: Request,
  context: { params: Promise<{ callId: string }> }
) {
  try {
    const { callId } = await context.params;
    const response = await fetchBackend(`jobs/${callId}`);
    const data = await readBackendJson(response);
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to load benchmark job" },
      { status: 500 }
    );
  }
}
