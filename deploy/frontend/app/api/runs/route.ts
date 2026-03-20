import { NextResponse } from "next/server";

import { fetchBackend, readBackendJson } from "../../../lib/backend";
import { listArtifactRuns } from "../../../lib/local-runs";

export const runtime = "nodejs";

export async function GET() {
  try {
    const localRuns = await listArtifactRuns();
    if (localRuns.length > 0) {
      return NextResponse.json(localRuns, { status: 200 });
    }
  } catch {
    // Fall through to backend mode.
  }

  try {
    const response = await fetchBackend("runs");
    const data = await readBackendJson(response);
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      [],
      { status: 200 }
    );
  }
}
