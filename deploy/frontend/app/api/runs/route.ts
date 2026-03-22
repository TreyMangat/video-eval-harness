import { NextResponse } from "next/server";

import { fetchBackend, readBackendJson } from "../../../lib/backend";
import { listArtifactRuns } from "../../../lib/local-runs";

export const runtime = "nodejs";

function sortRunsByDate(
  runs: Array<{
    run_id: string;
    created_at: string;
    models: string[];
    prompt_version: string;
    video_ids: string[];
  }>
) {
  return [...runs].sort(
    (left, right) =>
      new Date(right.created_at).getTime() - new Date(left.created_at).getTime()
  );
}

export async function GET(request: Request) {
  const dataDir = new URL(request.url).searchParams.get("dataDir") ?? undefined;
  let localRuns: Array<{
    run_id: string;
    created_at: string;
    models: string[];
    prompt_version: string;
    video_ids: string[];
  }> = [];

  try {
    localRuns = await listArtifactRuns(dataDir);
  } catch (error) {
    console.error("Failed to load static runs:", error);
  }

  try {
    const response = await fetchBackend("runs");
    const data = await readBackendJson(response);
    const merged = new Map<string, (typeof localRuns)[number]>();
    for (const run of localRuns) {
      merged.set(run.run_id, run);
    }
    for (const run of Array.isArray(data) ? data : []) {
      if (run && typeof run === "object" && typeof run.run_id === "string") {
        merged.set(run.run_id, run as (typeof localRuns)[number]);
      }
    }
    return NextResponse.json(sortRunsByDate([...merged.values()]), { status: 200 });
  } catch (error) {
    console.error("Failed to load live runs:", error);
    return NextResponse.json(sortRunsByDate(localRuns), { status: 200 });
  }
}
