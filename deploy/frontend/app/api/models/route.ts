import { NextResponse } from "next/server";

import { fetchBackend, readBackendJson } from "../../../lib/backend";

export async function GET() {
  try {
    const response = await fetchBackend("models");
    const data = await readBackendJson(response);
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to load models" },
      { status: 500 }
    );
  }
}
