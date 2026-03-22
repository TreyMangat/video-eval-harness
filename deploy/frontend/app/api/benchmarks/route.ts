import { NextResponse } from "next/server";

import { fetchBackend, readBackendJson } from "../../../lib/backend";

export const runtime = "nodejs";

export async function POST(request: Request) {
  try {
    const body = await request.formData();
    const response = await fetchBackend("benchmark", {
      method: "POST",
      body,
    });
    const data = await readBackendJson(response);
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Failed to submit benchmark" },
      { status: 500 }
    );
  }
}
