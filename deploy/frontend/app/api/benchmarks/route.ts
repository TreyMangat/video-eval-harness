import { NextResponse } from "next/server";

import { fetchBackend, readBackendJson } from "../../../lib/backend";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const response = await fetchBackend("benchmarks", {
      method: "POST",
      body: JSON.stringify(body),
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
