import { NextResponse } from "next/server";

import { fetchBackend, readBackendJson } from "../../../lib/backend";
import { DEFAULT_MODEL_CATALOG } from "../../../lib/model-catalog";

export const runtime = "nodejs";

export async function GET() {
  try {
    const response = await fetchBackend("models");
    const data = await readBackendJson(response);
    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    return NextResponse.json(
      { models: DEFAULT_MODEL_CATALOG, source: "default_catalog" },
      { status: 200 }
    );
  }
}
