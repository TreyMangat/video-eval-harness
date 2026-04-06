import { ImageResponse } from "next/og";
import { listRuns } from "../lib/run-source";

export const runtime = "nodejs";
export const contentType = "image/png";
export const size = { width: 1200, height: 630 };
export const alt = "VBench — Multi-model video benchmark";

export default async function OgImage() {
  let runCount = 3;
  let modelCount = 10;
  let videoCount = 32;

  try {
    const runs = await listRuns();
    if (runs.length > 0) {
      runCount = runs.length;
      const allModels = new Set<string>();
      for (const run of runs) {
        for (const model of run.models ?? []) {
          allModels.add(model);
        }
      }
      if (allModels.size > 0) modelCount = allModels.size;
    }
  } catch {
    // fall back to static counts
  }

  return new ImageResponse(
    (
      <div
        style={{
          width: "1200px",
          height: "630px",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: "60px 72px",
          backgroundColor: "#0a0a0f",
          color: "#f8fafc",
          fontFamily: "system-ui, sans-serif",
        }}
      >
        <div style={{ display: "flex", fontSize: 32, fontWeight: 700, letterSpacing: "-0.02em" }}>
          VBench
        </div>

        <div style={{ display: "flex", flexDirection: "column" }}>
          <div style={{ display: "flex", fontSize: 52, fontWeight: 700, letterSpacing: "-0.03em", lineHeight: 1.15 }}>
            Multi-model video benchmark harness
          </div>
          <div style={{ display: "flex", fontSize: 26, color: "#94a3b8", lineHeight: 1.5, marginTop: 20 }}>
            {`How ${modelCount} frontier AI models interpret the same video content.`}
          </div>
          <div style={{ display: "flex", gap: 32, marginTop: 24 }}>
            <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
              <div style={{ fontSize: 36, fontWeight: 700, color: "#f59e0b" }}>{String(modelCount)}</div>
              <div style={{ fontSize: 22, color: "#94a3b8" }}>models</div>
            </div>
            <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
              <div style={{ fontSize: 36, fontWeight: 700, color: "#f59e0b" }}>{String(runCount)}</div>
              <div style={{ fontSize: 22, color: "#94a3b8" }}>benchmarks</div>
            </div>
            <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
              <div style={{ fontSize: 36, fontWeight: 700, color: "#f59e0b" }}>{String(videoCount)}</div>
              <div style={{ fontSize: 22, color: "#94a3b8" }}>videos</div>
            </div>
          </div>
        </div>

        <div style={{ display: "flex", fontSize: 20, color: "#64748b" }}>
          video-eval-harness-qu4m.vercel.app
        </div>
      </div>
    ),
    { ...size },
  );
}
