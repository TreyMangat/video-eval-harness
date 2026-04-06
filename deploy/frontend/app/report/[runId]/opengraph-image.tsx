import { ImageResponse } from "next/og";
import {
  bestOverallModel,
  bestValueModel,
  buildCoreComparisonRows,
  displayRunName,
  fastestModel,
  formatMoney,
  formatPercent,
  getSweepData,
} from "../../../lib/analysis";
import { loadRun } from "../../../lib/run-source";

export const runtime = "nodejs";
export const contentType = "image/png";
export const size = { width: 1200, height: 630 };
export const alt = "VBench benchmark run";

function shortModelName(name: string): string {
  return name
    .replace(/^(openai|google|anthropic|meta|qwen|alibaba|mistralai|x-ai)\//, "")
    .replace(/^(nvidia|novartis|deepseek)\//, "");
}

export default async function OgImage({
  params,
}: {
  params: Promise<{ runId: string }>;
}) {
  const { runId } = await params;
  const run = await loadRun(runId);

  if (!run) {
    return new ImageResponse(
      (
        <div
          style={{
            width: "100%",
            height: "100%",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: "12px",
            background: "#0a0a0f",
            color: "#94a3b8",
            fontFamily: "system-ui, sans-serif",
          }}
        >
          <div style={{ display: "flex", fontSize: "28px", fontWeight: 700, color: "#f8fafc" }}>
            VBench
          </div>
          <div style={{ display: "flex", fontSize: "36px" }}>Run not found</div>
        </div>
      ),
      { ...size }
    );
  }

  const sweepData = getSweepData(run);
  const rows = buildCoreComparisonRows(run, sweepData);
  const segmentCount = run.segments?.length ?? 0;

  const winner = bestOverallModel(rows);
  const value = bestValueModel(rows, segmentCount);
  const fastest = fastestModel(rows);

  const runLabel = displayRunName(run.run_id, run.config.created_at);
  const reportName = run.config.display_name?.trim() || runLabel;
  const modelCount = Object.keys(run.summaries ?? {}).length;
  const videoCount = run.config.video_ids?.length ?? 0;

  const winnerAccuracy = winner?.accuracy ?? winner?.llm_accuracy ?? winner?.agreement;
  const winnerStat = winnerAccuracy != null ? formatPercent(winnerAccuracy) : "-";
  const winnerCost = formatMoney(winner?.total_cost);

  const valueStat = value?.accuracy ?? value?.llm_accuracy ?? value?.agreement;
  const fastestLatency = fastest?.avg_latency_ms;

  // Build model cards data - up to 3, no duplicates
  const cards: Array<{ label: string; name: string; stat: string; cost: string; color: string }> = [];
  const usedModels = new Set<string>();

  if (winner) {
    usedModels.add(winner.model_name);
    cards.push({
      label: "WINNER",
      name: shortModelName(winner.model_name),
      stat: winnerStat,
      cost: winnerCost,
      color: "#22c55e",
    });
  }
  if (value && !usedModels.has(value.model_name)) {
    usedModels.add(value.model_name);
    cards.push({
      label: "BEST VALUE",
      name: shortModelName(value.model_name),
      stat: valueStat != null ? formatPercent(valueStat) : "-",
      cost: formatMoney(value.total_cost),
      color: "#4c9aff",
    });
  }
  if (fastest && !usedModels.has(fastest.model_name)) {
    cards.push({
      label: "FASTEST",
      name: shortModelName(fastest.model_name),
      stat: fastestLatency != null ? `${Math.round(fastestLatency)} ms` : "-",
      cost: formatMoney(fastest.total_cost),
      color: "#f59e0b",
    });
  }

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: "52px 64px",
          background: "linear-gradient(145deg, #0a0a0f 0%, #0f1729 50%, #0a0a0f 100%)",
          color: "#f8fafc",
          fontFamily: "system-ui, sans-serif",
        }}
      >
        {/* Top bar */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ display: "flex", fontSize: "28px", fontWeight: 700, letterSpacing: "-0.02em" }}>
            VBench
          </div>
          <div
            style={{
              display: "flex",
              fontSize: "14px",
              fontWeight: 600,
              letterSpacing: "0.08em",
              padding: "6px 16px",
              borderRadius: "8px",
              background: "rgba(34, 197, 94, 0.15)",
              color: "#22c55e",
            }}
          >
            LIVE DASHBOARD
          </div>
        </div>

        {/* Run name + sub-stats */}
        <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
          <div style={{ display: "flex", fontSize: "44px", fontWeight: 700, letterSpacing: "-0.03em", lineHeight: 1.15 }}>
            {reportName}
          </div>
          <div style={{ display: "flex", gap: "8px", fontSize: "22px", color: "#94a3b8" }}>
            <div style={{ display: "flex" }}>{modelCount} models</div>
            <div style={{ display: "flex", color: "#475569" }}>·</div>
            <div style={{ display: "flex" }}>{videoCount} videos</div>
            <div style={{ display: "flex", color: "#475569" }}>·</div>
            <div style={{ display: "flex" }}>{segmentCount} segments</div>
          </div>
        </div>

        {/* Model cards */}
        <div style={{ display: "flex", gap: "20px" }}>
          {cards.map((card) => (
            <div
              key={card.label}
              style={{
                display: "flex",
                flexDirection: "column",
                gap: "8px",
                padding: "20px 24px",
                borderRadius: "14px",
                background: "rgba(17, 24, 39, 0.9)",
                borderLeft: `4px solid ${card.color}`,
                minWidth: "240px",
              }}
            >
              <div
                style={{
                  display: "flex",
                  fontSize: "13px",
                  fontWeight: 700,
                  letterSpacing: "0.08em",
                  color: card.color,
                }}
              >
                {card.label}
              </div>
              <div style={{ display: "flex", fontSize: "26px", fontWeight: 700, color: "#f8fafc", lineHeight: 1.2 }}>
                {card.name}
              </div>
              <div style={{ display: "flex", gap: "12px", fontSize: "18px", color: "#94a3b8" }}>
                <div style={{ display: "flex" }}>{card.stat}</div>
                <div style={{ display: "flex", color: "#475569" }}>·</div>
                <div style={{ display: "flex" }}>{card.cost}</div>
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div style={{ display: "flex", fontSize: "18px", color: "#64748b" }}>
          video-eval-harness-qu4m.vercel.app
        </div>
      </div>
    ),
    { ...size }
  );
}
