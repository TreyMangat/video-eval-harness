import {
  buildCostBreakdown,
  buildCoreComparisonRows,
  displayRunName,
  getSweepData,
} from "../lib/analysis";
import type { RunListItem, RunPayload } from "../lib/types";
import { DashboardVisualsClient } from "./dashboard-visuals-client";
import { TopNav } from "./navigation";
import { UploadZone } from "./upload-zone";

function runSummaryLine(run: RunPayload, isSweep: boolean): string {
  const videoCount = run.config.video_ids.length || run.videos.length || 0;
  const modelCount = run.models.length;
  return [
    `${modelCount} ${modelCount === 1 ? "model" : "models"}`,
    `${videoCount} ${videoCount === 1 ? "video" : "videos"}`,
    isSweep ? "sweep" : "benchmark",
  ].join(" \u00b7 ");
}

export function DashboardSummary({
  runs,
  run,
  dataDir,
  basePath = "/",
}: {
  runs: RunListItem[];
  run: RunPayload | null;
  dataDir?: string;
  basePath?: string;
}) {
  const sweepData = run ? getSweepData(run) : null;
  const rows = run ? buildCoreComparisonRows(run, sweepData) : [];
  const costBreakdown = run ? buildCostBreakdown(run, sweepData) : null;

  return (
    <main className="analysis-shell">
      <TopNav
        active="dashboard"
        runSelector={
          run
            ? {
                runs,
                selectedRunId: run.run_id,
                dataDir,
                basePath,
              }
            : undefined
        }
      />

      <UploadZone />

      {run ? (
        <div id="dashboard-results" className="dashboard-results-anchor">
          <section className="visual-card dashboard-intro run-summary-card dashboard-section-card">
            <div className="run-summary-text">
              <p className="run-summary-primary">
                {displayRunName(run.run_id, run.config.created_at)}
                {" \u00b7 "}
                {runSummaryLine(run, Boolean(sweepData))}
              </p>
            </div>
          </section>

          <DashboardVisualsClient
            run={run}
            rows={rows}
            costBreakdown={costBreakdown}
            sweepData={sweepData}
          />
        </div>
      ) : (
        <section id="dashboard-results" className="visual-card">
          <p className="section-eyebrow">Results</p>
          <h2 className="run-title">Which run should you inspect?</h2>
          <p className="dashboard-copy">
            No benchmark runs found. Upload a video to get started.
          </p>
        </section>
      )}
    </main>
  );
}
