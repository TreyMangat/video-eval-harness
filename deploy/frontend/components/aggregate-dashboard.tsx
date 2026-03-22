import Link from "next/link";

import {
  buildCoreComparisonRows,
  displayRunName,
  displayVideoName,
  formatDateTime,
  formatLatency,
  formatMoney,
  formatPercent,
  getSweepData,
} from "../lib/analysis";
import type { RunListItem, RunPayload } from "../lib/types";
import { TopNav } from "./navigation";

export interface AggregateStats {
  total_runs: number;
  total_videos: number;
  total_cost: number;
  models_tested: string[];
}

export interface AggregateModelRow {
  model_name: string;
  avg_agreement: number | null;
  avg_accuracy: number | null;
  avg_confidence: number | null;
  avg_latency_ms: number | null;
  total_cost: number;
  run_count: number;
}

function average(values: number[]): number | null {
  if (values.length === 0) {
    return null;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function buildHref(
  pathname: string,
  query: Record<string, string | number | undefined | null>
): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value != null && value !== "") {
      params.set(key, String(value));
    }
  }
  const suffix = params.toString();
  return suffix ? `${pathname}?${suffix}` : pathname;
}

function summarizeModelNames(models: string[]): string {
  if (models.length <= 3) {
    return models.join(", ");
  }
  return `${models.slice(0, 3).join(", ")} +${models.length - 3} more`;
}

function formatConfidence(value: number | null): string {
  if (value == null) {
    return "-";
  }
  return value.toFixed(2);
}

function confidenceClass(value: number | null): string {
  if (value == null) {
    return "";
  }
  if (value >= 0.8) {
    return "confidence-high";
  }
  if (value >= 0.5) {
    return "confidence-mid";
  }
  return "confidence-low";
}

function sortAggregateRows(rows: AggregateModelRow[]): AggregateModelRow[] {
  const hasAnyAccuracy = rows.some((row) => row.avg_accuracy != null);

  return [...rows].sort((left, right) => {
    const primaryLeft = hasAnyAccuracy ? left.avg_accuracy : left.avg_agreement;
    const primaryRight = hasAnyAccuracy ? right.avg_accuracy : right.avg_agreement;
    const normalizedLeft = primaryLeft ?? Number.NEGATIVE_INFINITY;
    const normalizedRight = primaryRight ?? Number.NEGATIVE_INFINITY;
    if (normalizedLeft !== normalizedRight) {
      return normalizedRight - normalizedLeft;
    }

    const agreementLeft = left.avg_agreement ?? Number.NEGATIVE_INFINITY;
    const agreementRight = right.avg_agreement ?? Number.NEGATIVE_INFINITY;
    if (agreementLeft !== agreementRight) {
      return agreementRight - agreementLeft;
    }

    if (left.run_count !== right.run_count) {
      return right.run_count - left.run_count;
    }

    return left.model_name.localeCompare(right.model_name);
  });
}

export function computeAggregateStats(runs: RunPayload[]): AggregateStats {
  const videoIds = new Set<string>();
  const models = new Set<string>();
  let totalCost = 0;

  for (const run of runs) {
    for (const videoId of run.config.video_ids) {
      videoIds.add(videoId);
    }
    for (const modelName of run.models) {
      models.add(modelName);
    }
    totalCost += run.results.reduce((sum, result) => sum + (result.estimated_cost ?? 0), 0);
  }

  return {
    total_runs: runs.length,
    total_videos: videoIds.size,
    total_cost: totalCost,
    models_tested: [...models].sort(),
  };
}

export function computeAggregateLeaderboard(runs: RunPayload[]): AggregateModelRow[] {
  const byModel = new Map<
    string,
    {
      agreement: number[];
      accuracy: number[];
      confidence: number[];
      latency: number[];
      total_cost: number;
      run_ids: Set<string>;
    }
  >();

  for (const run of runs) {
    const rows = buildCoreComparisonRows(run, getSweepData(run));
    for (const row of rows) {
      const current = byModel.get(row.model_name) ?? {
        agreement: [],
        accuracy: [],
        confidence: [],
        latency: [],
        total_cost: 0,
        run_ids: new Set<string>(),
      };

      if (row.agreement != null) {
        current.agreement.push(row.agreement);
      }
      if (row.accuracy != null) {
        current.accuracy.push(row.accuracy);
      }
      if (row.confidence != null) {
        current.confidence.push(row.confidence);
      }
      if (row.avg_latency_ms != null) {
        current.latency.push(row.avg_latency_ms);
      }

      current.total_cost += row.total_cost ?? 0;
      current.run_ids.add(run.run_id);
      byModel.set(row.model_name, current);
    }
  }

  return sortAggregateRows(
    [...byModel.entries()].map(([model_name, values]) => ({
      model_name,
      avg_agreement: average(values.agreement),
      avg_accuracy: average(values.accuracy),
      avg_confidence: average(values.confidence),
      avg_latency_ms: average(values.latency),
      total_cost: values.total_cost,
      run_count: values.run_ids.size,
    }))
  );
}

function indexedVideoCount(runList: RunListItem[]): number {
  return new Set(runList.flatMap((run) => run.video_ids)).size;
}

function indexedModels(runList: RunListItem[]): string[] {
  return [...new Set(runList.flatMap((run) => run.models))].sort();
}

function AggregateStatCard({
  label,
  value,
  secondary,
}: {
  label: string;
  value: string;
  secondary: string;
}) {
  return (
    <article className="hero-card">
      <p className="hero-card-label">{label}</p>
      <p className="hero-card-number">{value}</p>
      <p className="hero-card-secondary">{secondary}</p>
    </article>
  );
}

export function AggregateDashboard({
  runs,
  runList,
  dataDir,
  basePath = "/",
  loadedRunCount,
  nextLoadCount,
}: {
  runs: RunPayload[];
  runList: RunListItem[];
  dataDir?: string;
  basePath?: string;
  loadedRunCount: number;
  nextLoadCount?: number | null;
}) {
  const recentRuns = runList.slice(0, 10);
  const aggregateStats = computeAggregateStats(runs);
  const leaderboard = computeAggregateLeaderboard(runs);
  const totalRuns = runList.length;
  const totalVideos = indexedVideoCount(runList);
  const allModels = indexedModels(runList);
  const allRunsLoaded = totalRuns === 0 || loadedRunCount >= totalRuns;
  const loadMoreHref =
    nextLoadCount && nextLoadCount > loadedRunCount
      ? buildHref(basePath, { dataDir, limit: nextLoadCount })
      : null;

  if (runList.length === 0) {
    return (
      <main className="analysis-shell">
        <TopNav active="dashboard" />
        <section className="visual-card aggregate-hero">
          <div className="section-heading">
            <p className="section-eyebrow">Dashboard</p>
            <h2 className="run-title">No benchmark runs found yet.</h2>
            <p className="chart-desc">
              Upload a short clip to start building the cross-run leaderboard, or deploy static run
              exports to browse benchmark history.
            </p>
          </div>
          <div className="aggregate-hero-actions">
            <Link href={buildHref("/new", { dataDir })} className="primary-btn">
              New Benchmark
            </Link>
            <Link href={buildHref("/runs", { dataDir })} className="ghost-btn">
              Browse Runs
            </Link>
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="analysis-shell">
      <TopNav active="dashboard" />

      <section className="visual-card aggregate-hero dashboard-section-card">
        <div className="aggregate-hero-copy">
          <div className="section-heading">
            <p className="section-eyebrow">Dashboard</p>
            <h2 className="run-title">How are models trending across recent benchmarks?</h2>
            <p className="chart-desc">
              This view merges committed static exports with live Modal runs, then rolls the loaded
              run payloads into one cross-run leaderboard.
            </p>
          </div>
          <p className="aggregate-hero-note">
            {allRunsLoaded
              ? `All ${loadedRunCount} runs are loaded for aggregate scoring.`
              : `Showing aggregate scoring from ${loadedRunCount} loaded runs out of ${totalRuns}. Load more history to include older runs.`}
          </p>
        </div>
        <div className="aggregate-hero-actions">
          <Link href={buildHref("/new", { dataDir })} className="primary-btn">
            New Benchmark
          </Link>
          <Link href={buildHref("/runs", { dataDir })} className="ghost-btn">
            Browse All Runs
          </Link>
          {loadMoreHref ? (
            <Link href={loadMoreHref} className="ghost-btn">
              Load More History
            </Link>
          ) : null}
        </div>
      </section>

      <section className="hero-grid aggregate-stat-grid">
        <AggregateStatCard
          label="Total Runs Completed"
          value={String(totalRuns)}
          secondary="Static exports plus live backend runs"
        />
        <AggregateStatCard
          label="Total Videos Analyzed"
          value={String(totalVideos)}
          secondary="Deduplicated across the indexed run history"
        />
        <AggregateStatCard
          label={allRunsLoaded ? "Total API Cost" : "Loaded API Cost"}
          value={formatMoney(aggregateStats.total_cost)}
          secondary={
            allRunsLoaded
              ? "Summed across every loaded run payload"
              : `Summed across the ${loadedRunCount} runs loaded for aggregate analysis`
          }
        />
        <AggregateStatCard
          label="Models Tested"
          value={String(allModels.length)}
          secondary={allModels.length > 0 ? summarizeModelNames(allModels) : "No models found"}
        />
      </section>

      <section className="visual-card dashboard-section-card">
        <div className="section-heading">
          <p className="section-eyebrow">Cross-Run Leaderboard</p>
          <h3>Which models are strongest across the loaded run history?</h3>
          <p className="chart-desc">
            Each row averages the model&apos;s per-run agreement, accuracy, confidence, and latency,
            then totals cost across the runs it participated in.
          </p>
        </div>

        {leaderboard.length === 0 ? (
          <p className="empty-state">No full run payloads were available to build the leaderboard.</p>
        ) : (
          <div className="leaderboard-scroll">
            <table className="leaderboard-table aggregate-leaderboard-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Model</th>
                  <th>Avg Agreement</th>
                  <th>Avg Accuracy</th>
                  <th>Avg Confidence</th>
                  <th>Avg Latency</th>
                  <th>Total Cost</th>
                  <th>Runs</th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.map((row, index) => (
                  <tr
                    key={row.model_name}
                    className={`leaderboard-row ${index === 0 ? "top-ranked" : ""}`}
                  >
                    <td className="leaderboard-rank">#{index + 1}</td>
                    <td>
                      <div className="aggregate-model-cell">
                        <strong>{row.model_name}</strong>
                        <span>
                          {row.run_count} {row.run_count === 1 ? "run" : "runs"} loaded
                        </span>
                      </div>
                    </td>
                    <td>{formatPercent(row.avg_agreement)}</td>
                    <td>{formatPercent(row.avg_accuracy)}</td>
                    <td className={confidenceClass(row.avg_confidence)}>
                      {formatConfidence(row.avg_confidence)}
                    </td>
                    <td>{formatLatency(row.avg_latency_ms)}</td>
                    <td>{formatMoney(row.total_cost)}</td>
                    <td>{row.run_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="visual-card dashboard-section-card">
        <div className="section-heading">
          <p className="section-eyebrow">Recent Runs</p>
          <h3>Jump into a specific benchmark report.</h3>
          <p className="chart-desc">
            Use the report view for the per-run deep dive. These are the 10 most recent runs from
            the merged static and live history.
          </p>
        </div>

        <div className="table-scroll">
          <table className="data-table recent-runs-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Date</th>
                <th>Models</th>
                <th>Videos</th>
                <th>Open</th>
              </tr>
            </thead>
            <tbody>
              {recentRuns.map((run) => (
                <tr key={run.run_id}>
                  <td>
                    <div className="run-list-cell">
                      <Link
                        href={buildHref(`/report/${run.run_id}`, { dataDir })}
                        className="recent-run-link"
                      >
                        {displayRunName(run.run_id, run.created_at)}
                      </Link>
                      <p className="run-list-raw">{run.run_id}</p>
                    </div>
                  </td>
                  <td>{formatDateTime(run.created_at)}</td>
                  <td title={run.models.join(", ")}>
                    <div className="aggregate-tag-list">
                      {run.models.map((modelName) => (
                        <span key={`${run.run_id}-${modelName}`} className="model-tag">
                          {modelName}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td title={run.video_ids.join(", ")}>
                    {run.video_ids.length === 0
                      ? "-"
                      : run.video_ids
                          .slice(0, 2)
                          .map((videoId) => displayVideoName(videoId))
                          .join(", ")}
                    {run.video_ids.length > 2 ? ` +${run.video_ids.length - 2}` : ""}
                  </td>
                  <td>
                    <Link
                      href={buildHref(`/report/${run.run_id}`, { dataDir })}
                      className="ghost-btn small"
                    >
                      Open report
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}
