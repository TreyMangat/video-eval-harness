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
  modelColor,
} from "../lib/analysis";
import type { RunListItem, RunPayload } from "../lib/types";
import { AggregateVisualsClient } from "./aggregate-visuals-client";
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

export type AggregateAgreementMatrix = Record<string, Record<string, number | null>>;

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
  if (models.length === 0) {
    return "No models loaded";
  }
  if (models.length === 1) {
    return models[0];
  }
  if (models.length === 2) {
    return `${models[0]}, ${models[1]}`;
  }
  return `${models[0]}, ${models[1]} +${models.length - 2} more`;
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

function scoreTone(value: number | null): "high" | "mid" | "low" | "none" {
  if (value == null) {
    return "none";
  }
  if (value >= 0.8) {
    return "high";
  }
  if (value >= 0.5) {
    return "mid";
  }
  return "low";
}

function heatColor(value: number): string {
  if (value >= 0.8) {
    return "rgba(34, 197, 94, 0.25)";
  }
  if (value >= 0.5) {
    return "rgba(245, 158, 11, 0.2)";
  }
  if (value >= 0.3) {
    return "rgba(245, 158, 11, 0.12)";
  }
  return "rgba(239, 68, 68, 0.1)";
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

export function computeAggregateAgreementMatrix(
  runs: RunPayload[]
): AggregateAgreementMatrix {
  const models = new Set<string>();
  const pairSums = new Map<string, number>();
  const pairCounts = new Map<string, number>();

  for (const run of runs) {
    for (const modelName of run.models) {
      models.add(modelName);
    }

    for (const [rowModel, row] of Object.entries(run.agreement)) {
      models.add(rowModel);
      for (const [columnModel, score] of Object.entries(row)) {
        models.add(columnModel);
        if (rowModel === columnModel) {
          continue;
        }
        const key = `${rowModel}::${columnModel}`;
        pairSums.set(key, (pairSums.get(key) ?? 0) + score);
        pairCounts.set(key, (pairCounts.get(key) ?? 0) + 1);
      }
    }
  }

  const matrix: AggregateAgreementMatrix = {};
  const modelList = [...models].sort();

  for (const rowModel of modelList) {
    matrix[rowModel] = {};
    for (const columnModel of modelList) {
      if (rowModel === columnModel) {
        matrix[rowModel][columnModel] = 1;
        continue;
      }
      const key = `${rowModel}::${columnModel}`;
      const count = pairCounts.get(key) ?? 0;
      matrix[rowModel][columnModel] =
        count > 0 ? (pairSums.get(key) ?? 0) / count : null;
    }
  }

  return matrix;
}

function indexedVideoCount(runList: RunListItem[]): number {
  return new Set(runList.flatMap((run) => run.video_ids)).size;
}

function indexedModels(runList: RunListItem[]): string[] {
  return [...new Set(runList.flatMap((run) => run.models))].sort();
}

function medalClass(index: number): string {
  if (index === 0) {
    return "medal-gold";
  }
  if (index === 1) {
    return "medal-silver";
  }
  if (index === 2) {
    return "medal-bronze";
  }
  return "";
}

function AggregateStatCard({
  label,
  value,
  secondary,
  accentColor,
}: {
  label: string;
  value: string;
  secondary: string;
  accentColor: string;
}) {
  return (
    <article className="hero-card" style={{ borderTopColor: accentColor }}>
      <p className="hero-card-label">{label}</p>
      <p className="hero-card-number" style={{ color: accentColor }}>
        {value}
      </p>
      <p className="hero-card-secondary">{secondary}</p>
    </article>
  );
}

function AggregateAgreementMatrixCard({
  matrix,
}: {
  matrix: AggregateAgreementMatrix;
}) {
  const models = Object.keys(matrix).filter((modelName) =>
    Object.entries(matrix[modelName] ?? {}).some(
      ([otherModel, value]) => otherModel !== modelName && value != null
    )
  );

  if (models.length < 2) {
    return null;
  }

  return (
    <section className="visual-card agreement-section dashboard-section-card">
      <div className="section-heading">
        <p className="section-eyebrow">Agreement Heatmap</p>
        <h3>Where does cross-run consensus stay strong?</h3>
        <p className="chart-desc">
          These cells average pairwise agreement only across runs where both models appeared
          together, so blank cells simply mean there was no overlap in the loaded history.
        </p>
      </div>

      <div className="matrix-scroll">
        <table className="agreement-table aggregate-matrix-table">
          <thead>
            <tr>
              <th />
              {models.map((modelName) => (
                <th key={`column-${modelName}`}>{modelName}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((rowModel) => (
              <tr key={`row-${rowModel}`}>
                <td className="matrix-row-label">{rowModel}</td>
                {models.map((columnModel) => {
                  const value = matrix[rowModel]?.[columnModel] ?? null;
                  const isMissing = value == null;
                  return (
                    <td
                      key={`${rowModel}-${columnModel}`}
                      className={`matrix-cell mono ${isMissing ? "aggregate-matrix-missing" : ""}`}
                      style={value == null ? undefined : { background: heatColor(value) }}
                    >
                      {value == null ? "—" : formatPercent(value)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
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
  const aggregateAgreementMatrix = computeAggregateAgreementMatrix(runs);
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
        <section className="visual-card aggregate-hero dashboard-section-card">
          <div className="section-heading">
            <p className="section-eyebrow">Dashboard</p>
            <h2 className="run-title">No benchmark runs found yet.</h2>
            <p className="chart-desc">
              Upload a short clip to start building the cross-run leaderboard, or deploy static run
              exports to browse benchmark history.
            </p>
          </div>
          <div className="aggregate-hero-actions">
            <Link href={buildHref("/new", { dataDir })} className="ghost-btn small aggregate-action-btn">
              New Benchmark
            </Link>
            <Link href={buildHref("/runs", { dataDir })} className="ghost-btn small aggregate-action-btn">
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

      <div className="aggregate-dashboard-stack">
        <section className="visual-card aggregate-hero dashboard-section-card">
          <div className="aggregate-hero-copy">
            <div className="section-heading">
              <p className="section-eyebrow">Dashboard</p>
              <h2 className="run-title">How are models trending across recent benchmarks?</h2>
              <p className="chart-desc">
                The aggregate view keeps the upload flow out of the way and surfaces the patterns
                that matter across both committed static runs and live Modal benchmarks.
              </p>
            </div>
            <p className="aggregate-hero-note">
              {allRunsLoaded
                ? `All ${loadedRunCount} available runs are loaded into the aggregate leaderboard.`
                : `Showing aggregate metrics from ${loadedRunCount} loaded runs out of ${totalRuns}. Load more history to pull older runs into the charts and leaderboard.`}
            </p>
          </div>

          <div className="aggregate-hero-actions">
            <Link href={buildHref("/new", { dataDir })} className="ghost-btn small aggregate-action-btn">
              New Benchmark
            </Link>
            <Link href={buildHref("/runs", { dataDir })} className="ghost-btn small aggregate-action-btn">
              Browse All Runs
            </Link>
            {loadMoreHref ? (
              <Link href={loadMoreHref} className="ghost-btn small aggregate-action-btn">
                Load More History
              </Link>
            ) : null}
          </div>
        </section>

        <section className="aggregate-stats-grid">
          <AggregateStatCard
            label="Total Runs Completed"
            value={String(totalRuns)}
            secondary="Merged static exports and live backend runs"
            accentColor="#4c9aff"
          />
          <AggregateStatCard
            label="Videos Analyzed"
            value={String(totalVideos)}
            secondary="Deduplicated across the indexed benchmark history"
            accentColor="#38bdf8"
          />
          <AggregateStatCard
            label={allRunsLoaded ? "Total API Cost" : "Loaded API Cost"}
            value={formatMoney(aggregateStats.total_cost)}
            secondary={
              allRunsLoaded
                ? "Summed across every loaded run payload"
                : `${loadedRunCount} loaded runs contributing to current charts`
            }
            accentColor="#f59e0b"
          />
          <AggregateStatCard
            label="Models Tested"
            value={String(allModels.length)}
            secondary={summarizeModelNames(allModels)}
            accentColor="#22c55e"
          />
        </section>

        <AggregateVisualsClient rows={leaderboard} />

        <AggregateAgreementMatrixCard matrix={aggregateAgreementMatrix} />

        <section className="visual-card dashboard-section-card">
          <div className="section-heading">
            <p className="section-eyebrow">Cross-Run Leaderboard</p>
            <h3>Which models stay strongest across the loaded run history?</h3>
            <p className="chart-desc">
              Agreement is shown with an inline score bar, confidence is color-coded, and the run
              count tells you how much history sits behind each average.
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
                  {leaderboard.map((row, index) => {
                    const medal = medalClass(index);
                    const agreementTone = scoreTone(row.avg_agreement);
                    return (
                      <tr
                        key={row.model_name}
                        className={`leaderboard-row aggregate-leaderboard-row ${index === 0 ? "top-ranked" : ""} ${medal}`}
                        style={{ boxShadow: `inset 4px 0 0 ${modelColor(row.model_name)}` }}
                      >
                        <td className={`leaderboard-rank ${medal}`}>
                          <span className={`leaderboard-rank-pill ${medal}`}>#{index + 1}</span>
                        </td>
                        <td>
                          <div className="aggregate-model-cell">
                            <strong>{row.model_name}</strong>
                            <span>
                              {row.run_count} {row.run_count === 1 ? "run" : "runs"} loaded
                            </span>
                          </div>
                        </td>
                        <td>
                          <div className={`aggregate-score-track ${agreementTone}`}>
                            <span
                              className={`aggregate-score-fill ${agreementTone}`}
                              style={{ width: `${Math.max(0, Math.min(100, (row.avg_agreement ?? 0) * 100))}%` }}
                            />
                            <span className="aggregate-score-label">
                              {formatPercent(row.avg_agreement)}
                            </span>
                          </div>
                        </td>
                        <td>{formatPercent(row.avg_accuracy)}</td>
                        <td className={confidenceClass(row.avg_confidence)}>
                          {formatConfidence(row.avg_confidence)}
                        </td>
                        <td>{formatLatency(row.avg_latency_ms)}</td>
                        <td>{formatMoney(row.total_cost)}</td>
                        <td>{row.run_count}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="aggregate-recent-section">
          <div className="section-heading">
            <p className="section-eyebrow">Recent Runs</p>
            <h3>Jump from the overview into a specific benchmark report.</h3>
            <p className="chart-desc">
              Each card links straight into the per-run report page, with model chips and recent
              video context kept visible at a glance.
            </p>
          </div>

          <div className="recent-runs-grid">
            {recentRuns.map((run) => (
              <article key={run.run_id} className="visual-card recent-run-card">
                <div className="recent-run-card-head">
                  <div className="run-list-cell">
                    <p className="run-list-name">{displayRunName(run.run_id, run.created_at)}</p>
                    <p className="run-list-raw">{run.run_id}</p>
                  </div>
                  <p className="recent-run-card-date">{formatDateTime(run.created_at)}</p>
                </div>

                <div className="aggregate-tag-list">
                  {run.models.map((modelName) => (
                    <span
                      key={`${run.run_id}-${modelName}`}
                      className="model-tag"
                      style={{
                        background: `${modelColor(modelName)}22`,
                        color: modelColor(modelName),
                        borderColor: `${modelColor(modelName)}55`,
                      }}
                    >
                      {modelName}
                    </span>
                  ))}
                </div>

                <p className="recent-run-card-videos">
                  {run.video_ids.length === 0
                    ? "No video metadata available"
                    : run.video_ids.map((videoId) => displayVideoName(videoId)).join(", ")}
                </p>

                <Link
                  href={buildHref(`/report/${run.run_id}`, { dataDir })}
                  className="recent-run-card-link"
                >
                  Open Report →
                </Link>
              </article>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
