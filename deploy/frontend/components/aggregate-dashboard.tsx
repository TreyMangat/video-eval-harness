import Link from "next/link";

import {
  buildCoreComparisonRows,
  displayRunName,
  formatDateTime,
  formatLatency,
  formatMoney,
  formatPercent,
  getSweepData,
  modelColor,
} from "../lib/analysis";
import { isVisibleRun } from "../lib/run-visibility";
import { getRunType } from "../lib/run-type";
import type { RunListItem, RunPayload } from "../lib/types";
import { AggregateLeaderboardClient } from "./aggregate-leaderboard-client";
import { AggregateVisualsClient } from "./aggregate-visuals-client";
import { TopNav } from "./navigation";
import { RunTypeBadge } from "./run-type-badge";

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
  avg_llm_accuracy: number | null;
  avg_confidence: number | null;
  avg_latency_ms: number | null;
  total_cost: number;
  run_count: number;
  input_mode: string;
}

export type AggregateAgreementMatrix = Record<string, Record<string, number | null>>;

type AggregateAgreementVerdict = {
  strongestPair: string;
  strongestPairScore: number;
  weakestPair: string;
  weakestPairScore: number;
};

type AggregateSourceRow = ReturnType<typeof buildCoreComparisonRows>[number];

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

function preferredAccuracy(row: AggregateModelRow): number | null {
  return row.avg_llm_accuracy ?? row.avg_accuracy;
}

export function signalIndicator(row: AggregateModelRow): { label: string; className: string } | null {
  const accuracy = preferredAccuracy(row);
  if (accuracy == null || row.avg_agreement == null) {
    return null;
  }

  const delta = row.avg_agreement - accuracy;
  if (delta >= 0.1) {
    return {
      label: `Warning: overconfident +${Math.round(delta * 100)} pts`,
      className: "aggregate-signal-badge warning",
    };
  }
  if (delta <= -0.1) {
    return {
      label: `Underrated +${Math.round(Math.abs(delta) * 100)} pts`,
      className: "aggregate-signal-badge positive",
    };
  }
  return null;
}

function resolveAggregateInputMode(modes: Set<string>): string {
  if (modes.has("video")) {
    return "video";
  }
  return "frames";
}

export function inputModeLabel(mode: string): string {
  return mode === "video" ? "Video" : "Frames";
}

export function inputModeClass(mode: string): string {
  return mode === "video"
    ? "aggregate-input-mode video"
    : "aggregate-input-mode frames";
}

export function scoreTone(value: number | null): "high" | "mid" | "low" | "none" {
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
  const hasAnyLlmAccuracy = rows.some((row) => row.avg_llm_accuracy != null);
  const hasAnyAccuracy = rows.some((row) => row.avg_accuracy != null);

  return [...rows].sort((left, right) => {
    const primaryLeft = hasAnyLlmAccuracy
      ? left.avg_llm_accuracy
      : hasAnyAccuracy
        ? left.avg_accuracy
        : left.avg_agreement;
    const primaryRight = hasAnyLlmAccuracy
      ? right.avg_llm_accuracy
      : hasAnyAccuracy
        ? right.avg_accuracy
        : right.avg_agreement;
    const normalizedLeft = primaryLeft ?? Number.NEGATIVE_INFINITY;
    const normalizedRight = primaryRight ?? Number.NEGATIVE_INFINITY;
    if (normalizedLeft !== normalizedRight) {
      return normalizedRight - normalizedLeft;
    }

    const fallbackAccuracyLeft = preferredAccuracy(left) ?? Number.NEGATIVE_INFINITY;
    const fallbackAccuracyRight = preferredAccuracy(right) ?? Number.NEGATIVE_INFINITY;
    if (fallbackAccuracyLeft !== fallbackAccuracyRight) {
      return fallbackAccuracyRight - fallbackAccuracyLeft;
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

function bestAccuracyRow(rows: AggregateModelRow[]): AggregateModelRow | null {
  return [...rows]
    .filter((row) => preferredAccuracy(row) != null)
    .sort((left, right) => {
      const leftAccuracy = preferredAccuracy(left) ?? Number.NEGATIVE_INFINITY;
      const rightAccuracy = preferredAccuracy(right) ?? Number.NEGATIVE_INFINITY;
      if (leftAccuracy !== rightAccuracy) {
        return rightAccuracy - leftAccuracy;
      }

      const leftAgreement = left.avg_agreement ?? Number.NEGATIVE_INFINITY;
      const rightAgreement = right.avg_agreement ?? Number.NEGATIVE_INFINITY;
      if (leftAgreement !== rightAgreement) {
        return rightAgreement - leftAgreement;
      }

      return left.model_name.localeCompare(right.model_name);
    })[0] ?? null;
}

function bestAgreementRow(rows: AggregateModelRow[]): AggregateModelRow | null {
  return [...rows]
    .filter((row) => row.avg_agreement != null)
    .sort((left, right) => {
      const leftAgreement = left.avg_agreement ?? Number.NEGATIVE_INFINITY;
      const rightAgreement = right.avg_agreement ?? Number.NEGATIVE_INFINITY;
      if (leftAgreement !== rightAgreement) {
        return rightAgreement - leftAgreement;
      }

      const leftAccuracy = preferredAccuracy(left) ?? Number.NEGATIVE_INFINITY;
      const rightAccuracy = preferredAccuracy(right) ?? Number.NEGATIVE_INFINITY;
      if (leftAccuracy !== rightAccuracy) {
        return rightAccuracy - leftAccuracy;
      }

      return left.model_name.localeCompare(right.model_name);
    })[0] ?? null;
}

function bestLatencyRow(rows: AggregateModelRow[]): AggregateModelRow | null {
  return [...rows]
    .filter((row) => row.avg_latency_ms != null)
    .sort((left, right) => {
      const leftLatency = left.avg_latency_ms ?? Number.POSITIVE_INFINITY;
      const rightLatency = right.avg_latency_ms ?? Number.POSITIVE_INFINITY;
      if (leftLatency !== rightLatency) {
        return leftLatency - rightLatency;
      }

      const leftAccuracy = preferredAccuracy(left) ?? Number.NEGATIVE_INFINITY;
      const rightAccuracy = preferredAccuracy(right) ?? Number.NEGATIVE_INFINITY;
      if (leftAccuracy !== rightAccuracy) {
        return rightAccuracy - leftAccuracy;
      }

      return left.model_name.localeCompare(right.model_name);
    })[0] ?? null;
}

function bestValueRow(rows: AggregateModelRow[]): AggregateModelRow | null {
  return [...rows]
    .filter((row) => preferredAccuracy(row) != null && row.total_cost > 0)
    .sort((left, right) => {
      const leftRatio = (preferredAccuracy(left) ?? 0) / left.total_cost;
      const rightRatio = (preferredAccuracy(right) ?? 0) / right.total_cost;
      if (leftRatio !== rightRatio) {
        return rightRatio - leftRatio;
      }

      const leftAccuracy = preferredAccuracy(left) ?? Number.NEGATIVE_INFINITY;
      const rightAccuracy = preferredAccuracy(right) ?? Number.NEGATIVE_INFINITY;
      if (leftAccuracy !== rightAccuracy) {
        return rightAccuracy - leftAccuracy;
      }

      if (left.total_cost !== right.total_cost) {
        return left.total_cost - right.total_cost;
      }

      return left.model_name.localeCompare(right.model_name);
    })[0] ?? null;
}

function isValidAggregateModelRow(row: AggregateSourceRow): boolean {
  return row.parse_rate == null || row.parse_rate > 0;
}

function aggregateRowsForRun(run: RunPayload): AggregateSourceRow[] {
  return buildCoreComparisonRows(run, getSweepData(run)).filter(isValidAggregateModelRow);
}

function hasSegmentData(run: RunPayload): boolean {
  return (run.results?.length ?? 0) > 0 || (run.segments?.length ?? 0) > 0;
}

function isQualityAggregateRun(run: RunPayload): boolean {
  if (!isVisibleRun(run)) {
    return false;
  }
  const summaryModelCount = Object.keys(run.summaries ?? {}).length;
  if (summaryModelCount < 2 || !hasSegmentData(run)) {
    return false;
  }

  return aggregateRowsForRun(run).length >= 2;
}

function runHasAccuracyData(run: RunPayload): boolean {
  return aggregateRowsForRun(run).some((row) => row.llm_accuracy != null || row.accuracy != null);
}

export function computeAggregateStats(runs: RunPayload[]): AggregateStats {
  const videoIds = new Set<string>();
  const models = new Set<string>();
  let totalCost = 0;

  for (const run of runs) {
    const rows = aggregateRowsForRun(run);
    for (const videoId of run.config.video_ids) {
      videoIds.add(videoId);
    }
    for (const row of rows) {
      models.add(row.model_name);
    }
    totalCost += rows.reduce((sum, row) => sum + (row.total_cost ?? 0), 0);
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
      llm_accuracy: number[];
      confidence: number[];
      latency: number[];
      total_cost: number;
      run_ids: Set<string>;
      input_modes: Set<string>;
    }
  >();

  for (const run of runs) {
    const rows = aggregateRowsForRun(run);
    for (const row of rows) {
      const current = byModel.get(row.model_name) ?? {
        agreement: [],
        accuracy: [],
        llm_accuracy: [],
        confidence: [],
        latency: [],
        total_cost: 0,
        run_ids: new Set<string>(),
        input_modes: new Set<string>(),
      };

      if (row.agreement != null) {
        current.agreement.push(row.agreement);
      }
      if (row.accuracy != null) {
        current.accuracy.push(row.accuracy);
      }
      if (row.llm_accuracy != null) {
        current.llm_accuracy.push(row.llm_accuracy);
      }
      if (row.confidence != null) {
        current.confidence.push(row.confidence);
      }
      if (row.avg_latency_ms != null) {
        current.latency.push(row.avg_latency_ms);
      }

      current.total_cost += row.total_cost ?? 0;
      current.run_ids.add(run.run_id);
      current.input_modes.add(row.input_mode);
      byModel.set(row.model_name, current);
    }
  }

  return sortAggregateRows(
    [...byModel.entries()].map(([model_name, values]) => ({
      model_name,
      avg_agreement: average(values.agreement),
      avg_accuracy: average(values.accuracy),
      avg_llm_accuracy: average(values.llm_accuracy),
      avg_confidence: average(values.confidence),
      avg_latency_ms: average(values.latency),
      total_cost: values.total_cost,
      run_count: values.run_ids.size,
      input_mode: resolveAggregateInputMode(values.input_modes),
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
    const validModels = new Set(aggregateRowsForRun(run).map((row) => row.model_name));

    for (const modelName of validModels) {
      models.add(modelName);
    }

    for (const [rowModel, row] of Object.entries(run.agreement)) {
      if (!validModels.has(rowModel)) {
        continue;
      }
      models.add(rowModel);
      for (const [columnModel, score] of Object.entries(row)) {
        if (!validModels.has(columnModel)) {
          continue;
        }
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
      matrix[rowModel][columnModel] = count > 0 ? (pairSums.get(key) ?? 0) / count : null;
    }
  }

  return matrix;
}

function countAccuracyRuns(runs: RunPayload[]): number {
  return runs.filter(runHasAccuracyData).length;
}

function accuracySourceBreakdown(runs: RunPayload[]): {
  benchmark: number;
  accuracy_test: number;
  comparison: number;
} {
  return runs.reduce(
    (counts, run) => {
      const runType = getRunType(run);
      if (runType === "benchmark") {
        counts.benchmark += 1;
      } else if (runType === "accuracy_test") {
        counts.accuracy_test += 1;
      } else {
        counts.comparison += 1;
      }
      return counts;
    },
    {
      benchmark: 0,
      accuracy_test: 0,
      comparison: 0,
    }
  );
}

function computeAgreementVerdict(
  matrix: AggregateAgreementMatrix
): AggregateAgreementVerdict | null {
  const models = Object.keys(matrix).sort();
  let strongest: { pair: string; score: number } | null = null;
  let weakest: { pair: string; score: number } | null = null;

  for (let rowIndex = 0; rowIndex < models.length; rowIndex += 1) {
    const rowModel = models[rowIndex];
    for (let columnIndex = rowIndex + 1; columnIndex < models.length; columnIndex += 1) {
      const columnModel = models[columnIndex];
      const score = matrix[rowModel]?.[columnModel] ?? matrix[columnModel]?.[rowModel] ?? null;
      if (score == null) {
        continue;
      }

      const pair = `${rowModel} + ${columnModel}`;
      if (!strongest || score > strongest.score) {
        strongest = { pair, score };
      }
      if (!weakest || score < weakest.score) {
        weakest = { pair, score };
      }
    }
  }

  if (!strongest || !weakest) {
    return null;
  }

  return {
    strongestPair: strongest.pair,
    strongestPairScore: strongest.score,
    weakestPair: weakest.pair,
    weakestPairScore: weakest.score,
  };
}

export function medalClass(index: number): string {
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

function AggregateAgreementMatrixCard({
  matrix,
  verdict,
}: {
  matrix: AggregateAgreementMatrix;
  verdict: AggregateAgreementVerdict | null;
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

      {verdict ? (
        <p className="heatmap-verdict">
          Strongest pair: <strong>{verdict.strongestPair}</strong> (
          {formatPercent(verdict.strongestPairScore)}). Weakest pair:{" "}
          <strong>{verdict.weakestPair}</strong> ({formatPercent(verdict.weakestPairScore)}).
        </p>
      ) : null}
    </section>
  );
}

export function AggregateDashboard({
  runs,
  runList,
  dataDir,
}: {
  runs: RunPayload[];
  runList: RunListItem[];
  dataDir?: string;
}) {
  const runPayloadById = new Map(runs.map((run) => [run.run_id, run]));
  const visibleRuns = runs.filter(isVisibleRun);
  const visibleRunIds = new Set(visibleRuns.map((run) => run.run_id));
  const qualityRuns = runs.filter(isQualityAggregateRun);
  const recentRuns = runList.filter((run) => visibleRunIds.has(run.run_id)).slice(0, 10);
  const accuracyRuns = qualityRuns.filter(runHasAccuracyData);
  const aggregateStats = computeAggregateStats(qualityRuns);
  const leaderboard = computeAggregateLeaderboard(qualityRuns);
  const accuracyLeaderboard = computeAggregateLeaderboard(accuracyRuns);
  const bestAccuracy = bestAccuracyRow(accuracyLeaderboard);
  const bestAgreement = bestAgreementRow(leaderboard);
  const fastestModel = bestLatencyRow(leaderboard);
  const bestValue = bestValueRow(leaderboard);
  const hasExactAccuracy = leaderboard.some((row) => row.avg_accuracy != null);
  const hasLlmAccuracy = leaderboard.some((row) => row.avg_llm_accuracy != null);
  const aggregateAgreementMatrix = computeAggregateAgreementMatrix(qualityRuns);
  const agreementVerdict = computeAgreementVerdict(aggregateAgreementMatrix);
  const accuracyRunCount = countAccuracyRuns(qualityRuns);
  const accuracySources = accuracySourceBreakdown(accuracyRuns);
  const filteredRunCount = Math.max(runs.length - qualityRuns.length, 0);

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
              All available static exports and live API runs load on first render, so the
              dashboard stays stable across refreshes and route changes.
            </p>
            <p className="chart-desc">
              Aggregate cards and charts use {aggregateStats.total_runs} quality runs from the full
              merged history (committed static exports plus live Modal runs). Single-model runs,
              empty runs, hidden internal test/debug runs, and models with 0% parse success are
              filtered out before aggregation.
            </p>
          </div>
        </section>

        <section className="winner-cards">
          <div className="winner-card winner-accuracy">
            <span className="winner-label">Most accurate</span>
            <span className="winner-model">{bestAccuracy?.model_name ?? "—"}</span>
            <span className="winner-value">
              {bestAccuracy ? formatPercent(preferredAccuracy(bestAccuracy)) : "—"}
            </span>
            <span className="winner-sub">
              {bestAccuracy
                ? accuracySources.accuracy_test === 0 && accuracySources.comparison === 0
                  ? `LLM-judged accuracy across ${accuracyRunCount} benchmark runs`
                  : accuracySources.comparison > 0
                    ? `LLM-judged accuracy across ${accuracyRunCount} runs (${accuracySources.benchmark} benchmarks, ${accuracySources.accuracy_test} accuracy tests, ${accuracySources.comparison} comparisons)`
                    : `LLM-judged accuracy across ${accuracyRunCount} runs (${accuracySources.benchmark} benchmarks, ${accuracySources.accuracy_test} accuracy tests)`
                : "Run with --ground-truth to enable"}
            </span>
          </div>
          <div className="winner-card winner-agreement">
            <span className="winner-label">Best consensus</span>
            <span className="winner-model">{bestAgreement?.model_name ?? "—"}</span>
            <span className="winner-value">
              {bestAgreement ? formatPercent(bestAgreement.avg_agreement) : "—"}
            </span>
            <span className="winner-sub">Average cross-model agreement</span>
          </div>
          <div className="winner-card winner-speed">
            <span className="winner-label">Fastest</span>
            <span className="winner-model">{fastestModel?.model_name ?? "—"}</span>
            <span className="winner-value">
              {fastestModel ? formatLatency(fastestModel.avg_latency_ms) : "—"}
            </span>
            <span className="winner-sub">Average response latency</span>
          </div>
          <div className="winner-card winner-value">
            <span className="winner-label">Best value</span>
            <span className="winner-model">{bestValue?.model_name ?? "—"}</span>
            <span className="winner-value">
              {bestValue
                ? `${formatPercent(preferredAccuracy(bestValue))} at ${formatMoney(bestValue.total_cost)}`
                : "—"}
            </span>
            <span className="winner-sub">
              {bestValue ? "Highest accuracy per dollar" : "Run with --ground-truth to enable"}
            </span>
          </div>
        </section>

        <p className="meta-line">
          {aggregateStats.total_runs} quality runs · {aggregateStats.total_videos} videos · $
          {aggregateStats.total_cost.toFixed(2)} API cost · {aggregateStats.models_tested.length}{" "}
          models tested
          {filteredRunCount > 0 ? ` · ${filteredRunCount} runs filtered from aggregation` : ""}
        </p>

        <AggregateVisualsClient rows={leaderboard} />

        <AggregateAgreementMatrixCard
          matrix={aggregateAgreementMatrix}
          verdict={agreementVerdict}
        />

        <section className="visual-card dashboard-section-card">
          <div className="section-heading">
            <p className="section-eyebrow">Cross-Run Leaderboard</p>
            <h3>Which models stay strongest across the loaded run history?</h3>
            <p className="chart-desc">
              The leaderboard defaults to the strongest available accuracy signal when it exists,
              while still making agreement visible so overconfident and underrated models stand
              out immediately.
            </p>
          </div>

          {leaderboard.length === 0 ? (
            <p className="empty-state">No full run payloads were available to build the leaderboard.</p>
          ) : (
            <AggregateLeaderboardClient
              rows={leaderboard}
              hasExactAccuracy={hasExactAccuracy}
              hasLlmAccuracy={hasLlmAccuracy}
            />
          )}
        </section>

        <section className="aggregate-recent-section">
          <div className="section-heading">
            <p className="section-eyebrow">Recent Runs</p>
            <h3>Jump from the overview into a specific benchmark report.</h3>
            <p className="chart-desc">
              Each card links straight into the per-run report page, with model chips and benchmark
              scope kept visible at a glance.
            </p>
          </div>

          <div className="recent-runs-grid">
            {recentRuns.map((run) => {
              const runPayload = runPayloadById.get(run.run_id);
              const segmentCount = runPayload?.segments?.length;
              const runBadgeSource = runPayload ?? run;

              return (
                <article key={run.run_id} className="visual-card recent-run-card">
                  <div className="recent-run-card-head">
                    <div className="run-list-cell">
                      <div className="run-list-title-row">
                        <p className="run-list-name">{displayRunName(run.run_id, run.created_at)}</p>
                        <RunTypeBadge run={runBadgeSource} />
                      </div>
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

                  <p className="recent-run-card-summary">
                    {run.video_ids.length} videos · {segmentCount ?? "—"} segments
                  </p>

                  <Link
                    href={buildHref(`/report/${run.run_id}`, { dataDir })}
                    className="recent-run-card-link"
                  >
                    Open Report →
                  </Link>
                </article>
              );
            })}
          </div>
        </section>
      </div>
    </main>
  );
}
