"use client";

import type { CSSProperties } from "react";
import { useEffect, useMemo, useState } from "react";
import Link from "next/link";

import { DEFAULT_MODEL_CATALOG } from "../lib/model-catalog";
import { computeSweepMetrics } from "../lib/run-metrics";
import type {
  LabelResult,
  ModelCatalogItem,
  RunListItem,
  RunPayload,
  SegmentMedia,
  SegmentSummary,
  SweepCell,
  SweepMetrics,
} from "../lib/types";
import { ConfidenceChart, CostChart, LatencyChart, ParseRateChart } from "./charts";

type Tab = "overview" | "segments" | "raw";

const ALL_VARIANTS = "All variants";

async function readJson<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const response = await fetch(input, {
    cache: "no-store",
    ...init,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error ?? data.detail ?? "Request failed");
  }
  return data as T;
}

function buildApiPath(
  pathname: string,
  query: Record<string, string | null | undefined>
): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value) {
      params.set(key, value);
    }
  }
  const suffix = params.toString();
  return suffix ? `${pathname}?${suffix}` : pathname;
}

function formatPercent(value: number | null | undefined): string {
  if (value == null) {
    return "-";
  }
  return `${Math.round(value * 100)}%`;
}

function formatMoney(value: number | null | undefined): string {
  if (value == null) {
    return "-";
  }
  return `$${value.toFixed(4)}`;
}

function formatLatency(value: number | null | undefined): string {
  if (value == null) {
    return "-";
  }
  return `${Math.round(value)} ms`;
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${mins}:${secs}`;
}

function formatDateTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function heatColor(value: number): string {
  if (value >= 0.8) {
    return "rgba(34,197,94,0.25)";
  }
  if (value >= 0.5) {
    return "rgba(245,158,11,0.2)";
  }
  if (value >= 0.3) {
    return "rgba(245,158,11,0.12)";
  }
  return "rgba(239,68,68,0.1)";
}

function heatStyle(value: number): CSSProperties {
  return { background: heatColor(value) };
}

function resultVariantLabel(result: LabelResult): string {
  if (result.extraction_variant_id?.trim()) {
    return result.extraction_label?.trim() || result.extraction_variant_id;
  }
  return "default";
}

function AgreementTable({
  title,
  matrix,
}: {
  title: string;
  matrix: Record<string, Record<string, number>>;
}) {
  const models = Object.keys(matrix).sort();
  if (models.length === 0) {
    return null;
  }

  return (
    <section className="agreement-section">
      <h3>{title}</h3>
      <p className="chart-desc">Pairwise primary-action agreement across loaded model outputs.</p>
      <div className="matrix-scroll">
        <table className="agreement-table">
          <thead>
            <tr>
              <th />
              {models.map((model) => (
                <th key={model}>{model}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((rowModel) => (
              <tr key={rowModel}>
                <td className="matrix-row-label">{rowModel}</td>
                {models.map((columnModel) => {
                  const value = matrix[rowModel]?.[columnModel] ?? 0;
                  return (
                    <td
                      key={`${rowModel}-${columnModel}`}
                      className="matrix-cell mono"
                      style={heatStyle(value)}
                    >
                      {formatPercent(value)}
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

export function BenchmarkDashboard({ dataDir }: { dataDir?: string }) {
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [runData, setRunData] = useState<RunPayload | null>(null);
  const [segmentMedia, setSegmentMedia] = useState<SegmentMedia | null>(null);
  const [activeSegmentId, setActiveSegmentId] = useState<string | null>(null);
  const [catalog, setCatalog] = useState<ModelCatalogItem[]>(DEFAULT_MODEL_CATALOG);
  const [loadingRuns, setLoadingRuns] = useState(true);
  const [loadingRun, setLoadingRun] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("overview");
  const [variantFilter, setVariantFilter] = useState<string>(ALL_VARIANTS);

  const sweepData = useMemo<SweepMetrics | null>(() => {
    if (!runData) {
      return null;
    }
    if (runData.sweep?.has_sweep) {
      return runData.sweep;
    }
    const computed = computeSweepMetrics(runData.results);
    return computed.has_sweep ? computed : null;
  }, [runData]);

  const activeVariantId = useMemo(() => {
    if (!sweepData || variantFilter === ALL_VARIANTS) {
      return null;
    }
    return sweepData.variant_id_by_label[variantFilter] ?? null;
  }, [sweepData, variantFilter]);

  const activeSegment = useMemo<SegmentSummary | null>(() => {
    if (!runData || !activeSegmentId) {
      return null;
    }
    return runData.segments.find((segment) => segment.segment_id === activeSegmentId) ?? null;
  }, [runData, activeSegmentId]);

  const activeResults = useMemo<LabelResult[]>(() => {
    if (!runData || !activeSegmentId) {
      return [];
    }
    const segmentResults = runData.results.filter((result) => result.segment_id === activeSegmentId);
    if (!sweepData || variantFilter === ALL_VARIANTS) {
      return segmentResults.sort(
        (left, right) =>
          resultVariantLabel(left).localeCompare(resultVariantLabel(right)) ||
          left.model_name.localeCompare(right.model_name)
      );
    }
    return segmentResults
      .filter((result) => resultVariantLabel(result) === variantFilter)
      .sort((left, right) => left.model_name.localeCompare(right.model_name));
  }, [runData, activeSegmentId, sweepData, variantFilter]);

  const sweepCellsByModelVariant = useMemo(() => {
    const cells = new Map<string, SweepCell>();
    for (const cell of sweepData?.cells ?? []) {
      cells.set(`${cell.model_name}::${cell.variant_label}`, cell);
    }
    return cells;
  }, [sweepData]);

  const totalRunCost = useMemo(() => {
    if (!runData) {
      return 0;
    }
    return runData.results.reduce((sum, result) => sum + (result.estimated_cost ?? 0), 0);
  }, [runData]);

  const costPerModelRows = useMemo(
    () =>
      runData
        ? runData.models
            .map((model) => ({
              model,
              totalCost: runData.results
                .filter((result) => result.model_name === model)
                .reduce((sum, result) => sum + (result.estimated_cost ?? 0), 0),
            }))
            .sort((left, right) => right.totalCost - left.totalCost)
        : [],
    [runData]
  );

  const costPerVariantRows = useMemo(() => {
    if (!runData) {
      return [];
    }

    const byVariant = new Map<string, number>();
    for (const result of runData.results) {
      const variant = resultVariantLabel(result);
      byVariant.set(variant, (byVariant.get(variant) ?? 0) + (result.estimated_cost ?? 0));
    }

    const preferredOrder = sweepData?.variants ?? [];
    return [...byVariant.entries()]
      .map(([variant, totalCost]) => ({ variant, totalCost }))
      .sort((left, right) => {
        const leftIndex = preferredOrder.indexOf(left.variant);
        const rightIndex = preferredOrder.indexOf(right.variant);
        if (leftIndex !== -1 || rightIndex !== -1) {
          return (
            (leftIndex === -1 ? Number.MAX_SAFE_INTEGER : leftIndex) -
            (rightIndex === -1 ? Number.MAX_SAFE_INTEGER : rightIndex)
          );
        }
        return right.totalCost - left.totalCost;
      });
  }, [runData, sweepData]);

  const costPerSegmentRows = useMemo(() => {
    if (!runData) {
      return [];
    }

    const segmentLookup = new Map(
      runData.segments.map((segment) => [segment.segment_id, segment] as const)
    );
    const bySegment = new Map<
      string,
      { segmentId: string; videoId: string; startTimeS: number; totalCost: number }
    >();

    for (const result of runData.results) {
      const segment = segmentLookup.get(result.segment_id);
      const current = bySegment.get(result.segment_id) ?? {
        segmentId: result.segment_id,
        videoId: result.video_id,
        startTimeS: segment?.start_time_s ?? result.start_time_s,
        totalCost: 0,
      };
      current.totalCost += result.estimated_cost ?? 0;
      bySegment.set(result.segment_id, current);
    }

    return [...bySegment.values()].sort(
      (left, right) =>
        left.videoId.localeCompare(right.videoId) || left.startTimeS - right.startTimeS
    );
  }, [runData]);

  const runSourceLabel = dataDir || "Auto-detecting artifacts/runs";

  useEffect(() => {
    void loadCatalog();
    void loadRuns(true);
  }, [dataDir]);

  useEffect(() => {
    setVariantFilter(ALL_VARIANTS);
  }, [runData?.run_id]);

  useEffect(() => {
    if (!runData || !activeSegmentId) {
      setSegmentMedia(null);
      return;
    }
    void loadSegmentMedia(runData.run_id, activeSegmentId, activeVariantId);
  }, [runData, activeSegmentId, activeVariantId, dataDir]);

  async function loadCatalog() {
    try {
      const data = await readJson<{ models: ModelCatalogItem[] }>("/api/models");
      if (Array.isArray(data.models) && data.models.length > 0) {
        setCatalog(
          data.models.map((item) => ({
            ...item,
            notes: item.notes ?? "",
          }))
        );
        return;
      }
      setCatalog(DEFAULT_MODEL_CATALOG);
    } catch {
      setCatalog(DEFAULT_MODEL_CATALOG);
    }
  }

  async function loadRuns(autoSelectNewest = false) {
    try {
      setLoadingRuns(true);
      setError(null);
      const data = await readJson<RunListItem[]>(buildApiPath("/api/runs", { dataDir }));
      const sortedRuns = [...data].sort(
        (left, right) =>
          new Date(right.created_at).getTime() - new Date(left.created_at).getTime()
      );
      setRuns(sortedRuns);

      if (autoSelectNewest && sortedRuns[0]) {
        await loadRun(sortedRuns[0].run_id);
        return;
      }

      if (sortedRuns.length === 0) {
        setRunData(null);
        setActiveSegmentId(null);
      }
    } catch (loadError) {
      setRuns([]);
      setRunData(null);
      setActiveSegmentId(null);
      setError(loadError instanceof Error ? loadError.message : "Failed to load runs");
    } finally {
      setLoadingRuns(false);
    }
  }

  async function loadRun(runId: string) {
    try {
      setLoadingRun(true);
      setError(null);
      const data = await readJson<RunPayload>(buildApiPath(`/api/runs/${runId}`, { dataDir }));
      setRunData(data);
      setActiveSegmentId(data.segments[0]?.segment_id ?? null);
      setTab("overview");
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to load run");
    } finally {
      setLoadingRun(false);
    }
  }

  async function loadSegmentMedia(runId: string, segmentId: string, variantId: string | null) {
    try {
      const data = await readJson<SegmentMedia>(
        buildApiPath(`/api/runs/${runId}/segments/${segmentId}/media`, {
          dataDir,
          variantId,
        })
      );
      setSegmentMedia(data);
    } catch {
      setSegmentMedia(null);
    }
  }

  const videoLabel =
    runData?.videos[0]?.filename ||
    runData?.videos[0]?.video_id ||
    runData?.config.video_ids[0] ||
    "Unknown video";

  return (
    <div className="page-shell">
      <header className="top-bar">
        <div className="top-bar-left">
          <h1 className="logo">VBench Dashboard</h1>
          <span className="logo-sub">React frontend wired to exported run artifacts</span>
        </div>
        <div className="top-bar-actions">
          <Link href="/compare" className="ghost-btn small">
            Compare Runs
          </Link>
          <div className="badge">{runs.length} runs</div>
        </div>
      </header>

      <div className="main-layout">
        <aside className="sidebar">
          <div className="sidebar-heading">
            <div>
              <h2>Available Runs</h2>
              <p className="helper-copy">Source: {runSourceLabel}</p>
            </div>
            <div className="sidebar-actions">
              <button
                type="button"
                className="ghost-btn small"
                onClick={() => void loadRuns(true)}
              >
                Refresh
              </button>
            </div>
          </div>

          {loadingRuns ? (
            <p className="loading-copy">Scanning run directories...</p>
          ) : runs.length === 0 ? (
            <p className="empty-state">
              No runs found. Point the dashboard at your exported artifacts/runs directory.
            </p>
          ) : (
            <div className="run-list">
              {runs.map((run) => (
                <button
                  key={run.run_id}
                  type="button"
                  className={`run-item ${runData?.run_id === run.run_id ? "active" : ""}`}
                  onClick={() => void loadRun(run.run_id)}
                >
                  <div className="run-item-top">
                    <strong>{run.run_id}</strong>
                    <span className="run-date">{formatDateTime(run.created_at)}</span>
                  </div>
                  <div className="run-item-models">
                    {run.models.map((model) => (
                      <span key={`${run.run_id}-${model}`} className="model-tag">
                        {model}
                      </span>
                    ))}
                  </div>
                </button>
              ))}
            </div>
          )}
        </aside>

        <main className="content">
          {error ? <div className="error-banner">{error}</div> : null}

          {!runData ? (
            <section className="empty-hero">
              <h2>No run selected</h2>
              <p>
                Pick a run from the sidebar to load raw results, agreement views, segment
                comparisons, and sweep metrics.
              </p>
            </section>
          ) : (
            <>
              <section className="run-header">
                <div>
                  <h2 className="run-title">{runData.run_id}</h2>
                  <p className="run-meta">
                    {videoLabel} | {runData.models.length} models | {runData.segments.length}{" "}
                    segments | {formatDateTime(runData.config.created_at)}
                  </p>
                </div>
                <div className="run-header-actions">
                  <Link href={`/report/${runData.run_id}`} className="ghost-btn small">
                    Printable Report
                  </Link>
                  <Link href={`/compare?runA=${runData.run_id}`} className="ghost-btn small">
                    Compare This Run
                  </Link>
                  <div className="tab-bar">
                    {(["overview", "segments", "raw"] as Tab[]).map((currentTab) => (
                      <button
                        key={currentTab}
                        type="button"
                        className={`tab-btn ${tab === currentTab ? "active" : ""}`}
                        onClick={() => setTab(currentTab)}
                      >
                        {currentTab === "overview"
                          ? "Overview"
                          : currentTab === "segments"
                            ? "Segments"
                            : "Raw Data"}
                      </button>
                    ))}
                  </div>
                </div>
              </section>

              {loadingRun ? <p className="loading-copy">Loading run data...</p> : null}

              {tab === "overview" ? (
                <div className="tab-content">
                  <section className="summary-grid">
                    <article className="summary-card">
                      <p className="card-label">Models</p>
                      <p className="card-value">{runData.models.length}</p>
                      <span className="card-sublabel">Compared in this run</span>
                    </article>
                    <article className="summary-card">
                      <p className="card-label">Segments</p>
                      <p className="card-value">{runData.segments.length}</p>
                      <span className="card-sublabel">Segment comparisons loaded</span>
                    </article>
                    <article className="summary-card">
                      <p className="card-label">Results</p>
                      <p className="card-value">{runData.results.length}</p>
                      <span className="card-sublabel">Raw rows in export JSON/CSV</span>
                    </article>
                    <article className="summary-card">
                      <p className="card-label">Sweep</p>
                      <p className="card-value">
                        {sweepData ? sweepData.variants.length : 0}
                      </p>
                      <span className="card-sublabel">
                        {sweepData ? "Variants available" : "No sweep results yet"}
                      </span>
                    </article>
                    <article className="summary-card">
                      <p className="card-label">Total Cost</p>
                      <p className="card-value">{formatMoney(totalRunCost)}</p>
                      <span className="card-sublabel">
                        {runData.segments.length > 0
                          ? `${formatMoney(totalRunCost / runData.segments.length)} per segment`
                          : "No segment cost data"}
                      </span>
                    </article>
                  </section>

                  <section className="raw-section" style={{ marginBottom: 20 }}>
                    <h3>Run Configuration</h3>
                    <p className="chart-desc">
                      Loaded from exported run artifacts with sweep summary preference when
                      available.
                    </p>
                    <div className="table-scroll">
                      <table className="data-table">
                        <tbody>
                          <tr>
                            <th>Run ID</th>
                            <td className="mono">{runData.run_id}</td>
                          </tr>
                          <tr>
                            <th>Data Directory</th>
                            <td className="mono">{runSourceLabel}</td>
                          </tr>
                          <tr>
                            <th>Prompt Version</th>
                            <td>{runData.config.prompt_version}</td>
                          </tr>
                          <tr>
                            <th>Segmentation</th>
                            <td>{runData.config.segmentation_mode}</td>
                          </tr>
                          <tr>
                            <th>Videos</th>
                            <td>{runData.config.video_ids.join(", ") || "-"}</td>
                          </tr>
                          <tr>
                            <th>Models</th>
                            <td>{runData.models.join(", ")}</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </section>

                  <section className="raw-section" style={{ marginBottom: 20 }}>
                    <h3>Model Performance Table</h3>
                    <p className="chart-desc">
                      Parse rate, latency, confidence, and cost rolled up from raw result rows.
                    </p>
                    <div className="table-scroll">
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Model</th>
                            <th>Parse Rate</th>
                            <th>Avg Latency</th>
                            <th>P95 Latency</th>
                            <th>Avg Confidence</th>
                            <th>Total Cost</th>
                          </tr>
                        </thead>
                        <tbody>
                          {runData.models.map((model) => {
                            const summary = runData.summaries[model];
                            return (
                              <tr key={model}>
                                <td>{model}</td>
                                <td>{formatPercent(summary?.parse_success_rate)}</td>
                                <td>{formatLatency(summary?.avg_latency_ms)}</td>
                                <td>{formatLatency(summary?.p95_latency_ms)}</td>
                                <td>
                                  {summary?.avg_confidence == null
                                    ? "-"
                                    : summary.avg_confidence.toFixed(3)}
                                </td>
                                <td>{formatMoney(summary?.total_estimated_cost)}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </section>

                  <section className="raw-section" style={{ marginBottom: 20 }}>
                    <h3>Cost Summary</h3>
                    <p className="chart-desc">
                      Economic rollup for the full run, including model, variant, and segment-level
                      cost totals.
                    </p>

                    <div className="analysis-grid three-up">
                      <article className="summary-card">
                        <p className="card-label">Run Total</p>
                        <p className="card-value">{formatMoney(totalRunCost)}</p>
                        <span className="card-sublabel">Summed from result-level estimated_cost</span>
                      </article>
                      <article className="summary-card">
                        <p className="card-label">Cost / Segment</p>
                        <p className="card-value">
                          {runData.segments.length > 0
                            ? formatMoney(totalRunCost / runData.segments.length)
                            : "-"}
                        </p>
                        <span className="card-sublabel">
                          Average across {runData.segments.length} segments
                        </span>
                      </article>
                      <article className="summary-card">
                        <p className="card-label">Costed Rows</p>
                        <p className="card-value">
                          {
                            runData.results.filter((result) => result.estimated_cost != null).length
                          }
                        </p>
                        <span className="card-sublabel">
                          Results with non-null estimated cost
                        </span>
                      </article>
                    </div>

                    <div className="analysis-grid three-up">
                      <div className="table-scroll">
                        <table className="data-table">
                          <thead>
                            <tr>
                              <th>Model</th>
                              <th>Total Cost</th>
                            </tr>
                          </thead>
                          <tbody>
                            {costPerModelRows.map((row) => (
                              <tr key={row.model}>
                                <td>{row.model}</td>
                                <td>{formatMoney(row.totalCost)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>

                      <div className="table-scroll">
                        <table className="data-table">
                          <thead>
                            <tr>
                              <th>Variant</th>
                              <th>Total Cost</th>
                            </tr>
                          </thead>
                          <tbody>
                            {costPerVariantRows.map((row) => (
                              <tr key={row.variant}>
                                <td>{row.variant}</td>
                                <td>{formatMoney(row.totalCost)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>

                      <div className="table-scroll">
                        <table className="data-table">
                          <thead>
                            <tr>
                              <th>Segment</th>
                              <th>Cost</th>
                            </tr>
                          </thead>
                          <tbody>
                            {costPerSegmentRows.map((row) => (
                              <tr key={row.segmentId}>
                                <td>
                                  {row.videoId} / {row.segmentId}
                                </td>
                                <td>{formatMoney(row.totalCost)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  </section>

                  <section className="charts-grid">
                    <article className="chart-card">
                      <h3>Parse Success</h3>
                      <p className="chart-desc">Model-level parse success rate from exported rows.</p>
                      <ParseRateChart summaries={runData.summaries} models={runData.models} />
                    </article>
                    <article className="chart-card">
                      <h3>Latency</h3>
                      <p className="chart-desc">Average and P95 latency for successful parses.</p>
                      <LatencyChart summaries={runData.summaries} models={runData.models} />
                    </article>
                    <article className="chart-card">
                      <h3>Confidence</h3>
                      <p className="chart-desc">Average confidence for successfully parsed outputs.</p>
                      <ConfidenceChart summaries={runData.summaries} models={runData.models} />
                    </article>
                    <article className="chart-card">
                      <h3>Cost</h3>
                      <p className="chart-desc">Total estimated cost per model for the selected run.</p>
                      <CostChart summaries={runData.summaries} models={runData.models} />
                    </article>
                  </section>

                  <AgreementTable title="Agreement Matrix" matrix={runData.agreement} />

                  {sweepData ? (
                    <>
                      <section className="agreement-section">
                        <h3>Model x Variant Heatmap</h3>
                        <p className="chart-desc">
                          Parse success rate by model and extraction variant. Pre-computed sweep
                          summary is used when available; otherwise the frontend derives these
                          metrics from raw results.
                        </p>
                        <div className="matrix-scroll">
                          <table className="agreement-table">
                            <thead>
                              <tr>
                                <th />
                                {sweepData.variants.map((variant) => (
                                  <th key={variant}>{variant}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {runData.models.map((model) => (
                                <tr key={model}>
                                  <td className="matrix-row-label">{model}</td>
                                  {sweepData.variants.map((variant) => {
                                    const value =
                                      sweepData.parse_success_matrix[model]?.[variant] ?? 0;
                                    return (
                                      <td
                                        key={`${model}-${variant}`}
                                        className="matrix-cell mono"
                                        style={heatStyle(value)}
                                      >
                                        {formatPercent(value)}
                                      </td>
                                    );
                                  })}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </section>

                      <section className="raw-section" style={{ marginBottom: 20 }}>
                        <h3>Stability Scores</h3>
                        <p className="chart-desc">
                          Self-agreement and rank stability across extraction variants.
                        </p>
                        <div className="table-scroll">
                          <table className="data-table">
                            <thead>
                              <tr>
                                <th>Model</th>
                                <th>Self Agreement</th>
                                <th>Rank Stability</th>
                                <th>Rank Positions</th>
                              </tr>
                            </thead>
                            <tbody>
                              {sweepData.stability.map((entry) => (
                                <tr key={entry.model_name}>
                                  <td>{entry.model_name}</td>
                                  <td>{formatPercent(entry.self_agreement)}</td>
                                  <td>{entry.rank_stability.toFixed(3)}</td>
                                  <td>{entry.rank_positions.map((value) => `#${value}`).join(", ")}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </section>

                      <section className="raw-section" style={{ marginBottom: 20 }}>
                        <h3>Cost Comparison</h3>
                        <p className="chart-desc">
                          Variant-level cost totals, aligned with the sweep analysis shown in the
                          Streamlit viewer.
                        </p>
                        <div className="table-scroll">
                          <table className="data-table">
                            <thead>
                              <tr>
                                <th>Model</th>
                                {sweepData.variants.map((variant) => (
                                  <th key={variant}>{variant}</th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {runData.models.map((model) => (
                                <tr key={model}>
                                  <td>{model}</td>
                                  {sweepData.variants.map((variant) => {
                                    const cell = sweepCellsByModelVariant.get(
                                      `${model}::${variant}`
                                    );
                                    return (
                                      <td key={`${model}-${variant}-cost`}>
                                        {formatMoney(cell?.total_estimated_cost)}
                                      </td>
                                    );
                                  })}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </section>

                      {Object.entries(sweepData.agreement_by_variant)
                        .sort(([left], [right]) => left.localeCompare(right))
                        .map(([variant, matrix]) => (
                          <AgreementTable
                            key={variant}
                            title={`Agreement Matrix: ${variant}`}
                            matrix={matrix}
                          />
                        ))}
                    </>
                  ) : (
                    <section className="agreement-section">
                      <h3>Sweep View</h3>
                      <p className="chart-desc">
                        No sweep results available for this run yet. The dashboard stays up even
                        when a sweep summary export is not present.
                      </p>
                    </section>
                  )}

                  <section className="raw-section">
                    <h3>Model Catalog</h3>
                    <p className="chart-desc">Current model metadata returned by the frontend API.</p>
                    <div className="table-scroll">
                      <table className="catalog-table">
                        <thead>
                          <tr>
                            <th>Name</th>
                            <th>Provider</th>
                            <th>Model ID</th>
                            <th>Notes</th>
                          </tr>
                        </thead>
                        <tbody>
                          {catalog.map((model) => (
                            <tr key={model.name}>
                              <td>{model.name}</td>
                              <td>{model.provider}</td>
                              <td className="mono">{model.model_id}</td>
                              <td>{model.notes || "-"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </section>
                </div>
              ) : null}

              {tab === "segments" ? (
                <div className="tab-content">
                  {sweepData ? (
                    <section className="raw-section" style={{ marginBottom: 20 }}>
                      <h3>Variant Filter</h3>
                      <p className="chart-desc">
                        Filter segment comparisons to a specific extraction variant when sweep data
                        is available.
                      </p>
                      <select
                        value={variantFilter}
                        onChange={(event) => setVariantFilter(event.target.value)}
                      >
                        <option value={ALL_VARIANTS}>{ALL_VARIANTS}</option>
                        {sweepData.variants.map((variant) => (
                          <option key={variant} value={variant}>
                            {variant}
                          </option>
                        ))}
                      </select>
                    </section>
                  ) : null}

                  <div className="segment-strip">
                    {runData.segments.map((segment) => (
                      <button
                        key={segment.segment_id}
                        type="button"
                        className={`segment-chip ${
                          activeSegmentId === segment.segment_id ? "active" : ""
                        }`}
                        onClick={() => setActiveSegmentId(segment.segment_id)}
                      >
                        <strong>Segment {segment.segment_index + 1}</strong>
                        <span>
                          {formatTime(segment.start_time_s)} - {formatTime(segment.end_time_s)}
                        </span>
                      </button>
                    ))}
                  </div>

                  {activeSegment ? (
                    <section className="segment-detail">
                      <div className="segment-info-bar">
                        <h3>{activeSegment.segment_id}</h3>
                        <p>
                          {formatTime(activeSegment.start_time_s)} -{" "}
                          {formatTime(activeSegment.end_time_s)} | Duration{" "}
                          {Math.round(activeSegment.duration_s)}s
                        </p>
                      </div>

                      {segmentMedia?.contact_sheet_data_url ? (
                        <img
                          src={segmentMedia.contact_sheet_data_url}
                          alt={`Contact sheet for ${activeSegment.segment_id}`}
                          className="contact-sheet"
                        />
                      ) : null}

                      {segmentMedia?.frames.length ? (
                        <div className="frame-grid">
                          {segmentMedia.frames.map((frame) => (
                            <figure
                              key={`${activeSegment.segment_id}-${frame.timestamp_s}`}
                              className="frame-card"
                            >
                              {frame.data_url ? (
                                <img
                                  src={frame.data_url}
                                  alt={`Frame at ${frame.timestamp_s.toFixed(2)}s`}
                                />
                              ) : (
                                <div className="empty-state">Frame unavailable</div>
                              )}
                              <figcaption>{frame.timestamp_s.toFixed(2)}s</figcaption>
                            </figure>
                          ))}
                        </div>
                      ) : (
                        <p className="empty-state">
                          No extracted frame previews found for this segment.
                        </p>
                      )}

                      <div className="comparison-grid">
                        {activeResults.map((result) => (
                          <article
                            key={`${result.model_name}-${result.segment_id}-${resultVariantLabel(
                              result
                            )}`}
                            className="result-card"
                          >
                            <header>
                              <p className="result-model">{result.model_name}</p>
                              <span
                                className={`parse-badge ${
                                  result.parsed_success ? "ok" : "fail"
                                }`}
                              >
                                {result.parsed_success ? "Parsed" : "Failed"}
                              </span>
                            </header>
                            <div className="result-body">
                              <div className="result-field">
                                <h4>Primary Action</h4>
                                <p className="action-text">{result.primary_action || "-"}</p>
                              </div>

                              {result.secondary_actions.length ? (
                                <div className="result-field">
                                  <h4>Secondary Actions</h4>
                                  <div className="tag-list">
                                    {result.secondary_actions.map((action) => (
                                      <span
                                        key={`${result.model_name}-${action}`}
                                        className="action-tag"
                                      >
                                        {action}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              ) : null}

                              {result.objects.length ? (
                                <div className="result-field">
                                  <h4>Objects</h4>
                                  <div className="tag-list">
                                    {result.objects.map((objectName) => (
                                      <span
                                        key={`${result.model_name}-${objectName}`}
                                        className="object-tag"
                                      >
                                        {objectName}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              ) : null}

                              <div className="result-field">
                                <h4>Description</h4>
                                <p>{result.description || result.parse_error || "-"}</p>
                              </div>

                              <div className="mini-stats">
                                <span>{formatLatency(result.latency_ms)}</span>
                                <span>{formatMoney(result.estimated_cost)}</span>
                                <span>
                                  {result.confidence == null ? "-" : result.confidence.toFixed(3)}
                                </span>
                                <span>{resultVariantLabel(result)}</span>
                              </div>
                            </div>
                          </article>
                        ))}
                      </div>
                    </section>
                  ) : (
                    <p className="empty-state">Select a segment to compare model outputs.</p>
                  )}
                </div>
              ) : null}

              {tab === "raw" ? (
                <section className="raw-section tab-content">
                  <h3>Raw Results</h3>
                  <p className="chart-desc">
                    Loaded from {runData.run_id}_results.json when available, with CSV fallback for
                    older runs.
                  </p>
                  <div className="table-scroll">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Segment</th>
                          <th>Model</th>
                          <th>Variant</th>
                          <th>Parsed</th>
                          <th>Primary Action</th>
                          <th>Latency</th>
                          <th>Cost</th>
                        </tr>
                      </thead>
                      <tbody>
                        {runData.results.map((result) => (
                          <tr
                            key={`${result.model_name}-${result.segment_id}-${result.timestamp ?? ""}`}
                          >
                            <td className="mono">{result.segment_id}</td>
                            <td>{result.model_name}</td>
                            <td>{resultVariantLabel(result)}</td>
                            <td>
                              <span
                                className={`parse-badge small ${
                                  result.parsed_success ? "ok" : "fail"
                                }`}
                              >
                                {result.parsed_success ? "yes" : "no"}
                              </span>
                            </td>
                            <td>{result.primary_action || "-"}</td>
                            <td>{formatLatency(result.latency_ms)}</td>
                            <td>{formatMoney(result.estimated_cost)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              ) : null}
            </>
          )}
        </main>
      </div>
    </div>
  );
}
