"use client";

import type { CSSProperties } from "react";
import { useEffect, useMemo, useState } from "react";

import type {
  LabelResult,
  ModelCatalogItem,
  RunListItem,
  RunPayload,
  SegmentMedia,
  SegmentSummary,
} from "../lib/types";
import { DEMO_RUNS, DEMO_RUN_LIST } from "../lib/demo-data";
import { LatencyChart, CostChart, ConfidenceChart, ParseRateChart } from "./charts";

type DataMode = "demo" | "live";

async function readJson<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error ?? data.detail ?? "Request failed");
  }
  return data as T;
}

function formatPercent(value: number | null | undefined): string {
  if (value == null) return "-";
  return `${Math.round(value * 100)}%`;
}

function formatMoney(value: number | null | undefined): string {
  if (value == null) return "-";
  return `$${value.toFixed(4)}`;
}

function formatLatency(value: number | null | undefined): string {
  if (value == null) return "-";
  return `${Math.round(value)} ms`;
}

function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60)
    .toString()
    .padStart(2, "0");
  return `${mins}:${secs}`;
}

function heatColor(value: number): string {
  if (value >= 0.8) return "rgba(34,197,94,0.25)";
  if (value >= 0.5) return "rgba(245,158,11,0.2)";
  if (value >= 0.3) return "rgba(245,158,11,0.12)";
  return "rgba(239,68,68,0.1)";
}

function heatStyle(value: number): CSSProperties {
  return { background: heatColor(value) };
}

type Tab = "overview" | "segments" | "raw";

export function BenchmarkDashboard() {
  const [mode, setMode] = useState<DataMode>("demo");
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [runData, setRunData] = useState<RunPayload | null>(null);
  const [segmentMedia, setSegmentMedia] = useState<SegmentMedia | null>(null);
  const [activeSegmentId, setActiveSegmentId] = useState<string | null>(null);
  const [loadingRun, setLoadingRun] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("overview");

  // Initialize with demo data
  useEffect(() => {
    if (mode === "demo") {
      setRuns(DEMO_RUN_LIST);
      setRunData(DEMO_RUNS[0]);
      setActiveSegmentId(DEMO_RUNS[0].segments[0]?.segment_id ?? null);
      setError(null);
    } else {
      void loadRuns();
    }
  }, [mode]);

  // Load segment media when segment changes (live mode only)
  useEffect(() => {
    if (!runData || !activeSegmentId || mode === "demo") {
      setSegmentMedia(null);
      return;
    }
    void loadSegmentMedia(runData.run_id, activeSegmentId);
  }, [runData, activeSegmentId, mode]);

  const activeSegment = useMemo<SegmentSummary | null>(() => {
    if (!runData || !activeSegmentId) return null;
    return runData.segments.find((s) => s.segment_id === activeSegmentId) ?? null;
  }, [runData, activeSegmentId]);

  const activeResults = useMemo<LabelResult[]>(() => {
    if (!runData || !activeSegmentId) return [];
    return runData.results.filter((r) => r.segment_id === activeSegmentId);
  }, [runData, activeSegmentId]);

  async function loadRuns() {
    try {
      const data = await readJson<RunListItem[]>("/api/runs");
      setRuns(data);
    } catch {
      setError("Could not connect to backend. Switch to Demo mode to explore.");
    }
  }

  function loadDemoRun(runId: string) {
    const run = DEMO_RUNS.find((r) => r.run_id === runId);
    if (run) {
      setRunData(run);
      setActiveSegmentId(run.segments[0]?.segment_id ?? null);
      setTab("overview");
    }
  }

  async function loadLiveRun(runId: string) {
    try {
      setLoadingRun(true);
      setError(null);
      const data = await readJson<RunPayload>(`/api/runs/${runId}`);
      setRunData(data);
      setActiveSegmentId(data.segments[0]?.segment_id ?? null);
      setTab("overview");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load run");
    } finally {
      setLoadingRun(false);
    }
  }

  async function loadSegmentMedia(runId: string, segmentId: string) {
    try {
      const data = await readJson<SegmentMedia>(
        `/api/runs/${runId}/segments/${segmentId}/media`
      );
      setSegmentMedia(data);
    } catch {
      setSegmentMedia(null);
    }
  }

  function selectRun(runId: string) {
    if (mode === "demo") {
      loadDemoRun(runId);
    } else {
      void loadLiveRun(runId);
    }
  }

  return (
    <main className="page-shell">
      {/* Header */}
      <header className="top-bar">
        <div className="top-bar-left">
          <h1 className="logo">VBench</h1>
          <span className="logo-sub">Model Comparison Dashboard</span>
        </div>
        <div className="mode-toggle">
          <button
            className={`mode-btn ${mode === "demo" ? "active" : ""}`}
            onClick={() => setMode("demo")}
          >
            Demo Data
          </button>
          <button
            className={`mode-btn ${mode === "live" ? "active" : ""}`}
            onClick={() => setMode("live")}
          >
            Live Backend
          </button>
        </div>
      </header>

      {error && <p className="error-banner">{error}</p>}

      <div className="main-layout">
        {/* Sidebar - Run List */}
        <aside className="sidebar">
          <div className="sidebar-heading">
            <h2>Benchmark Runs</h2>
            <span className="badge">{runs.length}</span>
          </div>
          <div className="run-list">
            {runs.map((run) => (
              <button
                key={run.run_id}
                className={`run-item ${runData?.run_id === run.run_id ? "active" : ""}`}
                onClick={() => selectRun(run.run_id)}
              >
                <div className="run-item-top">
                  <strong>{run.run_id.slice(0, 16)}</strong>
                  <span className="run-date">
                    {new Date(run.created_at).toLocaleDateString()}
                  </span>
                </div>
                <div className="run-item-models">
                  {run.models.map((m) => (
                    <span key={m} className="model-tag">
                      {m}
                    </span>
                  ))}
                </div>
              </button>
            ))}
            {runs.length === 0 && <p className="empty-state">No runs available.</p>}
          </div>
        </aside>

        {/* Main Content */}
        <div className="content">
          {loadingRun && <p className="loading-copy">Loading run data...</p>}

          {runData && !loadingRun && (
            <>
              {/* Run Header */}
              <div className="run-header">
                <div>
                  <h2 className="run-title">
                    {runData.videos?.[0]
                      ? (runData.videos[0] as { filename?: string }).filename ??
                        runData.run_id
                      : runData.run_id}
                  </h2>
                  <p className="run-meta">
                    {runData.models.length} models &middot;{" "}
                    {runData.segments.length} segments &middot;{" "}
                    {runData.config.prompt_version} prompt &middot;{" "}
                    {new Date(runData.config.created_at).toLocaleString()}
                  </p>
                </div>
                <div className="tab-bar">
                  <button
                    className={`tab-btn ${tab === "overview" ? "active" : ""}`}
                    onClick={() => setTab("overview")}
                  >
                    Overview
                  </button>
                  <button
                    className={`tab-btn ${tab === "segments" ? "active" : ""}`}
                    onClick={() => setTab("segments")}
                  >
                    Segment Explorer
                  </button>
                  <button
                    className={`tab-btn ${tab === "raw" ? "active" : ""}`}
                    onClick={() => setTab("raw")}
                  >
                    Raw Data
                  </button>
                </div>
              </div>

              {/* Overview Tab */}
              {tab === "overview" && (
                <div className="tab-content">
                  {/* Summary Cards */}
                  <div className="summary-grid">
                    {runData.models.map((model) => {
                      const s = runData.summaries[model];
                      return (
                        <article key={model} className="summary-card">
                          <p className="card-label">{model}</p>
                          <h3 className="card-value">
                            {formatPercent(s?.parse_success_rate)}
                          </h3>
                          <span className="card-sublabel">parse success</span>
                          <dl className="card-stats">
                            <div>
                              <dt>Avg Latency</dt>
                              <dd>{formatLatency(s?.avg_latency_ms)}</dd>
                            </div>
                            <div>
                              <dt>P95 Latency</dt>
                              <dd>{formatLatency(s?.p95_latency_ms)}</dd>
                            </div>
                            <div>
                              <dt>Confidence</dt>
                              <dd>{s?.avg_confidence?.toFixed(2) ?? "-"}</dd>
                            </div>
                            <div>
                              <dt>Total Cost</dt>
                              <dd>{formatMoney(s?.total_estimated_cost)}</dd>
                            </div>
                            <div>
                              <dt>Segments</dt>
                              <dd>
                                {s?.successful_parses ?? 0}/{s?.total_segments ?? 0}
                              </dd>
                            </div>
                          </dl>
                        </article>
                      );
                    })}
                  </div>

                  {/* Charts Grid */}
                  <div className="charts-grid">
                    <div className="chart-card">
                      <h3>Latency Comparison</h3>
                      <p className="chart-desc">
                        Average and P95 response times per model (ms)
                      </p>
                      <LatencyChart
                        summaries={runData.summaries}
                        models={runData.models}
                      />
                    </div>
                    <div className="chart-card">
                      <h3>Total Cost</h3>
                      <p className="chart-desc">
                        Estimated API cost across all segments (USD)
                      </p>
                      <CostChart
                        summaries={runData.summaries}
                        models={runData.models}
                      />
                    </div>
                    <div className="chart-card">
                      <h3>Average Confidence</h3>
                      <p className="chart-desc">
                        Self-reported model confidence (0-1)
                      </p>
                      <ConfidenceChart
                        summaries={runData.summaries}
                        models={runData.models}
                      />
                    </div>
                    <div className="chart-card">
                      <h3>Parse Success Rate</h3>
                      <p className="chart-desc">
                        Percentage of segments with valid JSON output
                      </p>
                      <ParseRateChart
                        summaries={runData.summaries}
                        models={runData.models}
                      />
                    </div>
                  </div>

                  {/* Agreement Matrix */}
                  {runData.models.length > 1 && (
                    <div className="agreement-section">
                      <h3>Primary Action Agreement Matrix</h3>
                      <p className="chart-desc">
                        How often each pair of models agree on the primary action label
                      </p>
                      <div className="matrix-scroll">
                        <table className="agreement-table">
                          <thead>
                            <tr>
                              <th />
                              {runData.models.map((m) => (
                                <th key={m}>{m}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {runData.models.map((rowModel) => (
                              <tr key={rowModel}>
                                <td className="matrix-row-label">{rowModel}</td>
                                {runData.models.map((colModel) => {
                                  const val =
                                    runData.agreement[rowModel]?.[colModel] ?? 0;
                                  return (
                                    <td
                                      key={colModel}
                                      className="matrix-cell"
                                      style={heatStyle(val)}
                                    >
                                      {formatPercent(val)}
                                    </td>
                                  );
                                })}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Segments Tab */}
              {tab === "segments" && (
                <div className="tab-content">
                  <div className="segment-strip">
                    {runData.segments.map((seg) => (
                      <button
                        key={seg.segment_id}
                        className={`segment-chip ${
                          seg.segment_id === activeSegmentId ? "active" : ""
                        }`}
                        onClick={() => setActiveSegmentId(seg.segment_id)}
                      >
                        <strong>#{seg.segment_index + 1}</strong>
                        <span>
                          {formatTime(seg.start_time_s)} -{" "}
                          {formatTime(seg.end_time_s)}
                        </span>
                      </button>
                    ))}
                  </div>

                  {activeSegment && (
                    <div className="segment-detail">
                      <div className="segment-info-bar">
                        <h3>
                          {activeSegment.video_filename ?? activeSegment.video_id}
                        </h3>
                        <p>
                          {formatTime(activeSegment.start_time_s)} -{" "}
                          {formatTime(activeSegment.end_time_s)} &middot;{" "}
                          {activeSegment.duration_s.toFixed(1)}s &middot;{" "}
                          {activeSegment.frame_count} frames &middot;{" "}
                          {activeSegment.segmentation_mode}
                        </p>
                      </div>

                      {/* Frame previews (live mode only) */}
                      {segmentMedia?.contact_sheet_data_url && (
                        <img
                          className="contact-sheet"
                          alt="Contact sheet"
                          src={segmentMedia.contact_sheet_data_url}
                        />
                      )}

                      {segmentMedia?.frames && segmentMedia.frames.length > 0 && (
                        <div className="frame-grid">
                          {segmentMedia.frames.map((frame, i) =>
                            frame.data_url ? (
                              <figure key={i} className="frame-card">
                                <img
                                  src={frame.data_url}
                                  alt={`Frame at ${frame.timestamp_s}s`}
                                />
                                <figcaption>{frame.timestamp_s.toFixed(1)}s</figcaption>
                              </figure>
                            ) : null
                          )}
                        </div>
                      )}

                      {/* Model comparison cards */}
                      <div className="comparison-grid">
                        {runData.models.map((model) => {
                          const result = activeResults.find(
                            (r) => r.model_name === model
                          );
                          return (
                            <article key={model} className="result-card">
                              <header>
                                <p className="result-model">{model}</p>
                                <span
                                  className={`parse-badge ${
                                    result?.parsed_success ? "ok" : "fail"
                                  }`}
                                >
                                  {result?.parsed_success ? "parsed" : "error"}
                                </span>
                              </header>

                              {!result ? (
                                <p className="empty-state">No result</p>
                              ) : !result.parsed_success ? (
                                <p className="error-copy">
                                  {result.parse_error ?? "Parse failed"}
                                </p>
                              ) : (
                                <div className="result-body">
                                  <div className="result-field">
                                    <h4>Primary Action</h4>
                                    <p className="action-text">
                                      {result.primary_action ?? "-"}
                                    </p>
                                  </div>
                                  <div className="result-field">
                                    <h4>Description</h4>
                                    <p>{result.description ?? "-"}</p>
                                  </div>
                                  {result.secondary_actions.length > 0 && (
                                    <div className="result-field">
                                      <h4>Secondary Actions</h4>
                                      <div className="tag-list">
                                        {result.secondary_actions.map((a) => (
                                          <span key={a} className="action-tag">
                                            {a}
                                          </span>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                  {result.objects.length > 0 && (
                                    <div className="result-field">
                                      <h4>Objects</h4>
                                      <div className="tag-list">
                                        {result.objects.map((o) => (
                                          <span key={o} className="object-tag">
                                            {o}
                                          </span>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                  {result.environment_context && (
                                    <div className="result-field">
                                      <h4>Environment</h4>
                                      <p>{result.environment_context}</p>
                                    </div>
                                  )}
                                  <div className="mini-stats">
                                    <span>
                                      Confidence:{" "}
                                      {result.confidence?.toFixed(2) ?? "-"}
                                    </span>
                                    <span>
                                      Latency: {formatLatency(result.latency_ms)}
                                    </span>
                                    <span>
                                      Cost: {formatMoney(result.estimated_cost)}
                                    </span>
                                  </div>
                                </div>
                              )}
                            </article>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Raw Data Tab */}
              {tab === "raw" && (
                <div className="tab-content">
                  <div className="raw-section">
                    <h3>All Results</h3>
                    <p className="chart-desc">
                      {runData.results.length} label results across{" "}
                      {runData.models.length} models
                    </p>
                    <div className="table-scroll">
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Segment</th>
                            <th>Model</th>
                            <th>Primary Action</th>
                            <th>Confidence</th>
                            <th>Latency</th>
                            <th>Cost</th>
                            <th>Parsed</th>
                          </tr>
                        </thead>
                        <tbody>
                          {runData.results.map((r, i) => (
                            <tr key={i}>
                              <td className="mono">
                                {formatTime(r.start_time_s)}-{formatTime(r.end_time_s)}
                              </td>
                              <td>{r.model_name}</td>
                              <td>{r.primary_action ?? "-"}</td>
                              <td>{r.confidence?.toFixed(2) ?? "-"}</td>
                              <td>{formatLatency(r.latency_ms)}</td>
                              <td>{formatMoney(r.estimated_cost)}</td>
                              <td>
                                <span
                                  className={`parse-badge small ${
                                    r.parsed_success ? "ok" : "fail"
                                  }`}
                                >
                                  {r.parsed_success ? "OK" : "FAIL"}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}

          {!runData && !loadingRun && (
            <div className="empty-hero">
              <h2>Select a benchmark run</h2>
              <p>Choose a run from the sidebar to view the model comparison.</p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
