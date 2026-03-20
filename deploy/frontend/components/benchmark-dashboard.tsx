"use client";

import type { CSSProperties, FormEvent } from "react";
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
import { DEFAULT_MODEL_CATALOG, DEFAULT_MODEL_SELECTION } from "../lib/model-catalog";
import { LatencyChart, CostChart, ConfidenceChart, ParseRateChart } from "./charts";

type DataMode = "demo" | "live";
type Tab = "overview" | "segments" | "raw";
type BenchmarkStatus = "idle" | "queued" | "running" | "completed" | "failed";

type BenchmarkFormState = {
  videoUrl: string;
  videoName: string;
  segmentationMode: string;
  windowSize: string;
  stride: string;
  numFrames: string;
  promptVersion: string;
  models: string[];
};

type BenchmarkSubmissionResponse = {
  call_id: string;
  status: BenchmarkStatus | string;
};

type BenchmarkJobResponse = {
  call_id: string;
  status: BenchmarkStatus | string;
  result?: RunPayload;
  error?: string;
};

type BenchmarkJobState = {
  callId: string | null;
  status: BenchmarkStatus;
  runId: string | null;
  error: string | null;
};

const PROMPT_OPTIONS = ["concise", "rich", "strict_json"];
const SEGMENTATION_OPTIONS = [
  { value: "fixed_window", label: "Fixed Window" },
  { value: "scene_heuristic", label: "Scene Heuristic" },
];

function seededDefaultForm(models: string[]): BenchmarkFormState {
  return {
    videoUrl: "",
    videoName: "",
    segmentationMode: "fixed_window",
    windowSize: "10",
    stride: "",
    numFrames: "8",
    promptVersion: "concise",
    models: models.length > 0 ? models : [...DEFAULT_MODEL_SELECTION],
  };
}

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

function formatStatus(status: BenchmarkStatus): string {
  switch (status) {
    case "queued":
      return "Queued";
    case "running":
      return "Running";
    case "completed":
      return "Completed";
    case "failed":
      return "Failed";
    default:
      return "Idle";
  }
}

function inferVideoName(videoUrl: string): string {
  try {
    const url = new URL(videoUrl);
    const lastPathSegment = url.pathname.split("/").filter(Boolean).pop();
    return lastPathSegment || "video.mp4";
  } catch {
    return "video.mp4";
  }
}

function sortRunList(items: RunListItem[]): RunListItem[] {
  return [...items].sort(
    (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
  );
}

function toRunListItem(run: RunPayload): RunListItem {
  return {
    run_id: run.run_id,
    created_at: run.config.created_at,
    models: run.models,
    prompt_version: run.config.prompt_version,
    video_ids: run.config.video_ids,
  };
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

export function BenchmarkDashboard() {
  const [mode, setMode] = useState<DataMode>("demo");
  const [runs, setRuns] = useState<RunListItem[]>([]);
  const [runData, setRunData] = useState<RunPayload | null>(null);
  const [segmentMedia, setSegmentMedia] = useState<SegmentMedia | null>(null);
  const [activeSegmentId, setActiveSegmentId] = useState<string | null>(null);
  const [catalog, setCatalog] = useState<ModelCatalogItem[]>(DEFAULT_MODEL_CATALOG);
  const [form, setForm] = useState<BenchmarkFormState>(() =>
    seededDefaultForm(DEFAULT_MODEL_SELECTION)
  );
  const [job, setJob] = useState<BenchmarkJobState>({
    callId: null,
    status: "idle",
    runId: null,
    error: null,
  });
  const [submitting, setSubmitting] = useState(false);
  const [loadingRun, setLoadingRun] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("overview");

  useEffect(() => {
    if (mode === "demo") {
      setRuns(DEMO_RUN_LIST);
      setRunData(DEMO_RUNS[0]);
      setActiveSegmentId(DEMO_RUNS[0].segments[0]?.segment_id ?? null);
      setSegmentMedia(null);
      setCatalog(DEFAULT_MODEL_CATALOG);
      setJob({ callId: null, status: "idle", runId: null, error: null });
      setError(null);
      return;
    }

    setRunData(null);
    setActiveSegmentId(null);
    setSegmentMedia(null);
    setTab("overview");
    setError(null);
    void loadCatalog();
    void loadRuns(true);
  }, [mode]);

  useEffect(() => {
    setForm((current) => {
      const availableModels = catalog.map((item) => item.name);
      const nextModels = current.models.filter((name) => availableModels.includes(name));
      return {
        ...current,
        models: nextModels.length > 0 ? nextModels : availableModels,
      };
    });
  }, [catalog]);

  useEffect(() => {
    if (!runData || !activeSegmentId || mode === "demo") {
      setSegmentMedia(null);
      return;
    }
    void loadSegmentMedia(runData.run_id, activeSegmentId);
  }, [runData, activeSegmentId, mode]);

  useEffect(() => {
    if (mode !== "live" || !job.callId || !["queued", "running"].includes(job.status)) {
      return;
    }

    let cancelled = false;

    async function pollJob() {
      try {
        const data = await readJson<BenchmarkJobResponse>(`/api/benchmarks/${job.callId}`);
        if (cancelled) return;

        if (data.status === "completed" && data.result) {
          const nextRun = data.result;
          setJob({
            callId: data.call_id,
            status: "completed",
            runId: nextRun.run_id,
            error: null,
          });
          setRuns((current) =>
            sortRunList([
              toRunListItem(nextRun),
              ...current.filter((item) => item.run_id !== nextRun.run_id),
            ])
          );
          setRunData(nextRun);
          setActiveSegmentId(nextRun.segments[0]?.segment_id ?? null);
          setTab("overview");
          setError(null);
          return;
        }

        if (data.status === "failed") {
          setJob({
            callId: data.call_id,
            status: "failed",
            runId: null,
            error: data.error ?? "Benchmark failed",
          });
          return;
        }

        setJob((current) => ({
          ...current,
          status: data.status === "queued" ? "queued" : "running",
        }));
      } catch (pollError) {
        if (cancelled) return;
        setJob((current) => ({
          ...current,
          status: "failed",
          error:
            pollError instanceof Error ? pollError.message : "Failed to poll benchmark job",
        }));
      }
    }

    void pollJob();
    const intervalId = window.setInterval(() => {
      void pollJob();
    }, 4000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [job.callId, job.status, mode]);

  const activeSegment = useMemo<SegmentSummary | null>(() => {
    if (!runData || !activeSegmentId) return null;
    return runData.segments.find((segment) => segment.segment_id === activeSegmentId) ?? null;
  }, [runData, activeSegmentId]);

  const activeResults = useMemo<LabelResult[]>(() => {
    if (!runData || !activeSegmentId) return [];
    return runData.results.filter((result) => result.segment_id === activeSegmentId);
  }, [runData, activeSegmentId]);

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
      const data = sortRunList(await readJson<RunListItem[]>("/api/runs"));
      setRuns(data);

      if (autoSelectNewest && data[0]) {
        await loadLiveRun(data[0].run_id);
      }

      if (data.length === 0) {
        setRunData(null);
        setActiveSegmentId(null);
      }
    } catch {
      setRuns([]);
      setRunData(null);
      setActiveSegmentId(null);
      setError("Could not connect to the Modal backend. Switch to Demo mode or configure MODAL_API_BASE_URL.");
    }
  }

  function loadDemoRun(runId: string) {
    const run = DEMO_RUNS.find((item) => item.run_id === runId);
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
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : "Failed to load run");
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

  function updateFormField<K extends keyof BenchmarkFormState>(
    key: K,
    value: BenchmarkFormState[K]
  ) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  function toggleModel(modelName: string) {
    setForm((current) => {
      const hasModel = current.models.includes(modelName);
      const nextModels = hasModel
        ? current.models.filter((name) => name !== modelName)
        : [...current.models, modelName];
      return {
        ...current,
        models: nextModels,
      };
    });
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    if (mode !== "live") {
      setError("Switch to Live Backend mode before launching a benchmark.");
      return;
    }

    const videoUrl = form.videoUrl.trim();
    if (!videoUrl) {
      setError("Paste a public or pre-signed video URL before starting a benchmark.");
      return;
    }

    if (form.models.length === 0) {
      setError("Select at least one model to compare.");
      return;
    }

    const windowSize = Number(form.windowSize);
    const stride = form.stride.trim() ? Number(form.stride) : null;
    const numFrames = Number(form.numFrames);

    if (!Number.isFinite(windowSize) || windowSize <= 0) {
      setError("Window size must be a positive number.");
      return;
    }

    if (stride != null && (!Number.isFinite(stride) || stride <= 0)) {
      setError("Stride must be blank or a positive number.");
      return;
    }

    if (!Number.isFinite(numFrames) || numFrames <= 0) {
      setError("Frames per segment must be a positive number.");
      return;
    }

    const payload = {
      video_url: videoUrl,
      video_name: form.videoName.trim() || inferVideoName(videoUrl),
      models: form.models,
      segmentation_mode: form.segmentationMode,
      window_size: windowSize,
      stride,
      num_frames: numFrames,
      prompt_version: form.promptVersion,
      max_concurrency: 2,
    };

    try {
      setSubmitting(true);
      setError(null);
      const data = await readJson<BenchmarkSubmissionResponse>("/api/benchmarks", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      setJob({
        callId: data.call_id,
        status: data.status === "queued" ? "queued" : "running",
        runId: null,
        error: null,
      });
    } catch (submitError) {
      setJob({
        callId: null,
        status: "failed",
        runId: null,
        error: submitError instanceof Error ? submitError.message : "Failed to submit benchmark",
      });
      setError(
        submitError instanceof Error ? submitError.message : "Failed to submit benchmark"
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="page-shell">
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

      <section className="setup-grid">
        <article className="setup-card setup-card-form">
          <div className="setup-card-head">
            <div>
              <p className="eyebrow">Clip Ingestion</p>
              <h2>Launch a Benchmark Run</h2>
            </div>
            <span className={`status-chip ${mode === "live" ? "live" : "demo"}`}>
              {mode === "live" ? "Live backend ready" : "Demo only"}
            </span>
          </div>
          <p className="setup-copy">
            Paste a public `.mp4` URL or a pre-signed object URL. Modal will download the
            clip, segment it, extract frames, run each selected model, and persist the run
            for the dashboard.
          </p>
          <form className="launch-form" onSubmit={handleSubmit}>
            <label className="field">
              <span>Video URL</span>
              <input
                type="url"
                placeholder="https://.../clip.mp4"
                value={form.videoUrl}
                onChange={(event) => updateFormField("videoUrl", event.target.value)}
              />
            </label>

            <div className="field-row">
              <label className="field">
                <span>Display Name</span>
                <input
                  type="text"
                  placeholder="office-walkthrough.mp4"
                  value={form.videoName}
                  onChange={(event) => updateFormField("videoName", event.target.value)}
                />
              </label>
              <label className="field">
                <span>Prompt</span>
                <select
                  value={form.promptVersion}
                  onChange={(event) => updateFormField("promptVersion", event.target.value)}
                >
                  {PROMPT_OPTIONS.map((option) => (
                    <option key={option} value={option}>
                      {option}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <div className="field-row field-row-compact">
              <label className="field">
                <span>Segmentation</span>
                <select
                  value={form.segmentationMode}
                  onChange={(event) => updateFormField("segmentationMode", event.target.value)}
                >
                  {SEGMENTATION_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Window (s)</span>
                <input
                  type="number"
                  min="1"
                  step="0.5"
                  value={form.windowSize}
                  onChange={(event) => updateFormField("windowSize", event.target.value)}
                />
              </label>
              <label className="field">
                <span>Stride (s)</span>
                <input
                  type="number"
                  min="1"
                  step="0.5"
                  placeholder="blank = no overlap"
                  value={form.stride}
                  onChange={(event) => updateFormField("stride", event.target.value)}
                />
              </label>
              <label className="field">
                <span>Frames</span>
                <input
                  type="number"
                  min="1"
                  step="1"
                  value={form.numFrames}
                  onChange={(event) => updateFormField("numFrames", event.target.value)}
                />
              </label>
            </div>

            <div className="field">
              <span>Models</span>
              <div className="model-picker">
                {catalog.map((model) => {
                  const selected = form.models.includes(model.name);
                  return (
                    <label key={model.name} className={`model-option ${selected ? "active" : ""}`}>
                      <input
                        type="checkbox"
                        checked={selected}
                        onChange={() => toggleModel(model.name)}
                      />
                      <div>
                        <strong>{model.name}</strong>
                        <p>{model.notes}</p>
                      </div>
                    </label>
                  );
                })}
              </div>
            </div>

            <div className="launch-actions">
              <button className="primary-btn" disabled={mode !== "live" || submitting}>
                {submitting ? "Submitting..." : "Start Benchmark"}
              </button>
              <p className="helper-copy">
                {mode === "live"
                  ? "Cheapest path: keep upload out of Vercel and submit a pre-hosted clip URL."
                  : "Demo mode is visual-only. Switch to Live Backend after setting MODAL_API_BASE_URL."}
              </p>
            </div>
          </form>
        </article>

        <article className="setup-card">
          <p className="eyebrow">Segmentation</p>
          <h2>How the Video Gets Broken Down</h2>
          <div className="info-list">
            <div>
              <strong>Fixed window</strong>
              <p>
                Splits the clip into windows of `window size` seconds. If `stride` is blank,
                segments are back-to-back. If `stride` is smaller than the window, the run
                creates overlapping segments.
              </p>
            </div>
            <div>
              <strong>Scene heuristic</strong>
              <p>
                Samples frames at low FPS, looks for histogram jumps that suggest a shot or
                scene change, then turns those boundaries into segments. If detection fails,
                the backend falls back to fixed windows.
              </p>
            </div>
            <div>
              <strong>Frame extraction</strong>
              <p>
                Each segment is sampled uniformly into `num_frames` still images. The public
                backend now also builds a contact sheet so the segment explorer is easier to
                inspect visually.
              </p>
            </div>
          </div>
        </article>

        <article className="setup-card setup-card-wide">
          <div className="setup-card-head">
            <div>
              <p className="eyebrow">Model Set</p>
              <h2>What the Comparison Table Is Measuring</h2>
            </div>
            {mode === "live" && (
              <button className="ghost-btn" onClick={() => void loadRuns(true)}>
                Refresh Live Runs
              </button>
            )}
          </div>
          <p className="setup-copy">
            The dashboard compares parse success, latency, cost, confidence, and inter-model
            agreement for the same segments. "Most accurate" still means proxy accuracy unless
            you add ground truth or a dedicated judge pass.
          </p>
          <div className="table-scroll">
            <table className="catalog-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Provider</th>
                  <th>Vision</th>
                  <th>Positioning</th>
                  <th>Selected</th>
                </tr>
              </thead>
              <tbody>
                {catalog.map((model) => (
                  <tr key={model.name}>
                    <td>{model.name}</td>
                    <td>{model.provider}</td>
                    <td>{model.supports_images ? "images" : "-"}</td>
                    <td>{model.notes}</td>
                    <td>{form.models.includes(model.name) ? "yes" : "no"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="job-panel">
            <div>
              <strong>Latest job status</strong>
              <p>
                {job.callId ? `${formatStatus(job.status)} - ${job.callId}` : "No live job yet."}
              </p>
            </div>
            <div>
              <strong>Result handoff</strong>
              <p>
                When the Modal job completes, this dashboard automatically opens the new run
                and adds it to the sidebar.
              </p>
            </div>
            {job.error && (
              <div>
                <strong>Last error</strong>
                <p>{job.error}</p>
              </div>
            )}
          </div>
        </article>
      </section>

      <div className="main-layout">
        <aside className="sidebar">
          <div className="sidebar-heading">
            <h2>Benchmark Runs</h2>
            <div className="sidebar-actions">
              <span className="badge">{runs.length}</span>
              {mode === "live" && (
                <button className="ghost-btn small" onClick={() => void loadRuns(true)}>
                  Refresh
                </button>
              )}
            </div>
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
                  {run.models.map((modelName) => (
                    <span key={modelName} className="model-tag">
                      {modelName}
                    </span>
                  ))}
                </div>
              </button>
            ))}
            {runs.length === 0 && (
              <p className="empty-state">
                {mode === "live"
                  ? "No live runs yet. Launch one from the clip ingestion form."
                  : "No runs available."}
              </p>
            )}
          </div>
        </aside>

        <div className="content">
          {loadingRun && <p className="loading-copy">Loading run data...</p>}

          {runData && !loadingRun && (
            <>
              <div className="run-header">
                <div>
                  <h2 className="run-title">
                    {runData.videos?.[0]
                      ? (runData.videos[0] as { filename?: string }).filename ?? runData.run_id
                      : runData.run_id}
                  </h2>
                  <p className="run-meta">
                    {runData.models.length} models &middot; {runData.segments.length} segments
                    &middot; {runData.config.prompt_version} prompt &middot;{" "}
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

              {tab === "overview" && (
                <div className="tab-content">
                  <div className="summary-grid">
                    {runData.models.map((model) => {
                      const summary = runData.summaries[model];
                      return (
                        <article key={model} className="summary-card">
                          <p className="card-label">{model}</p>
                          <h3 className="card-value">
                            {formatPercent(summary?.parse_success_rate)}
                          </h3>
                          <span className="card-sublabel">parse success</span>
                          <dl className="card-stats">
                            <div>
                              <dt>Avg Latency</dt>
                              <dd>{formatLatency(summary?.avg_latency_ms)}</dd>
                            </div>
                            <div>
                              <dt>P95 Latency</dt>
                              <dd>{formatLatency(summary?.p95_latency_ms)}</dd>
                            </div>
                            <div>
                              <dt>Confidence</dt>
                              <dd>{summary?.avg_confidence?.toFixed(2) ?? "-"}</dd>
                            </div>
                            <div>
                              <dt>Total Cost</dt>
                              <dd>{formatMoney(summary?.total_estimated_cost)}</dd>
                            </div>
                            <div>
                              <dt>Segments</dt>
                              <dd>
                                {summary?.successful_parses ?? 0}/{summary?.total_segments ?? 0}
                              </dd>
                            </div>
                          </dl>
                        </article>
                      );
                    })}
                  </div>

                  <div className="charts-grid">
                    <div className="chart-card">
                      <h3>Latency Comparison</h3>
                      <p className="chart-desc">
                        Average and P95 response times per model (ms)
                      </p>
                      <LatencyChart summaries={runData.summaries} models={runData.models} />
                    </div>
                    <div className="chart-card">
                      <h3>Total Cost</h3>
                      <p className="chart-desc">
                        Estimated API cost across all segments (USD)
                      </p>
                      <CostChart summaries={runData.summaries} models={runData.models} />
                    </div>
                    <div className="chart-card">
                      <h3>Average Confidence</h3>
                      <p className="chart-desc">
                        Self-reported model confidence (0-1)
                      </p>
                      <ConfidenceChart summaries={runData.summaries} models={runData.models} />
                    </div>
                    <div className="chart-card">
                      <h3>Parse Success Rate</h3>
                      <p className="chart-desc">
                        Percentage of segments with valid JSON output
                      </p>
                      <ParseRateChart summaries={runData.summaries} models={runData.models} />
                    </div>
                  </div>

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
                              {runData.models.map((model) => (
                                <th key={model}>{model}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {runData.models.map((rowModel) => (
                              <tr key={rowModel}>
                                <td className="matrix-row-label">{rowModel}</td>
                                {runData.models.map((colModel) => {
                                  const value = runData.agreement[rowModel]?.[colModel] ?? 0;
                                  return (
                                    <td
                                      key={colModel}
                                      className="matrix-cell"
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
                    </div>
                  )}
                </div>
              )}

              {tab === "segments" && (
                <div className="tab-content">
                  <div className="segment-strip">
                    {runData.segments.map((segment) => (
                      <button
                        key={segment.segment_id}
                        className={`segment-chip ${
                          segment.segment_id === activeSegmentId ? "active" : ""
                        }`}
                        onClick={() => setActiveSegmentId(segment.segment_id)}
                      >
                        <strong>#{segment.segment_index + 1}</strong>
                        <span>
                          {formatTime(segment.start_time_s)} - {formatTime(segment.end_time_s)}
                        </span>
                      </button>
                    ))}
                  </div>

                  {activeSegment && (
                    <div className="segment-detail">
                      <div className="segment-info-bar">
                        <h3>{activeSegment.video_filename ?? activeSegment.video_id}</h3>
                        <p>
                          {formatTime(activeSegment.start_time_s)} -{" "}
                          {formatTime(activeSegment.end_time_s)} &middot;{" "}
                          {activeSegment.duration_s.toFixed(1)}s &middot;{" "}
                          {activeSegment.frame_count} frames &middot;{" "}
                          {activeSegment.segmentation_mode}
                        </p>
                      </div>

                      {segmentMedia?.contact_sheet_data_url && (
                        <img
                          className="contact-sheet"
                          alt="Contact sheet"
                          src={segmentMedia.contact_sheet_data_url}
                        />
                      )}

                      {segmentMedia?.frames && segmentMedia.frames.length > 0 && (
                        <div className="frame-grid">
                          {segmentMedia.frames.map((frame, index) =>
                            frame.data_url ? (
                              <figure key={index} className="frame-card">
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

                      <div className="comparison-grid">
                        {runData.models.map((model) => {
                          const result = activeResults.find((item) => item.model_name === model);
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
                                        {result.secondary_actions.map((action) => (
                                          <span key={action} className="action-tag">
                                            {action}
                                          </span>
                                        ))}
                                      </div>
                                    </div>
                                  )}
                                  {result.objects.length > 0 && (
                                    <div className="result-field">
                                      <h4>Objects</h4>
                                      <div className="tag-list">
                                        {result.objects.map((objectName) => (
                                          <span key={objectName} className="object-tag">
                                            {objectName}
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
                                      Confidence: {result.confidence?.toFixed(2) ?? "-"}
                                    </span>
                                    <span>Latency: {formatLatency(result.latency_ms)}</span>
                                    <span>Cost: {formatMoney(result.estimated_cost)}</span>
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

              {tab === "raw" && (
                <div className="tab-content">
                  <div className="raw-section">
                    <h3>All Results</h3>
                    <p className="chart-desc">
                      {runData.results.length} label results across {runData.models.length} models
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
                          {runData.results.map((result, index) => (
                            <tr key={index}>
                              <td className="mono">
                                {formatTime(result.start_time_s)}-{formatTime(result.end_time_s)}
                              </td>
                              <td>{result.model_name}</td>
                              <td>{result.primary_action ?? "-"}</td>
                              <td>{result.confidence?.toFixed(2) ?? "-"}</td>
                              <td>{formatLatency(result.latency_ms)}</td>
                              <td>{formatMoney(result.estimated_cost)}</td>
                              <td>
                                <span
                                  className={`parse-badge small ${
                                    result.parsed_success ? "ok" : "fail"
                                  }`}
                                >
                                  {result.parsed_success ? "OK" : "FAIL"}
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
              <h2>No run loaded yet</h2>
              <p>
                Launch a live benchmark from the clip ingestion form or switch to Demo mode to
                inspect the comparison UI immediately.
              </p>
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
