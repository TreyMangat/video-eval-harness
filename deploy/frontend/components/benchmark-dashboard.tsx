"use client";

import type { CSSProperties, FormEvent } from "react";
import { useEffect, useMemo, useState } from "react";

import { DEFAULT_MODEL_CATALOG, DEFAULT_MODEL_SELECTION } from "../lib/model-catalog";
import { DEMO_RUNS, DEMO_RUN_LIST } from "../lib/demo-data";
import type {
  LabelResult,
  ModelCatalogItem,
  RunListItem,
  RunPayload,
  SegmentMedia,
  SegmentSummary,
  SweepMetrics,
} from "../lib/types";
import { ConfidenceChart, CostChart, LatencyChart, ParseRateChart } from "./charts";

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
const ALL_VARIANTS = "All variants";

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

function sweepFor(runData: RunPayload | null): SweepMetrics | null {
  if (!runData?.sweep?.has_sweep) {
    return null;
  }
  return runData.sweep;
}

function resultVariantLabel(result: LabelResult): string {
  if (result.extraction_variant_id) {
    return result.extraction_label || result.extraction_variant_id;
  }
  return "default";
}

export function BenchmarkDashboard() {
  const [mode, setMode] = useState<DataMode>("live");
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
  const [variantFilter, setVariantFilter] = useState<string>(ALL_VARIANTS);

  const sweepData = useMemo(() => sweepFor(runData), [runData]);
  const activeVariantId = useMemo(() => {
    if (!sweepData || variantFilter === ALL_VARIANTS) {
      return null;
    }
    return sweepData.variant_id_by_label[variantFilter] ?? null;
  }, [sweepData, variantFilter]);

  useEffect(() => {
    if (mode === "demo") {
      setRuns(DEMO_RUN_LIST);
      setRunData(DEMO_RUNS[0]);
      setActiveSegmentId(DEMO_RUNS[0].segments[0]?.segment_id ?? null);
      setSegmentMedia(null);
      setCatalog(DEFAULT_MODEL_CATALOG);
      setVariantFilter(ALL_VARIANTS);
      setJob({ callId: null, status: "idle", runId: null, error: null });
      setError(null);
      return;
    }

    setRunData(null);
    setActiveSegmentId(null);
    setSegmentMedia(null);
    setVariantFilter(ALL_VARIANTS);
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
    setVariantFilter(ALL_VARIANTS);
  }, [runData?.run_id]);

  useEffect(() => {
    if (!runData || !activeSegmentId || mode === "demo") {
      setSegmentMedia(null);
      return;
    }
    void loadSegmentMedia(runData.run_id, activeSegmentId, activeVariantId);
  }, [runData, activeSegmentId, mode, activeVariantId]);

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
    } catch (loadError) {
      setRuns([]);
      setRunData(null);
      setActiveSegmentId(null);
      setError(loadError instanceof Error ? loadError.message : "Failed to load runs");
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

  async function loadSegmentMedia(runId: string, segmentId: string, variantId: string | null) {
    try {
      const params = new URLSearchParams();
      if (variantId) {
        params.set("variantId", variantId);
      }
      const suffix = params.toString() ? `?${params.toString()}` : "";
      const data = await readJson<SegmentMedia>(
        `/api/runs/${runId}/segments/${segmentId}/media${suffix}`
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
      setError("Switch to Live data mode before launching a benchmark.");
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
