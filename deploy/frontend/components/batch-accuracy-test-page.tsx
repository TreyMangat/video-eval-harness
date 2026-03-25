"use client";

import Link from "next/link";
import { ChangeEvent, DragEvent, useEffect, useMemo, useRef, useState } from "react";

import {
  fetchModels,
  getHealth,
  pollJob,
  uploadBatchBenchmark,
  type ApiModel,
  type BatchBenchmarkJobResponse,
  type HealthPayload,
} from "../lib/api-client";
import { DEFAULT_MODEL_CATALOG } from "../lib/model-catalog";
import { TopNav } from "./navigation";

const ACCEPTED_TYPES = [".mp4", ".avi", ".mov", ".mkv", ".webm"];
const DEFAULT_ALLOWED_MODELS = DEFAULT_MODEL_CATALOG.filter((model) => model.tier === "fast").map(
  (model) => model.name
);
const API_WARMUP_ATTEMPTS = 12;
const API_WARMUP_INTERVAL_MS = 5_000;
const BATCH_SUBMIT_WARMUP_ATTEMPTS = 6;

type Phase = "upload" | "running" | "complete";
type ServerState = "checking" | "warming" | "ready" | "unavailable";
type CsvVideo = {
  filename: string;
  labels: string[];
};
type CsvData = {
  videos: CsvVideo[];
  totalLabels: number;
};
type JobState = {
  jobId: string;
  status: "queued" | "running";
  estimatedTimeS: number;
  videoCount: number;
  stage: string | null;
  progress: string | null;
};

const DEFAULT_LIMITS: HealthPayload["limits"] = {
  max_clip_s: 60,
  max_file_size_mb: 100,
  max_models: DEFAULT_MODEL_CATALOG.length,
  allowed_models: DEFAULT_MODEL_CATALOG.map((model) => model.name),
};

type BatchAccuracyTestPageProps = {
  embedded?: boolean;
};

function normalizeFilename(value: string): string {
  return value.split(/[\\/]/).pop()?.trim().toLowerCase() ?? "";
}

function isAcceptedFile(file: File): boolean {
  return ACCEPTED_TYPES.some((suffix) => file.name.toLowerCase().endsWith(suffix));
}

function formatFileSize(file: File): string {
  return `${(file.size / 1_048_576).toFixed(1)}MB`;
}

function fallbackModelOptions(): ApiModel[] {
  return DEFAULT_MODEL_CATALOG.map((model) => ({
    name: model.name,
    display_name: model.display_name ?? model.name,
    model_id: model.model_id,
    provider: model.provider,
    supports_images: model.supports_images,
    description: model.description ?? model.notes,
    tier: model.tier,
    estimated_cost_per_segment: model.estimated_cost_per_segment,
  }));
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function isJobActive(job: JobState | null): job is JobState {
  return job != null;
}

function fallbackTier(modelName: string): "fast" | "frontier" {
  return DEFAULT_MODEL_CATALOG.find((model) => model.name === modelName)?.tier ?? "fast";
}

function modelTier(model: ApiModel | undefined, modelName?: string): "fast" | "frontier" {
  if (model?.tier) {
    return model.tier;
  }
  return fallbackTier(modelName ?? model?.name ?? "");
}

function modelCostPerSegment(model: ApiModel | undefined, modelName: string): number {
  if (typeof model?.estimated_cost_per_segment === "number") {
    return model.estimated_cost_per_segment;
  }

  const fallback = DEFAULT_MODEL_CATALOG.find((entry) => entry.name === modelName);
  if (typeof fallback?.estimated_cost_per_segment === "number") {
    return fallback.estimated_cost_per_segment;
  }

  return fallbackTier(modelName) === "frontier" ? 0.08 : 0.01;
}

function formatBackendError(error: unknown, fallback: string): string {
  const message = error instanceof Error ? error.message : fallback;

  if (message === "Network request to benchmark server failed.") {
    return "Could not reach the benchmark server. It may be starting up - please try again in 30 seconds.";
  }
  if (message.startsWith("Request timed out after")) {
    return "The benchmark server took too long to respond. It may still be starting up - please try again in 30 seconds.";
  }

  return message || fallback;
}

function splitDelimitedLine(line: string, delimiter: string): string[] {
  return line
    .split(delimiter)
    .map((part) => part.trim().replace(/^["']|["']$/g, ""))
    .filter((part, index, parts) => part.length > 0 || index < parts.length - 1);
}

function parseCsv(text: string): CsvData {
  const trimmed = text.trim();
  if (!trimmed) {
    return { videos: [], totalLabels: 0 };
  }

  const lines = trimmed
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  if (lines.length === 0) {
    return { videos: [], totalLabels: 0 };
  }

  const delimiter = lines[0].includes("\t") && !lines[0].includes(",") ? "\t" : ",";
  const firstLine = lines[0].toLowerCase();
  const hasHeader =
    firstLine.includes("video_filename") ||
    firstLine.includes("filename") ||
    firstLine.includes("label");
  const dataLines = hasHeader ? lines.slice(1) : lines;
  const videoMap = new Map<string, string[]>();

  for (const line of dataLines) {
    const parts = splitDelimitedLine(line, delimiter);
    if (parts.length < 2) {
      continue;
    }

    const filename = normalizeFilename(parts[0]);
    const label = parts[parts.length - 1]?.trim();
    if (!filename || !label) {
      continue;
    }

    const existing = videoMap.get(filename) ?? [];
    existing.push(label);
    videoMap.set(filename, existing);
  }

  const videos = [...videoMap.entries()].map(([filename, labels]) => ({
    filename,
    labels,
  }));

  return {
    videos,
    totalLabels: videos.reduce((sum, video) => sum + video.labels.length, 0),
  };
}

function mergeUploadedVideos(current: File[], nextFiles: File[]): File[] {
  const merged = new Map<string, File>();
  for (const file of current) {
    merged.set(normalizeFilename(file.name), file);
  }
  for (const file of nextFiles) {
    merged.set(normalizeFilename(file.name), file);
  }
  return [...merged.values()];
}

function findUploadedVideo(filename: string, uploadedVideos: File[]): File | null {
  const normalized = normalizeFilename(filename);
  return uploadedVideos.find((file) => normalizeFilename(file.name) === normalized) ?? null;
}

function ServerStatusBanner({
  serverState,
  warmupElapsed,
}: {
  serverState: ServerState;
  warmupElapsed: number;
}) {
  if (serverState === "checking") {
    return (
      <div className="server-status server-checking" aria-live="polite">
        <div className="status-spinner" aria-hidden="true" />
        <div className="status-text">
          <strong>Connecting to benchmark server...</strong>
          <p>This usually takes a few seconds.</p>
        </div>
      </div>
    );
  }

  if (serverState === "warming") {
    return (
      <div className="server-status server-warming" aria-live="polite">
        <div className="status-spinner" aria-hidden="true" />
        <div className="status-text">
          <strong>Benchmark server is waking up...</strong>
          <p>
            Modal serverless functions need a moment to start. {warmupElapsed}s elapsed -
            usually ready within 20-30 seconds.
          </p>
        </div>
      </div>
    );
  }

  if (serverState === "ready") {
    return (
      <div className="server-status server-ready" aria-live="polite">
        <div className="status-dot" aria-hidden="true" />
        <div className="status-text">
          <strong>Server ready</strong>
        </div>
      </div>
    );
  }

  return (
    <div className="server-status server-unavailable" aria-live="polite">
      <div className="status-text">
        <strong>Benchmark server is currently unavailable.</strong>
        <p>The server did not respond after 60 seconds. Try refreshing the page, or check back later.</p>
      </div>
    </div>
  );
}

export function BatchAccuracyTestPage({
  embedded = false,
}: BatchAccuracyTestPageProps = {}) {
  const videoInputRef = useRef<HTMLInputElement | null>(null);
  const interactiveMode = (process.env.NEXT_PUBLIC_API_URL?.trim() ?? "").length > 0;
  const [phase, setPhase] = useState<Phase>("upload");
  const [serverState, setServerState] = useState<ServerState>("checking");
  const [warmupStartTime, setWarmupStartTime] = useState<number | null>(null);
  const [warmupElapsed, setWarmupElapsed] = useState(0);
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [modelOptions, setModelOptions] = useState<ApiModel[]>(fallbackModelOptions);
  const [selectedModels, setSelectedModels] = useState<string[]>(DEFAULT_ALLOWED_MODELS);
  const [benchmarkName, setBenchmarkName] = useState("");
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [csvData, setCsvData] = useState<CsvData | null>(null);
  const [uploadedVideos, setUploadedVideos] = useState<File[]>([]);
  const [isDragActive, setIsDragActive] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [job, setJob] = useState<JobState | null>(null);
  const [completedRunId, setCompletedRunId] = useState<string | null>(null);

  const limits = health?.limits ?? DEFAULT_LIMITS;
  const csvVideos = csvData?.videos ?? [];
  const expectedVideoCount = csvVideos.length;
  const modelOptionsByName = useMemo(
    () => new Map(modelOptions.map((model) => [model.name, model])),
    [modelOptions]
  );
  const fastModels = useMemo(
    () => modelOptions.filter((model) => modelTier(model) === "fast"),
    [modelOptions]
  );
  const frontierModels = useMemo(
    () => modelOptions.filter((model) => modelTier(model) === "frontier"),
    [modelOptions]
  );
  const allVideosUploaded = useMemo(
    () =>
      expectedVideoCount > 0 &&
      csvVideos.every((video) => findUploadedVideo(video.filename, uploadedVideos) != null),
    [csvVideos, expectedVideoCount, uploadedVideos]
  );
  const orderedUploadedVideos = useMemo(
    () =>
      csvVideos
        .map((video) => findUploadedVideo(video.filename, uploadedVideos))
        .filter((file): file is File => file != null),
    [csvVideos, uploadedVideos]
  );
  const estimatedCalls = useMemo(
    () => (csvData?.totalLabels ?? 0) * selectedModels.length,
    [csvData, selectedModels.length]
  );
  const estimatedCost = useMemo(
    () =>
      selectedModels.reduce((sum, modelName) => {
        return (
          sum +
          (csvData?.totalLabels ?? 0) *
            modelCostPerSegment(modelOptionsByName.get(modelName), modelName)
        );
      }, 0),
    [csvData, modelOptionsByName, selectedModels]
  );

  function applyHealthPayload(healthPayload: HealthPayload): void {
    setHealth(healthPayload);
    setSelectedModels((current) => {
      const preserved = current
        .filter((modelName) => healthPayload.limits.allowed_models.includes(modelName))
        .slice(0, healthPayload.limits.max_models);
      return preserved.length > 0
        ? preserved
        : healthPayload.limits.allowed_models.slice(0, healthPayload.limits.max_models);
    });
  }

  useEffect(() => {
    if (!interactiveMode) {
      return;
    }

    let cancelled = false;

    setErrorMessage(null);
    setServerState("checking");
    setWarmupStartTime(Date.now());
    setWarmupElapsed(0);

    async function loadModelCatalog(healthPayload: HealthPayload): Promise<void> {
      try {
        const modelPayload = await fetchModels();
        if (cancelled || !Array.isArray(modelPayload.models) || modelPayload.models.length === 0) {
          return;
        }

        const allowedModelNames = new Set(
          modelPayload.models.map((model) => model.name).filter((modelName) =>
            healthPayload.limits.allowed_models.includes(modelName)
          )
        );

        setModelOptions(modelPayload.models);
        setSelectedModels((current) => {
          const preserved = current
            .filter((modelName) => allowedModelNames.has(modelName))
            .slice(0, healthPayload.limits.max_models);
          if (preserved.length > 0) {
            return preserved;
          }

          const preferredSelection = modelPayload.models
            .filter(
              (model) => allowedModelNames.has(model.name) && modelTier(model) === "fast"
            )
            .map((model) => model.name)
            .slice(0, healthPayload.limits.max_models);

          return preferredSelection.length > 0
            ? preferredSelection
            : modelPayload.models
                .map((model) => model.name)
                .filter((modelName) => allowedModelNames.has(modelName))
                .slice(0, healthPayload.limits.max_models);
        });
      } catch (error) {
        console.error("Failed to load model catalog during warm-up:", error);
      }
    }

    async function warmServer(): Promise<void> {
      for (let attempt = 0; attempt < API_WARMUP_ATTEMPTS; attempt += 1) {
        try {
          const healthPayload = await getHealth();
          if (cancelled) {
            return;
          }

          applyHealthPayload(healthPayload);
          setServerState("ready");
          void loadModelCatalog(healthPayload);
          return;
        } catch {
          if (cancelled) {
            return;
          }
          setServerState("warming");
        }

        if (attempt < API_WARMUP_ATTEMPTS - 1) {
          await wait(API_WARMUP_INTERVAL_MS);
        }
      }

      if (!cancelled) {
        setServerState("unavailable");
        setStatusMessage(null);
      }
    }

    void warmServer();
    return () => {
      cancelled = true;
    };
  }, [interactiveMode]);

  useEffect(() => {
    if (serverState !== "checking" && serverState !== "warming") {
      return;
    }

    const intervalId = window.setInterval(() => {
      if (warmupStartTime) {
        setWarmupElapsed(Math.floor((Date.now() - warmupStartTime) / 1000));
      }
    }, 1000);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [serverState, warmupStartTime]);

  useEffect(() => {
    if (!isJobActive(job)) {
      return;
    }

    let cancelled = false;

    const pollCurrentJob = async (): Promise<void> => {
      try {
        const jobStatus = await pollJob(job.jobId);
        if (cancelled) {
          return;
        }

        if (jobStatus.status === "complete" && jobStatus.run_id) {
          setJob(null);
          setCompletedRunId(jobStatus.run_id);
          setPhase("complete");
          setStatusMessage("Batch accuracy test complete.");
          return;
        }

        if (jobStatus.status === "failed") {
          setJob(null);
          setPhase("upload");
          setErrorMessage(jobStatus.error || "Batch accuracy test failed.");
          return;
        }

        setJob((current) =>
          current
            ? {
                ...current,
                status: jobStatus.status === "queued" ? "queued" : "running",
                stage: jobStatus.stage ?? current.stage,
                progress: jobStatus.progress ?? current.progress,
              }
            : current
        );
        setStatusMessage(jobStatus.progress || "Running batch accuracy test...");
      } catch (error) {
        if (!cancelled) {
          setJob(null);
          setPhase("upload");
          setErrorMessage(
            error instanceof Error ? error.message : "Failed to poll job status."
          );
        }
      }
    };

    void pollCurrentJob();
    const intervalId = window.setInterval(() => {
      void pollCurrentJob();
    }, 5000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [job]);

  function resetSubmissionState(): void {
    setPhase("upload");
    setJob(null);
    setCompletedRunId(null);
    setStatusMessage(null);
  }

  async function loadCsvFile(file: File): Promise<void> {
    setErrorMessage(null);
    resetSubmissionState();

    try {
      const parsed = parseCsv(await file.text());
      if ((parsed.videos || []).length === 0) {
        setCsvFile(null);
        setCsvData(null);
        setUploadedVideos([]);
        setErrorMessage("No valid CSV rows were found. Expected columns: video_filename,label.");
        return;
      }

      setCsvFile(file);
      setCsvData(parsed);
      setUploadedVideos([]);
      setStatusMessage(`Parsed ${parsed.videos.length} videos and ${parsed.totalLabels} labels.`);
    } catch (error) {
      setCsvFile(null);
      setCsvData(null);
      setUploadedVideos([]);
      setStatusMessage(null);
      setErrorMessage(
        error instanceof Error ? error.message : "Could not read the CSV file."
      );
    }
  }

  function updateUploadedFileList(candidateFiles: FileList | File[]): void {
    const files = Array.from(candidateFiles).filter(isAcceptedFile);
    if (files.length === 0) {
      setErrorMessage(`Supported file types: ${ACCEPTED_TYPES.join(" ")}`);
      return;
    }

    const oversized = files.find(
      (file) => file.size > limits.max_file_size_mb * 1024 * 1024
    );
    if (oversized) {
      setErrorMessage(`"${oversized.name}" is too large. Max ${limits.max_file_size_mb}MB.`);
      return;
    }

    setErrorMessage(null);
    resetSubmissionState();
    setUploadedVideos((current) => mergeUploadedVideos(current, files));
  }

  function handleCsvChange(event: ChangeEvent<HTMLInputElement>): void {
    const [file] = Array.from(event.target.files ?? []);
    if (file) {
      void loadCsvFile(file);
      event.target.value = "";
    }
  }

  function handleVideosSelect(event: ChangeEvent<HTMLInputElement>): void {
    if (event.target.files) {
      updateUploadedFileList(event.target.files);
      event.target.value = "";
    }
  }

  function handleVideosDrop(event: DragEvent<HTMLDivElement>): void {
    event.preventDefault();
    setIsDragActive(false);
    if (!interactiveMode || isSubmitting) {
      return;
    }
    updateUploadedFileList(event.dataTransfer.files);
  }

  function toggleModel(modelName: string): void {
    setErrorMessage(null);
    setSelectedModels((current) => {
      if (current.includes(modelName)) {
        return current.filter((value) => value !== modelName);
      }
      if (current.length >= limits.max_models) {
        setErrorMessage(`Select up to ${limits.max_models} models per run.`);
        return current;
      }
      return [...current, modelName];
    });
  }

  function renderModelOption(model: ApiModel) {
    const checked = selectedModels.includes(model.name);
    return (
      <label key={model.name} className={`upload-tier-row ${checked ? "active" : ""}`}>
        <span>
          <strong>{model.display_name}</strong>
          <span className="upload-inline-note">{model.description}</span>
        </span>
        <input
          type="checkbox"
          checked={checked}
          onChange={() => toggleModel(model.name)}
          disabled={isSubmitting || isJobActive(job) || (!checked && selectedModels.length >= limits.max_models)}
        />
      </label>
    );
  }

  function getVideoStatus(index: number): "pending" | "running" | "complete" {
    if (phase === "complete") {
      return "complete";
    }
    if (isJobActive(job)) {
      return "running";
    }
    const video = csvVideos[index];
    if (video && findUploadedVideo(video.filename, uploadedVideos)) {
      return "running";
    }
    return "pending";
  }

  function getVideoStatusText(index: number): string {
    const video = csvVideos[index];
    if (!video) {
      return "Waiting";
    }
    if (phase === "complete") {
      return "Complete";
    }
    if (isJobActive(job)) {
      return job.progress || (job.status === "queued" ? "Queued" : "Running benchmark...");
    }
    return findUploadedVideo(video.filename, uploadedVideos) ? "Uploaded" : "Waiting for file";
  }

  async function handleSubmitBatch(): Promise<void> {
    if (
      !csvFile ||
      !csvData ||
      !allVideosUploaded ||
      orderedUploadedVideos.length !== csvData.videos.length ||
      selectedModels.length === 0 ||
      isSubmitting ||
      isJobActive(job)
    ) {
      return;
    }

    setIsSubmitting(true);
    setErrorMessage(null);
    setStatusMessage("Preparing server...");
    setCompletedRunId(null);

    try {
      let serverReady = false;
      setServerState("checking");
      setWarmupStartTime(Date.now());
      setWarmupElapsed(0);

      for (let attempt = 0; attempt < BATCH_SUBMIT_WARMUP_ATTEMPTS; attempt += 1) {
        try {
          const healthPayload = await getHealth();
          applyHealthPayload(healthPayload);
          setServerState("ready");
          serverReady = true;
          break;
        } catch {
          setServerState("warming");
        }

        setStatusMessage(`Waking up server... (${(attempt + 1) * 5}s)`);
        if (attempt < BATCH_SUBMIT_WARMUP_ATTEMPTS - 1) {
          await wait(API_WARMUP_INTERVAL_MS);
        }
      }

      if (!serverReady) {
        setServerState("unavailable");
        setStatusMessage(null);
        setErrorMessage("Could not reach the benchmark server. Please try again in a minute.");
        return;
      }

      setStatusMessage("Submitting batch accuracy test...");
      const response: BatchBenchmarkJobResponse = await uploadBatchBenchmark(
        csvFile,
        orderedUploadedVideos,
        selectedModels,
        benchmarkName.trim() || undefined,
        {
          onStatus: (message) => {
            setStatusMessage(message);
          },
        }
      );

      setPhase("running");
      setJob({
        jobId: response.job_id,
        status: response.status === "queued" ? "queued" : "running",
        estimatedTimeS: response.estimated_time_s,
        videoCount: response.video_count || csvData.videos.length,
        stage: response.status === "queued" ? "queued" : "preparing",
        progress:
          response.status === "queued"
            ? "Upload received. Waiting for a worker..."
            : "Preparing batch accuracy test...",
      });
      setStatusMessage(
        `Batch accuracy test submitted for ${response.video_count || csvData.videos.length} videos.`
      );
    } catch (error) {
      setStatusMessage(null);
      setErrorMessage(
        formatBackendError(error, "Batch accuracy test failed. Please try again.")
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  if (!interactiveMode) {
    const disabledCard = (
      <section className="visual-card upload-inline-card dashboard-section-card is-disabled">
        <div className="accuracy-header">
          <h1>Batch Accuracy Test</h1>
          <p>Set `NEXT_PUBLIC_API_URL` to enable interactive CSV and multi-video uploads.</p>
        </div>
      </section>
    );

    if (embedded) {
      return disabledCard;
    }

    return (
      <main className="analysis-shell">
        <TopNav active="accuracy" />
        {disabledCard}
      </main>
    );
  }

  const pageContent = (
    <section className="visual-card upload-inline-card dashboard-section-card">
        <div className="batch-upload-section">
          <div className="section-heading">
            <p className="section-eyebrow">Batch Accuracy Test</p>
            <h1>Test model accuracy against your labeled videos</h1>
            <p>
              Upload a CSV of action labels and the corresponding video files. We&apos;ll segment
              each video, run your chosen models, and score accuracy against your labels.
            </p>
          </div>

          <ServerStatusBanner serverState={serverState} warmupElapsed={warmupElapsed} />

          {phase === "upload" ? (
            <>
              <div className="csv-upload">
                <h3>1. Upload your labels CSV</h3>
                <p className="upload-hint">
                  CSV format: <code>video_filename,label</code> - one row per segment. Labels are
                  matched to segments in order.
                </p>
                <div className="csv-format-example">
                  <pre>{`video_filename,label
cooking_demo.mp4,chopping vegetables
cooking_demo.mp4,stirring soup
assembly_line.mp4,operating press`}</pre>
                </div>
                <input
                  type="file"
                  accept=".csv,.tsv,.txt"
                  onChange={handleCsvChange}
                />
                {csvFile ? <p className="upload-inline-note">Selected CSV: {csvFile.name}</p> : null}
                {csvData ? (
                  <div className="csv-preview">
                    <p>
                      {csvData.videos.length} videos, {csvData.totalLabels} labels detected
                    </p>
                    <ul className="csv-video-list">
                      {csvVideos.map((video) => (
                        <li key={video.filename}>
                          <strong>{video.filename}</strong> - {video.labels.length} labels
                        </li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>

              {csvData ? (
                <div className="video-upload">
                  <h3>2. Upload your video files</h3>
                  <p className="upload-hint">
                    Upload the {csvData.videos.length} video files referenced in your CSV. Drag and
                    drop multiple files at once.
                  </p>
                  <div
                    className={`multi-drop ${isDragActive ? "drag-over" : ""}`}
                    onClick={() => {
                      if (!isSubmitting) {
                        videoInputRef.current?.click();
                      }
                    }}
                    onDrop={handleVideosDrop}
                    onDragEnter={(event) => {
                      event.preventDefault();
                      if (!isSubmitting) {
                        setIsDragActive(true);
                      }
                    }}
                    onDragOver={(event) => {
                      event.preventDefault();
                      if (!isSubmitting) {
                        setIsDragActive(true);
                      }
                    }}
                    onDragLeave={(event) => {
                      event.preventDefault();
                      if (event.currentTarget === event.target) {
                        setIsDragActive(false);
                      }
                    }}
                    role="button"
                    tabIndex={isSubmitting ? -1 : 0}
                    onKeyDown={(event) => {
                      if (isSubmitting) {
                        return;
                      }
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        videoInputRef.current?.click();
                      }
                    }}
                  >
                    <input
                      ref={videoInputRef}
                      type="file"
                      accept="video/*"
                      multiple
                      hidden
                      onChange={handleVideosSelect}
                      disabled={isSubmitting}
                    />
                    <p>Drop {csvData.videos.length} video files here</p>
                    <p className="drop-sub">or click to browse</p>
                    <p className="upload-inline-note">
                      Max {limits.max_file_size_mb}MB per file. Supported: {ACCEPTED_TYPES.join(" ")}
                    </p>
                  </div>

                  <div className="video-checklist">
                    {csvVideos.map((video) => {
                      const uploaded = findUploadedVideo(video.filename, uploadedVideos);
                      return (
                        <div
                          key={video.filename}
                          className={`video-check ${uploaded ? "found" : "missing"}`}
                        >
                          <span className="check-icon">{uploaded ? "OK" : "o"}</span>
                          <span className="check-name">{video.filename}</span>
                          <span className="check-labels">{video.labels.length} labels</span>
                          {uploaded ? (
                            <span className="check-size">{formatFileSize(uploaded)}</span>
                          ) : null}
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : null}

              {allVideosUploaded && csvData ? (
                <div className="batch-model-shell">
                  <h3>3. Choose models</h3>
                  <label className="field">
                    <span>Batch test name (optional)</span>
                    <input
                      type="text"
                      value={benchmarkName}
                      onChange={(event) => setBenchmarkName(event.target.value)}
                      placeholder="week-1-labeled-batch"
                      disabled={isSubmitting || isJobActive(job)}
                    />
                  </label>

                  <div className="upload-model-picker">
                    <section className="upload-model-tier">
                      <div className="upload-tier-heading">
                        <h3 className="tier-label">
                          Fast models <span className="tier-badge fast">~$0.01/segment</span>
                        </h3>
                      </div>
                      <p className="tier-desc">
                        Quick results, lowest cost. Good for broad batch validation.
                      </p>
                      <div className="upload-inline-tier-list">{fastModels.map(renderModelOption)}</div>
                    </section>

                    <section className="upload-model-tier">
                      <div className="upload-tier-heading">
                        <h3 className="tier-label">
                          Frontier models <span className="tier-badge frontier">~$0.05-0.10/segment</span>
                        </h3>
                      </div>
                      <p className="tier-desc">
                        Higher accuracy, slower, and meaningfully more expensive.
                      </p>
                      <div className="upload-inline-tier-list">
                        {frontierModels.map(renderModelOption)}
                      </div>
                    </section>
                  </div>

                  <div className="batch-summary">
                    <p>
                      <strong>{csvData.videos.length} videos</strong> x <strong>{selectedModels.length} models</strong> = ~
                      {estimatedCalls} API calls
                    </p>
                    <p className="cost-estimate">Estimated cost: ~${estimatedCost.toFixed(2)}</p>
                  </div>

                  <button
                    type="button"
                    className="primary-btn upload-inline-button"
                    onClick={() => void handleSubmitBatch()}
                    disabled={
                      serverState !== "ready" ||
                      selectedModels.length === 0 ||
                      isSubmitting ||
                      isJobActive(job)
                    }
                  >
                    {isSubmitting
                      ? statusMessage ?? "Working..."
                      : `Run Batch Accuracy Test (${csvData.videos.length} videos)`}
                  </button>
                </div>
              ) : null}
            </>
          ) : null}

          {phase === "running" && csvData ? (
            <div className="batch-progress" aria-live="polite">
              <h2>Processing {csvData.videos.length} videos...</h2>
              <div className="progress-list">
                {csvVideos.map((video, index) => (
                  <div key={video.filename} className={`progress-item ${getVideoStatus(index)}`}>
                    <span className="progress-icon">
                      {getVideoStatus(index) === "complete"
                        ? "OK"
                        : getVideoStatus(index) === "running"
                          ? "..."
                          : "o"}
                    </span>
                    <span className="progress-name">{video.filename}</span>
                    <span className="progress-status">{getVideoStatusText(index)}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : null}

          {phase === "complete" && completedRunId && csvData ? (
            <div className="batch-complete">
              <h2>Batch accuracy test complete!</h2>
              <p>
                {csvData.videos.length} videos processed across {selectedModels.length} models.
              </p>
              <Link href={`/report/${completedRunId}`} className="view-report-btn">
                {"View Accuracy Report ->"}
              </Link>
            </div>
          ) : null}

          {statusMessage ? <p className="upload-inline-status">{statusMessage}</p> : null}
          {errorMessage ? <p className="upload-inline-error">{errorMessage}</p> : null}
        </div>
      </section>
  );

  if (embedded) {
    return pageContent;
  }

  return (
    <main className="analysis-shell">
      <TopNav active="accuracy" />
      {pageContent}
    </main>
  );
}
