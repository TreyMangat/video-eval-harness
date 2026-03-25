"use client";

import Link from "next/link";
import { ChangeEvent, DragEvent, useEffect, useMemo, useRef, useState } from "react";

import {
  fetchModels,
  getHealth,
  pollJob,
  previewSegments,
  uploadAndBenchmark,
  type ApiModel,
  type BenchmarkJobResponse,
  type HealthPayload,
  type SegmentPreviewResponse,
} from "../lib/api-client";
import { DEFAULT_MODEL_CATALOG } from "../lib/model-catalog";
import { BatchAccuracyTestPage } from "./batch-accuracy-test-page";
import { TopNav } from "./navigation";

const ACCEPTED_TYPES = [".mp4", ".avi", ".mov", ".mkv", ".webm"];
const DEFAULT_ALLOWED_MODELS = DEFAULT_MODEL_CATALOG.filter((model) => model.tier === "fast").map(
  (model) => model.name
);
const DEFAULT_SEGMENT_ESTIMATE = 3;
const API_WARMUP_ATTEMPTS = 12;
const API_WARMUP_INTERVAL_MS = 5_000;
const ACCURACY_SUBMIT_WARMUP_ATTEMPTS = 6;

type Phase = "upload" | "label" | "running" | "complete";
type TestMode = "single" | "batch";
type ServerState = "checking" | "warming" | "ready" | "unavailable";
type JobState = {
  jobId: string;
  status: "queued" | "running";
  estimatedTimeS: number;
  modelCount: number;
  stage: string | null;
  progress: string | null;
};

const DEFAULT_LIMITS: HealthPayload["limits"] = {
  max_clip_s: 60,
  max_file_size_mb: 100,
  max_models: DEFAULT_MODEL_CATALOG.length,
  allowed_models: DEFAULT_MODEL_CATALOG.map((model) => model.name),
};

type AccuracyTestPageProps = {
  initialTestMode?: TestMode;
};

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

function formatTime(seconds: number): string {
  const safeSeconds = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(safeSeconds / 60);
  const remainder = safeSeconds % 60;
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
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

export function AccuracyTestPage({
  initialTestMode = "single",
}: AccuracyTestPageProps = {}) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const interactiveMode = (process.env.NEXT_PUBLIC_API_URL?.trim() ?? "").length > 0;
  const [testMode, setTestMode] = useState<TestMode>(initialTestMode);
  const [phase, setPhase] = useState<Phase>("upload");
  const [serverState, setServerState] = useState<ServerState>("checking");
  const [warmupStartTime, setWarmupStartTime] = useState<number | null>(null);
  const [warmupElapsed, setWarmupElapsed] = useState(0);
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [modelOptions, setModelOptions] = useState<ApiModel[]>(fallbackModelOptions);
  const [selectedModels, setSelectedModels] = useState<string[]>(DEFAULT_ALLOWED_MODELS);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<SegmentPreviewResponse | null>(null);
  const [labels, setLabels] = useState<string[]>([]);
  const [benchmarkName, setBenchmarkName] = useState("");
  const [isDragActive, setIsDragActive] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [job, setJob] = useState<JobState | null>(null);
  const [completedRunId, setCompletedRunId] = useState<string | null>(null);

  const limits = health?.limits ?? DEFAULT_LIMITS;
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
  const selectedFileLabel = useMemo(() => {
    if (!selectedFile) {
      return null;
    }
    return `${selectedFile.name} (${formatFileSize(selectedFile)})`;
  }, [selectedFile]);
  const estimatedSegmentCount = preview?.segment_count ?? DEFAULT_SEGMENT_ESTIMATE;
  const estimatedCost = useMemo(
    () =>
      selectedModels.reduce((sum, modelName) => {
        return (
          sum +
          estimatedSegmentCount * modelCostPerSegment(modelOptionsByName.get(modelName), modelName)
        );
      }, 0),
    [estimatedSegmentCount, modelOptionsByName, selectedModels]
  );
  const hasFrontierSelected = useMemo(
    () =>
      selectedModels.some((modelName) =>
        modelTier(modelOptionsByName.get(modelName), modelName) === "frontier"
      ),
    [modelOptionsByName, selectedModels]
  );
  const filledLabelCount = useMemo(
    () => labels.filter((label) => label.trim().length > 0).length,
    [labels]
  );

  useEffect(() => {
    setTestMode(initialTestMode);
  }, [initialTestMode]);

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
          setStatusMessage("Accuracy test complete.");
          return;
        }

        if (jobStatus.status === "failed") {
          setJob(null);
          setPhase("label");
          setErrorMessage(jobStatus.error || "Accuracy test failed.");
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
        setStatusMessage(jobStatus.progress || "Running accuracy test...");
      } catch (error) {
        if (!cancelled) {
          setJob(null);
          setPhase("label");
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

  function resetPreviewState(): void {
    setPreview(null);
    setLabels([]);
    setPhase("upload");
    setJob(null);
    setCompletedRunId(null);
    setStatusMessage(null);
  }

  function updateSelectedFile(candidateFiles: FileList | File[]): void {
    const files = Array.from(candidateFiles);
    const [file] = files.filter(isAcceptedFile);

    if (!file) {
      setErrorMessage(`Supported file types: ${ACCEPTED_TYPES.join(" ")}`);
      return;
    }

    if (file.size > limits.max_file_size_mb * 1024 * 1024) {
      setErrorMessage(`File too large. Max ${limits.max_file_size_mb}MB.`);
      return;
    }

    setSelectedFile(file);
    setErrorMessage(null);
    resetPreviewState();
  }

  function handleInputChange(event: ChangeEvent<HTMLInputElement>): void {
    if (event.target.files) {
      updateSelectedFile(event.target.files);
      event.target.value = "";
    }
  }

  function handleDrop(event: DragEvent<HTMLDivElement>): void {
    event.preventDefault();
    setIsDragActive(false);
    if (!interactiveMode) {
      return;
    }
    updateSelectedFile(event.dataTransfer.files);
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

  async function handlePreviewSubmit(): Promise<void> {
    if (serverState !== "ready" || !selectedFile || isSubmitting || isJobActive(job)) {
      return;
    }

    setIsSubmitting(true);
    setErrorMessage(null);
    setStatusMessage("Uploading video and generating segment previews...");
    setCompletedRunId(null);

    try {
      const previewPayload = await previewSegments(selectedFile, {
        onStatus: (message) => {
          setStatusMessage(message);
        },
      });
      setPreview(previewPayload);
      setLabels(Array(previewPayload.segment_count).fill(""));
      setPhase("label");
      setStatusMessage("Preview ready. Label the segments below to run the accuracy test.");
    } catch (error) {
      setStatusMessage(null);
      setErrorMessage(
        formatBackendError(error, "Failed to generate segment previews. Please try again.")
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  async function handleRunAccuracyTest(): Promise<void> {
    if (
      !preview ||
      selectedModels.length === 0 ||
      filledLabelCount === 0 ||
      isSubmitting ||
      isJobActive(job)
    ) {
      return;
    }

    setIsSubmitting(true);
    setErrorMessage(null);
    setStatusMessage("Preparing server...");
    setCompletedRunId(null);

    const groundTruth = labels
      .map((label, index) => ({
        segment_index: index,
        label: label.trim(),
      }))
      .filter((entry) => entry.label.length > 0);

    try {
      let serverReady = false;
      setServerState("checking");
      setWarmupStartTime(Date.now());
      setWarmupElapsed(0);

      for (let attempt = 0; attempt < ACCURACY_SUBMIT_WARMUP_ATTEMPTS; attempt += 1) {
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
        if (attempt < ACCURACY_SUBMIT_WARMUP_ATTEMPTS - 1) {
          await wait(API_WARMUP_INTERVAL_MS);
        }
      }

      if (!serverReady) {
        setServerState("unavailable");
        setStatusMessage(null);
        setErrorMessage("Could not reach the benchmark server. Please try again in a minute.");
        return;
      }

      setStatusMessage("Submitting accuracy test...");
      const response: BenchmarkJobResponse = await uploadAndBenchmark(
        null,
        selectedModels,
        benchmarkName.trim() || undefined,
        {
          runType: "accuracy_test",
          previewId: preview.preview_id,
          groundTruth,
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
        modelCount: selectedModels.length,
        stage: response.status === "queued" ? "queued" : "preparing",
        progress:
          response.status === "queued"
            ? "Upload received. Waiting for a worker..."
            : "Preparing accuracy test...",
      });
      setStatusMessage("Accuracy test submitted. Starting benchmark...");
    } catch (error) {
      setStatusMessage(null);
      setErrorMessage(formatBackendError(error, "Accuracy test failed. Please try again."));
    } finally {
      setIsSubmitting(false);
    }
  }

  const headerCard = (
    <section className="visual-card dashboard-section-card">
      <div className="accuracy-header">
        <span className="section-label">ACCURACY TEST</span>
        <h1>Test model accuracy against your labels</h1>
        <p>
          Upload a video and label what you see, or upload a CSV of pre-labeled videos. We&apos;ll
          run your chosen models and score how accurately they match your descriptions.
        </p>
      </div>
      <div className="test-mode-tabs" role="tablist" aria-label="Accuracy test mode">
        <button
          type="button"
          role="tab"
          aria-selected={testMode === "single"}
          className={`mode-tab ${testMode === "single" ? "active" : ""}`}
          onClick={() => setTestMode("single")}
        >
          Single video
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={testMode === "batch"}
          className={`mode-tab ${testMode === "batch" ? "active" : ""}`}
          onClick={() => setTestMode("batch")}
        >
          Batch (CSV + videos)
        </button>
      </div>
    </section>
  );

  const singleContent = !interactiveMode ? (
    <section className="visual-card upload-inline-card dashboard-section-card is-disabled">
      <div className="accuracy-header">
        <h1>Single-video accuracy testing is unavailable</h1>
        <p>Set `NEXT_PUBLIC_API_URL` to enable interactive uploads and preview-based labeling.</p>
      </div>
    </section>
  ) : (
    <section className="visual-card dashboard-section-card">
      <ServerStatusBanner serverState={serverState} warmupElapsed={warmupElapsed} />

        {phase === "upload" ? (
          <div className="upload-inline-grid">
            <div
              className={`upload-inline-dropzone ${isDragActive ? "drag-active" : ""}`}
              onClick={() => {
                if (!isSubmitting) {
                  fileInputRef.current?.click();
                }
              }}
              onDrop={handleDrop}
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
                  fileInputRef.current?.click();
                }
              }}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept={ACCEPTED_TYPES.join(",")}
                onChange={handleInputChange}
                hidden
                disabled={isSubmitting}
              />

              <div className="upload-inline-dropzone-main">
                <svg
                  width="40"
                  height="40"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  aria-hidden="true"
                  className="upload-inline-icon"
                >
                  <path
                    d="M12 16V8m0 0l-3 3m3-3l3 3"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <path
                    d="M20 16.7A4.5 4.5 0 0017.5 8h-1.1A7 7 0 104 14.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                <p className="upload-inline-title">Drop a short video clip here</p>
                <p className="upload-inline-copy">or click to browse</p>
                <p className="upload-inline-note">
                  Max {limits.max_clip_s} seconds | Max {limits.max_file_size_mb}MB | preview only,
                  no model cost yet
                </p>
                <div className="upload-inline-type-list" aria-label="Supported file types">
                  {ACCEPTED_TYPES.map((type) => (
                    <span key={type} className="upload-type-pill">
                      {type}
                    </span>
                  ))}
                </div>
              </div>

              {selectedFileLabel ? (
                <div className="upload-inline-queue" aria-label="Selected file">
                  <span className="upload-inline-queue-label">Selected clip</span>
                  <div className="upload-inline-queue-list">
                    <div className="upload-file-pill">
                      <span>{selectedFileLabel}</span>
                      <button
                        type="button"
                        onClick={(event) => {
                          event.stopPropagation();
                          setSelectedFile(null);
                          resetPreviewState();
                        }}
                        aria-label={`Remove ${selectedFile?.name ?? "selected clip"}`}
                        disabled={isSubmitting}
                      >
                        x
                      </button>
                    </div>
                  </div>
                </div>
              ) : null}
            </div>

            <div className="upload-inline-controls">
              <p className="label-instructions">
                Upload a video, let the backend split it into 10-second segments, then label each
                keyframe before you run the models. This gives you real accuracy scores instead of
                agreement-only comparisons.
              </p>

              <button
                type="button"
                className="primary-btn upload-inline-button"
                onClick={() => void handlePreviewSubmit()}
                disabled={serverState !== "ready" || isSubmitting || !selectedFile}
              >
                {serverState !== "ready" ? "Waiting for server..." : "Upload & Preview Segments"}
              </button>

              {statusMessage ? <p className="upload-inline-status">{statusMessage}</p> : null}
              {errorMessage ? <p className="upload-inline-error">{errorMessage}</p> : null}
            </div>
          </div>
        ) : null}

        {phase === "label" && preview ? (
          <div className="accuracy-label-shell">
            <div className="accuracy-phase-actions">
              <div>
                <p className="label-instructions">
                  Labels you provide will be used as ground truth to score model accuracy. The more
                  segments you label, the more reliable the accuracy score.
                </p>
                <p className="label-count">
                  {filledLabelCount} of {preview.segment_count} segments labeled
                </p>
              </div>
              <button
                type="button"
                className="ghost-btn"
                onClick={() => {
                  resetPreviewState();
                }}
              >
                Choose Another Video
              </button>
            </div>

            <div className="segment-grid">
              {preview.segments.map((segment, index) => (
                <div key={segment.segment_id} className="segment-card">
                  <img
                    src={`data:image/jpeg;base64,${segment.keyframe_base64}`}
                    alt={`Segment ${index + 1}`}
                    className="segment-keyframe"
                  />
                  <div className="segment-info">
                    <span className="segment-time">
                      Segment {index + 1} - {formatTime(segment.start_s)} to {formatTime(segment.end_s)}
                    </span>
                    <input
                      type="text"
                      className="segment-label-input"
                      placeholder="What action is happening? e.g., cooking food"
                      value={labels[index] || ""}
                      onChange={(event) => {
                        const nextLabels = [...labels];
                        nextLabels[index] = event.target.value;
                        setLabels(nextLabels);
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>

            <div className="accuracy-controls-grid">
              <label className="field">
                <span>Accuracy test name (optional)</span>
                <input
                  type="text"
                  value={benchmarkName}
                  onChange={(event) => setBenchmarkName(event.target.value)}
                  placeholder="kitchen-accuracy-check"
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
                  <p className="tier-desc">Quick results, lowest cost. Good for rapid label checks.</p>
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

              <div className="cost-estimate">
                Estimated cost: ~${estimatedCost.toFixed(2)} for {selectedModels.length} models across{" "}
                {preview.segment_count} segments
              </div>

              {hasFrontierSelected ? (
                <p className="cost-warning">
                  Warning: Frontier models are 5-10x more expensive. Estimated cost: ~$
                  {estimatedCost.toFixed(2)}
                </p>
              ) : null}

              <button
                type="button"
                className="primary-btn upload-inline-button"
                onClick={() => void handleRunAccuracyTest()}
                disabled={
                  isSubmitting ||
                  isJobActive(job) ||
                  selectedModels.length === 0 ||
                  filledLabelCount === 0
                }
              >
                {isSubmitting ? statusMessage ?? "Working..." : "Run Accuracy Test"}
              </button>

              {statusMessage ? <p className="upload-inline-status">{statusMessage}</p> : null}
              {errorMessage ? <p className="upload-inline-error">{errorMessage}</p> : null}
            </div>
          </div>
        ) : null}

        {phase === "running" && isJobActive(job) ? (
          <div className="accuracy-running-shell">
            <div className="benchmark-progress" aria-live="polite">
              <span className="progress-spinner upload-spinner" aria-hidden="true" />
              <div className="progress-info">
                <strong>
                  {job.status === "queued"
                    ? "Accuracy test queued..."
                    : `Running accuracy test... ~${job.estimatedTimeS} seconds`}
                </strong>
                <p className="progress-stage">{job.progress || "Starting..."}</p>
                <p className="progress-hint">
                  {job.status === "queued"
                    ? "Waiting for a Modal worker to start processing your clip."
                    : "This usually takes 60-120 seconds."}
                </p>
              </div>
            </div>

            {statusMessage ? <p className="upload-inline-status">{statusMessage}</p> : null}
            {errorMessage ? <p className="upload-inline-error">{errorMessage}</p> : null}
          </div>
        ) : null}

        {phase === "complete" && completedRunId ? (
          <div className="accuracy-complete">
            <h3>Accuracy test complete!</h3>
            <p>Models were scored against your labels.</p>
            <Link href={`/report/${completedRunId}`} className="view-report-btn">
              {"View Accuracy Report ->"}
            </Link>
          </div>
        ) : null}
      </section>
  );

  return (
    <main className="analysis-shell">
      <TopNav active="accuracy" />
      {headerCard}
      {testMode === "single" ? singleContent : <BatchAccuracyTestPage embedded />}
    </main>
  );
}
