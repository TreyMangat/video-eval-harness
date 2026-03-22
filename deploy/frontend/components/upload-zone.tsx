"use client";

import Link from "next/link";
import { ChangeEvent, DragEvent, useEffect, useMemo, useRef, useState } from "react";

import {
  fetchModels,
  fetchRun,
  pollJob,
  uploadAndBenchmark,
  type ApiModel,
  type BenchmarkJobResponse,
  type HealthPayload,
} from "../lib/api-client";
import { DEFAULT_MODEL_CATALOG } from "../lib/model-catalog";

const ACCEPTED_TYPES = [".mp4", ".avi", ".mov", ".mkv", ".webm"];
const DEFAULT_ALLOWED_MODELS = DEFAULT_MODEL_CATALOG.filter((model) => model.tier === "fast").map(
  (model) => model.name
);
const DEFAULT_SEGMENT_ESTIMATE = 3;
const API_WARMUP_ATTEMPTS = 12;
const API_WARMUP_INTERVAL_MS = 5_000;
const API_WARMUP_TIMEOUT_MS = 8_000;
const DEFAULT_LIMITS: HealthPayload["limits"] = {
  max_clip_s: 60,
  max_file_size_mb: 100,
  max_models: DEFAULT_MODEL_CATALOG.length,
  allowed_models: DEFAULT_MODEL_CATALOG.map((model) => model.name),
};

type ServerState = "checking" | "warming" | "ready" | "unavailable";
type JobState = {
  jobId: string;
  status: "queued" | "running";
  estimatedTimeS: number;
  modelCount: number;
  stage: string | null;
  progress: string | null;
};

type CompletedBenchmark = {
  runId: string;
  modelCount: number;
  segmentCount: number | null;
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

async function wait(ms: number): Promise<void> {
  await new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

async function checkServerHealth(apiUrl: string): Promise<HealthPayload> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    controller.abort();
  }, API_WARMUP_TIMEOUT_MS);

  try {
    const response = await fetch(
      new URL("api/health", apiUrl.endsWith("/") ? apiUrl : `${apiUrl}/`).toString(),
      {
        cache: "no-store",
        mode: "cors",
        signal: controller.signal,
      }
    );
    const payload = (await response.json().catch(() => null)) as HealthPayload | null;
    if (!response.ok || !payload) {
      throw new Error(`Server readiness check failed (${response.status}).`);
    }
    return payload;
  } finally {
    clearTimeout(timeoutId);
  }
}

export function UploadZone() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const apiUrl = process.env.NEXT_PUBLIC_API_URL?.trim() ?? "";
  const interactiveMode = apiUrl.length > 0;
  const [serverState, setServerState] = useState<ServerState>("checking");
  const [warmupStartTime, setWarmupStartTime] = useState<number | null>(null);
  const [warmupElapsed, setWarmupElapsed] = useState(0);
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [modelOptions, setModelOptions] = useState<ApiModel[]>(fallbackModelOptions);
  const [selectedModels, setSelectedModels] = useState<string[]>(DEFAULT_ALLOWED_MODELS);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [benchmarkName, setBenchmarkName] = useState("");
  const [isDragActive, setIsDragActive] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [job, setJob] = useState<JobState | null>(null);
  const [completedBenchmark, setCompletedBenchmark] = useState<CompletedBenchmark | null>(null);

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
  const estimatedCost = useMemo(
    () =>
      selectedModels.reduce((sum, modelName) => {
        return (
          sum +
          DEFAULT_SEGMENT_ESTIMATE *
            modelCostPerSegment(modelOptionsByName.get(modelName), modelName)
        );
      }, 0),
    [modelOptionsByName, selectedModels]
  );
  const hasFrontierSelected = useMemo(
    () =>
      selectedModels.some((modelName) =>
        modelTier(modelOptionsByName.get(modelName), modelName) === "frontier"
      ),
    [modelOptionsByName, selectedModels]
  );

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
              (model) =>
                allowedModelNames.has(model.name) && modelTier(model) === "fast"
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
          const healthPayload = await checkServerHealth(apiUrl);
          if (cancelled) {
            return;
          }

          setHealth(healthPayload);
          setSelectedModels((current) => {
            const preserved = current
              .filter((modelName) => healthPayload.limits.allowed_models.includes(modelName))
              .slice(0, healthPayload.limits.max_models);
            return preserved.length > 0
              ? preserved
              : healthPayload.limits.allowed_models.slice(0, healthPayload.limits.max_models);
          });
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
  }, [apiUrl, interactiveMode]);

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
            let segmentCount: number | null = null;
            try {
              const run = await fetchRun(jobStatus.run_id);
              if (!cancelled) {
                segmentCount = run.segments.length;
              }
            } catch (error) {
              console.error("Failed to load completed run payload:", error);
            }

            setJob(null);
            setCompletedBenchmark({
              runId: jobStatus.run_id,
              modelCount: job.modelCount,
              segmentCount,
            });
            setStatusMessage("Benchmark complete.");
            return;
          }

          if (jobStatus.status === "failed") {
            setJob(null);
            setCompletedBenchmark(null);
            setErrorMessage(jobStatus.error || "Benchmark failed.");
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
          setStatusMessage(jobStatus.progress || "Running benchmark...");
        } catch (error) {
          if (!cancelled) {
            setJob(null);
            setErrorMessage(error instanceof Error ? error.message : "Failed to poll job status.");
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
    setStatusMessage(null);
    setCompletedBenchmark(null);
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
          disabled={
            isSubmitting || isJobActive(job) || (!checked && selectedModels.length >= limits.max_models)
          }
        />
      </label>
    );
  }

  async function startBenchmark(): Promise<void> {
    if (serverState !== "ready" || !selectedFile || selectedModels.length === 0 || isSubmitting) {
      return;
    }

    setIsSubmitting(true);
    setErrorMessage(null);
    setStatusMessage("Uploading video...");
    setCompletedBenchmark(null);

    try {
      const response: BenchmarkJobResponse = await uploadAndBenchmark(
        selectedFile,
        selectedModels,
        benchmarkName.trim() || undefined,
        {
          onStatus: (message) => {
            setStatusMessage(message);
          },
        }
      );
      setJob({
        jobId: response.job_id,
        status: response.status === "queued" ? "queued" : "running",
        estimatedTimeS: response.estimated_time_s,
        modelCount: selectedModels.length,
        stage: response.status === "queued" ? "queued" : "preparing",
        progress:
          response.status === "queued"
            ? "Upload received. Waiting for a worker..."
            : "Preparing benchmark...",
      });
      setStatusMessage("Upload complete. Starting benchmark...");
    } catch (error) {
      setStatusMessage(null);
      setErrorMessage(formatBackendError(error, "Upload failed. Please try again."));
    } finally {
      setIsSubmitting(false);
    }
  }

  if (!interactiveMode) {
    return (
      <section className="visual-card upload-inline-card dashboard-section-card is-disabled">
        <div className="upload-inline-card-body">
          <div className="upload-inline-grid">
            <div className="upload-inline-dropzone">
              <div className="upload-inline-dropzone-main">
                <p className="upload-inline-title">Viewer-only deployment</p>
                <p className="upload-inline-copy">Set `NEXT_PUBLIC_API_URL` to enable uploads.</p>
                <p className="upload-inline-note">
                  Static mode still loads committed run data and exported local artifacts.
                </p>
              </div>
            </div>
            <div className="upload-inline-controls">
              <p className="upload-inline-status">
                Interactive uploads are disabled for this deployment.
              </p>
            </div>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="visual-card upload-inline-card dashboard-section-card">
      <div className="upload-inline-card-body">
        {serverState === "checking" ? (
          <div className="server-status server-checking" aria-live="polite">
            <div className="status-spinner" aria-hidden="true" />
            <div className="status-text">
              <strong>Connecting to benchmark server...</strong>
              <p>This usually takes a few seconds.</p>
            </div>
          </div>
        ) : null}

        {serverState === "warming" ? (
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
        ) : null}

        {serverState === "ready" ? (
          <div className="server-status server-ready" aria-live="polite">
            <div className="status-dot" aria-hidden="true" />
            <div className="status-text">
              <strong>Server ready</strong>
            </div>
          </div>
        ) : null}

        {serverState === "unavailable" ? (
          <div className="server-status server-unavailable" aria-live="polite">
            <div className="status-text">
              <strong>Benchmark server is currently unavailable.</strong>
              <p>
                The server did not respond after 60 seconds. Try refreshing the page, or check
                back later.
              </p>
            </div>
          </div>
        ) : null}

        <div className="upload-inline-grid">
          <div
            className={`upload-inline-dropzone ${isDragActive ? "drag-active" : ""}`}
            onClick={() => {
              if (!isSubmitting && !isJobActive(job)) {
                fileInputRef.current?.click();
              }
            }}
            onDrop={handleDrop}
            onDragEnter={(event) => {
              event.preventDefault();
              if (!isSubmitting && !isJobActive(job)) {
                setIsDragActive(true);
              }
            }}
            onDragOver={(event) => {
              event.preventDefault();
              if (!isSubmitting && !isJobActive(job)) {
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
            tabIndex={isSubmitting || isJobActive(job) ? -1 : 0}
            onKeyDown={(event) => {
              if (isSubmitting || isJobActive(job)) {
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
              disabled={isSubmitting || isJobActive(job)}
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
                Max {limits.max_clip_s} seconds | Max {limits.max_file_size_mb}MB | up to{" "}
                {limits.max_models} models
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
                      }}
                      aria-label={`Remove ${selectedFile?.name ?? "selected clip"}`}
                      disabled={isSubmitting || isJobActive(job)}
                    >
                      x
                    </button>
                  </div>
                </div>
              </div>
            ) : null}
          </div>

          <div className="upload-inline-controls">
            <label className="field">
              <span>Benchmark name (optional)</span>
              <input
                type="text"
                value={benchmarkName}
                onChange={(event) => setBenchmarkName(event.target.value)}
                placeholder="factory-floor-smoke-test"
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
                <p className="tier-desc">Quick results, lowest cost. Good for testing.</p>
                <div className="upload-inline-tier-list">
                  {fastModels.map((model) => renderModelOption(model))}
                </div>
              </section>

              <section className="upload-model-tier">
                <div className="upload-tier-heading">
                  <h3 className="tier-label">
                    Frontier models{" "}
                    <span className="tier-badge frontier">~$0.05-0.10/segment</span>
                  </h3>
                </div>
                <p className="tier-desc">
                  Higher accuracy, slower, and meaningfully more expensive.
                </p>
                <div className="upload-inline-tier-list">
                  {frontierModels.map((model) => renderModelOption(model))}
                </div>
              </section>
            </div>

            <div className="cost-estimate">
              Estimated cost: ~${estimatedCost.toFixed(2)} for {selectedModels.length} models
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
              onClick={() => void startBenchmark()}
              disabled={
                serverState !== "ready" ||
                isSubmitting ||
                isJobActive(job) ||
                !selectedFile ||
                selectedModels.length === 0
              }
            >
              {serverState === "unavailable"
                ? "Server unavailable"
                : serverState !== "ready"
                  ? "Waiting for server..."
                  : "Start Benchmark"}
            </button>

            {isJobActive(job) ? (
              <div className="benchmark-progress" aria-live="polite">
                <span className="progress-spinner upload-spinner" aria-hidden="true" />
                <div className="progress-info">
                  <strong>
                    {job.status === "queued"
                      ? "Benchmark queued..."
                      : `Running benchmark... ~${job.estimatedTimeS} seconds`}
                  </strong>
                  <p className="progress-stage">{job.progress || "Starting..."}</p>
                  <p className="progress-hint">
                    {job.status === "queued"
                      ? "Waiting for a Modal worker to start processing your clip."
                      : "This usually takes 60-120 seconds."}
                  </p>
                </div>
              </div>
            ) : null}

            {completedBenchmark ? (
              <div className="benchmark-complete" aria-live="polite">
                <div className="complete-icon">✓</div>
                <div className="complete-copy">
                  <h3>Benchmark complete!</h3>
                  <p>
                    {completedBenchmark.modelCount} models analyzed{" "}
                    {completedBenchmark.segmentCount ?? DEFAULT_SEGMENT_ESTIMATE} segments.
                  </p>
                </div>
                <Link
                  href={`/report/${completedBenchmark.runId}`}
                  className="view-report-btn"
                >
                  View Report →
                </Link>
              </div>
            ) : null}

            {statusMessage && !completedBenchmark ? (
              <p className="upload-inline-status">{statusMessage}</p>
            ) : null}
            {errorMessage ? <p className="upload-inline-error">{errorMessage}</p> : null}
          </div>
        </div>
      </div>
    </section>
  );
}
