"use client";

import { ChangeEvent, DragEvent, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import {
  fetchModels,
  getHealth,
  isInteractiveMode,
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
const DEFAULT_LIMITS: HealthPayload["limits"] = {
  max_clip_s: 60,
  max_file_size_mb: 100,
  max_models: DEFAULT_MODEL_CATALOG.length,
  allowed_models: DEFAULT_MODEL_CATALOG.map((model) => model.name),
};

type HealthStatus = "static" | "checking" | "ready" | "offline";
type JobState = {
  jobId: string;
  status: "queued" | "running";
  estimatedTimeS: number;
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

export function UploadZone() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const interactiveMode = isInteractiveMode();
  const [healthStatus, setHealthStatus] = useState<HealthStatus>(
    interactiveMode ? "checking" : "static"
  );
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

  const limits = health?.limits ?? DEFAULT_LIMITS;
  const readyForUpload = interactiveMode && healthStatus === "ready";
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
      setHealthStatus("static");
      return;
    }

    let cancelled = false;

    async function loadInteractiveState(): Promise<void> {
      try {
        const [healthPayload, modelPayload] = await Promise.all([getHealth(), fetchModels()]);
        if (cancelled) {
          return;
        }

        setHealth(healthPayload);
        if (Array.isArray(modelPayload.models) && modelPayload.models.length > 0) {
          setModelOptions(modelPayload.models);
          const preferredSelection = modelPayload.models
            .filter((model) => modelTier(model) === "fast")
            .map((model) => model.name)
            .slice(0, healthPayload.limits.max_models);
          setSelectedModels(
            preferredSelection.length > 0
              ? preferredSelection
              : modelPayload.models
                  .map((model) => model.name)
                  .slice(0, healthPayload.limits.max_models)
          );
        }
        setHealthStatus("ready");
      } catch (error) {
        if (cancelled) {
          return;
        }
        setHealthStatus("offline");
        setErrorMessage(
          error instanceof Error ? error.message : "Interactive backend is unavailable."
        );
      }
    }

    void loadInteractiveState();
    return () => {
      cancelled = true;
    };
  }, [interactiveMode]);

  useEffect(() => {
    if (!isJobActive(job)) {
      return;
    }

    let cancelled = false;
    const intervalId = window.setInterval(() => {
      void (async () => {
        try {
          const jobStatus = await pollJob(job.jobId);
          if (cancelled) {
            return;
          }

          if (jobStatus.status === "complete" && jobStatus.run_id) {
            setStatusMessage("Opening results...");
            setJob(null);
            router.push(`/report/${jobStatus.run_id}`);
            return;
          }

          if (jobStatus.status === "failed") {
            setJob(null);
            setErrorMessage(jobStatus.error || "Benchmark failed.");
            return;
          }

          setJob((current) =>
            current
              ? {
                  ...current,
                  status: jobStatus.status === "queued" ? "queued" : "running",
                }
              : current
          );
          setStatusMessage("Running benchmark...");
        } catch (error) {
          if (!cancelled) {
            setJob(null);
            setErrorMessage(error instanceof Error ? error.message : "Failed to poll job status.");
          }
        }
      })();
    }, 5000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [job, router]);

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
            !readyForUpload ||
            isSubmitting ||
            isJobActive(job) ||
            (!checked && selectedModels.length >= limits.max_models)
          }
        />
      </label>
    );
  }

  async function startBenchmark(): Promise<void> {
    if (!readyForUpload || !selectedFile || selectedModels.length === 0 || isSubmitting) {
      return;
    }

    setIsSubmitting(true);
    setErrorMessage(null);
    setStatusMessage("Uploading clip...");

    try {
      const response: BenchmarkJobResponse = await uploadAndBenchmark(
        selectedFile,
        selectedModels,
        benchmarkName.trim() || undefined
      );
      setJob({
        jobId: response.job_id,
        status: response.status === "queued" ? "queued" : "running",
        estimatedTimeS: response.estimated_time_s,
      });
      setStatusMessage("Running benchmark...");
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Upload failed.");
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
    <section
      className={`visual-card upload-inline-card dashboard-section-card ${
        healthStatus !== "ready" ? "is-disabled" : ""
      }`}
    >
      <div className="upload-inline-card-body">
        <div className="upload-inline-grid">
          <div
            className={`upload-inline-dropzone ${isDragActive ? "drag-active" : ""}`}
            onClick={() => {
              if (healthStatus === "ready") {
                fileInputRef.current?.click();
              }
            }}
            onDrop={handleDrop}
            onDragEnter={(event) => {
              event.preventDefault();
              if (healthStatus === "ready") {
                setIsDragActive(true);
              }
            }}
            onDragOver={(event) => {
              event.preventDefault();
              if (healthStatus === "ready") {
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
            tabIndex={healthStatus === "ready" ? 0 : -1}
            onKeyDown={(event) => {
              if (healthStatus !== "ready") {
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
              disabled={healthStatus !== "ready"}
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
              {healthStatus === "offline" ? (
                <p className="upload-inline-error">Backend unavailable. Check your API URL.</p>
              ) : null}
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
                disabled={!readyForUpload || isSubmitting || isJobActive(job)}
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
                !readyForUpload ||
                isSubmitting ||
                isJobActive(job) ||
                !selectedFile ||
                selectedModels.length === 0
              }
            >
              Start Benchmark
            </button>

            {isJobActive(job) ? (
              <div className="info-banner upload-inline-progress" aria-live="polite">
                <span className="upload-spinner" aria-hidden="true" />
                <div>
                  <strong>Running benchmark... ~{job.estimatedTimeS} seconds</strong>
                  <p>
                    {job.status === "queued"
                      ? "Your clip is queued in Modal."
                      : "Polling the backend every 5 seconds until results are ready."}
                  </p>
                </div>
              </div>
            ) : null}

            {statusMessage ? <p className="upload-inline-status">{statusMessage}</p> : null}
            {errorMessage ? <p className="upload-inline-error">{errorMessage}</p> : null}
          </div>
        </div>
      </div>
    </section>
  );
}
