import type { RunListItem, RunPayload, SegmentMedia } from "./types";

export type HealthPayload = {
  status: string;
  version: string;
  limits: {
    max_clip_s: number;
    max_file_size_mb: number;
    max_models: number;
    allowed_models: string[];
  };
};

export type ApiModel = {
  name: string;
  display_name: string;
  model_id: string;
  provider: string;
  supports_images: boolean;
  description: string;
  tier?: "fast" | "frontier";
  estimated_cost_per_segment?: number;
};

export type BenchmarkJobResponse = {
  job_id: string;
  status: string;
  estimated_time_s: number;
};

export type SegmentPreviewSegment = {
  segment_index: number;
  segment_id: string;
  start_s: number;
  end_s: number;
  keyframe_base64: string;
};

export type SegmentPreviewResponse = {
  preview_id: string;
  video_id: string;
  video_filename: string;
  duration_s: number;
  segment_count: number;
  segments: SegmentPreviewSegment[];
};

export type JobStatusResponse = {
  status: "queued" | "running" | "complete" | "failed";
  run_id: string | null;
  error: string | null;
  stage?: string | null;
  progress?: string | null;
};

export type UploadAndBenchmarkOptions = {
  onStatus?: (message: string) => void;
  previewId?: string;
  groundTruth?: Array<{
    segment_index: number;
    label: string;
  }>;
};

class NonRetryableUploadError extends Error {}

const HEALTH_TIMEOUT_MS = 15_000;
const MODELS_TIMEOUT_MS = 15_000;
const RUNS_TIMEOUT_MS = 30_000;
const JOB_TIMEOUT_MS = 15_000;
const UPLOAD_TIMEOUT_MS = 120_000;

function getApiBaseUrl(): string {
  return process.env.NEXT_PUBLIC_API_URL || "";
}

function buildApiUrl(path: string): string {
  const apiUrl = getApiBaseUrl();

  if (!apiUrl) {
    throw new Error("Interactive mode is not configured.");
  }

  return new URL(path, apiUrl.endsWith("/") ? apiUrl : `${apiUrl}/`).toString();
}

function parseJsonText(text: string): unknown {
  if (!text) {
    return {};
  }
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return {};
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

async function readJson<T>(response: Response): Promise<T> {
  const text = await response.text();
  const parsed = parseJsonText(text);
  const data = parsed as T & {
    detail?: string;
    error?: string;
  };

  if (!response.ok) {
    throw new Error(data.detail || data.error || `Request failed (${response.status})`);
  }

  return data;
}

async function fetchWithTimeout(
  input: RequestInfo | URL,
  init: RequestInit,
  timeoutMs: number
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    controller.abort();
  }, timeoutMs);

  try {
    return await fetch(input, {
      ...init,
      signal: controller.signal,
    });
  } catch (error) {
    if (error instanceof Error && error.name === "AbortError") {
      throw new Error(`Request timed out after ${Math.round(timeoutMs / 1000)}s.`);
    }
    if (error instanceof TypeError) {
      throw new Error("Network request to benchmark server failed.");
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

export function isInteractiveMode(): boolean {
  return getApiBaseUrl().length > 0;
}

export async function getHealth(): Promise<HealthPayload> {
  const response = await fetchWithTimeout(
    buildApiUrl("api/health"),
    { cache: "no-store" },
    HEALTH_TIMEOUT_MS
  );
  return readJson<HealthPayload>(response);
}

export async function fetchModels(): Promise<{ models: ApiModel[] }> {
  const response = await fetchWithTimeout(
    buildApiUrl("api/models"),
    { cache: "no-store" },
    MODELS_TIMEOUT_MS
  );
  return readJson<{ models: ApiModel[] }>(response);
}

export async function uploadAndBenchmark(
  file: File | null,
  models?: string[],
  name?: string,
  options?: UploadAndBenchmarkOptions
): Promise<BenchmarkJobResponse> {
  const maxRetries = 2;

  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    const form = new FormData();
    if (file) {
      form.append("video", file);
    }
    if (options?.previewId) {
      form.append("preview_id", options.previewId);
    }
    if (models && models.length > 0) {
      form.append("models", JSON.stringify(models));
    }
    if (name) {
      form.append("name", name);
    }
    if (options?.groundTruth && options.groundTruth.length > 0) {
      form.append("ground_truth", JSON.stringify(options.groundTruth));
    }

    try {
      options?.onStatus?.(
        attempt > 0
          ? `Retry ${attempt}/${maxRetries} - ${file ? "uploading video" : "submitting benchmark"}...`
          : file
            ? "Uploading video..."
            : "Submitting benchmark..."
      );
      const response = await fetchWithTimeout(
        buildApiUrl("api/benchmark"),
        {
          method: "POST",
          body: form,
        },
        UPLOAD_TIMEOUT_MS
      );
      const text = await response.text();
      const parsed = parseJsonText(text) as BenchmarkJobResponse & {
        detail?: string;
        error?: string;
      };

      if (!response.ok) {
        const message =
          parsed.detail || parsed.error || `Server error ${response.status}`;
        if (response.status >= 500) {
          throw new Error(message);
        }
        throw new NonRetryableUploadError(message);
      }

      return parsed;
    } catch (error) {
      if (error instanceof NonRetryableUploadError) {
        throw error;
      }
      const message = error instanceof Error ? error.message : "Upload failed.";
      const isTimeout = message.startsWith("Request timed out after");
      if (attempt < maxRetries) {
        if (isTimeout) {
          options?.onStatus?.("Upload timed out. Retrying...");
          try {
            await getHealth();
          } catch {
            // Best-effort health check between retries.
          }
        } else {
          options?.onStatus?.("Upload failed. Retrying in 3 seconds...");
        }
        await sleep(3000);
        continue;
      }

      if (isTimeout) {
        throw new Error(
          "Upload timed out after multiple attempts. The server may be overloaded - try again in a minute."
        );
      }
      throw error;
    }
  }

  throw new Error("Upload failed.");
}

export async function previewSegments(
  file: File,
  options?: Pick<UploadAndBenchmarkOptions, "onStatus">
): Promise<SegmentPreviewResponse> {
  const maxRetries = 2;

  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    const form = new FormData();
    form.append("file", file);

    try {
      options?.onStatus?.(
        attempt > 0
          ? `Retry ${attempt}/${maxRetries} - uploading video for preview...`
          : "Uploading video and generating segment previews..."
      );
      const response = await fetchWithTimeout(
        buildApiUrl("api/segment-preview"),
        {
          method: "POST",
          body: form,
        },
        UPLOAD_TIMEOUT_MS
      );
      const text = await response.text();
      const parsed = parseJsonText(text) as SegmentPreviewResponse & {
        detail?: string;
        error?: string;
      };

      if (!response.ok) {
        const message = parsed.detail || parsed.error || `Server error ${response.status}`;
        if (response.status >= 500) {
          throw new Error(message);
        }
        throw new NonRetryableUploadError(message);
      }

      return parsed;
    } catch (error) {
      if (error instanceof NonRetryableUploadError) {
        throw error;
      }

      const message = error instanceof Error ? error.message : "Preview generation failed.";
      const isTimeout = message.startsWith("Request timed out after");
      if (attempt < maxRetries) {
        if (isTimeout) {
          options?.onStatus?.("Segment preview timed out. Retrying...");
          try {
            await getHealth();
          } catch {
            // Best-effort health check between retries.
          }
        } else {
          options?.onStatus?.("Preview generation failed. Retrying in 3 seconds...");
        }
        await sleep(3000);
        continue;
      }

      if (isTimeout) {
        throw new Error(
          "Segment preview timed out after multiple attempts. The server may still be starting up - try again in a minute."
        );
      }
      throw error;
    }
  }

  throw new Error("Preview generation failed.");
}

export async function pollJob(jobId: string): Promise<JobStatusResponse> {
  const response = await fetchWithTimeout(
    buildApiUrl(`api/jobs/${jobId}`),
    { cache: "no-store" },
    JOB_TIMEOUT_MS
  );
  return readJson<JobStatusResponse>(response);
}

export async function fetchRuns(): Promise<RunListItem[]> {
  const response = await fetchWithTimeout(
    buildApiUrl("api/runs"),
    { cache: "no-store" },
    RUNS_TIMEOUT_MS
  );
  return readJson<RunListItem[]>(response);
}

export async function fetchRun(runId: string): Promise<RunPayload> {
  const response = await fetchWithTimeout(
    buildApiUrl(`api/runs/${runId}`),
    { cache: "no-store" },
    RUNS_TIMEOUT_MS
  );
  return readJson<RunPayload>(response);
}

export async function fetchSegmentMedia(
  runId: string,
  segmentId: string,
  variantId?: string | null
): Promise<SegmentMedia> {
  const url = new URL(buildApiUrl(`api/runs/${runId}/segments/${segmentId}/media`));
  if (variantId) {
    url.searchParams.set("variantId", variantId);
  }
  const response = await fetchWithTimeout(
    url.toString(),
    { cache: "no-store" },
    RUNS_TIMEOUT_MS
  );
  return readJson<SegmentMedia>(response);
}
