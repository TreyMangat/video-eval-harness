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

export type JobStatusResponse = {
  status: "queued" | "running" | "complete" | "failed";
  run_id: string | null;
  error: string | null;
  stage?: string | null;
  progress?: string | null;
};

export type UploadAndBenchmarkOptions = {
  onStatus?: (message: string) => void;
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
  file: File,
  models?: string[],
  name?: string,
  options?: UploadAndBenchmarkOptions
): Promise<BenchmarkJobResponse> {
  const maxRetries = 2;

  for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
    const form = new FormData();
    form.append("video", file);
    if (models && models.length > 0) {
      form.append("models", JSON.stringify(models));
    }
    if (name) {
      form.append("name", name);
    }

    try {
      options?.onStatus?.(
        attempt > 0 ? `Retry ${attempt}/${maxRetries} - uploading video...` : "Uploading video..."
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
        await new Promise((resolve) => {
          setTimeout(resolve, 3000);
        });
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
