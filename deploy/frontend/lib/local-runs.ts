import { promises as fs } from "fs";
import path from "path";

import type {
  FramePreview,
  LabelResult,
  ModelSummary,
  RunListItem,
  RunPayload,
  SegmentMedia,
  SweepMetrics,
} from "./types";
import { buildRunPayload, parseLabelResultRecord } from "./run-metrics";

type CsvRecord = Record<string, string>;
type JsonRunBundle = {
  results: LabelResult[];
  summary_overrides: Record<string, Partial<ModelSummary>>;
  agreement: Record<string, Record<string, number>> | null;
};
const FLAT_RUN_FILE_PATTERN = /^(run_.+)_results\.(json|csv)$/i;

async function exists(targetPath: string): Promise<boolean> {
  try {
    await fs.access(targetPath);
    return true;
  } catch {
    return false;
  }
}

function parseCsv(text: string): CsvRecord[] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];

    if (inQuotes) {
      if (char === '"' && next === '"') {
        field += '"';
        index += 1;
      } else if (char === '"') {
        inQuotes = false;
      } else {
        field += char;
      }
      continue;
    }

    if (char === '"') {
      inQuotes = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (char !== "\r") {
      field += char;
    }
  }

  if (field.length > 0 || row.length > 0) {
    row.push(field);
    rows.push(row);
  }

  if (rows.length === 0) {
    return [];
  }

  const headers = rows[0];
  return rows
    .slice(1)
    .filter((values) => values.some((value) => value.length > 0))
    .map((values) => {
      const record: CsvRecord = {};
      headers.forEach((header, headerIndex) => {
        record[header] = values[headerIndex] ?? "";
      });
      return record;
    });
}

function resolveRunsDirCandidate(candidate: string | undefined): string | null {
  if (!candidate) {
    return null;
  }
  const trimmed = candidate.trim();
  if (!trimmed) {
    return null;
  }
  return path.resolve(trimmed);
}

async function resolveRunsDir(dataDir?: string): Promise<string> {
  const candidates = [
    resolveRunsDirCandidate(dataDir),
    resolveRunsDirCandidate(process.env.VBENCH_RUNS_DIR),
    process.env.VBENCH_ARTIFACTS_DIR
      ? path.resolve(process.env.VBENCH_ARTIFACTS_DIR, "runs")
      : null,
    path.resolve(process.cwd(), "public", "data"),
    path.resolve(process.cwd(), "..", "..", "data"),
    path.resolve(process.cwd(), "..", "..", "artifacts", "runs"),
    path.resolve(process.cwd(), "artifacts", "runs"),
  ].filter((candidate): candidate is string => Boolean(candidate));

  for (const candidate of candidates) {
    if (await exists(candidate)) {
      return candidate;
    }
  }

  throw new Error(
    "Run data not found. Set VBENCH_RUNS_DIR or include exported JSON files in public/data."
  );
}

async function resolveArtifactsDir(dataDir?: string): Promise<string> {
  const runsDir = await resolveRunsDir(dataDir);
  return path.basename(runsDir) === "runs" ? path.dirname(runsDir) : runsDir;
}

function coerceSweepSummary(value: unknown): SweepMetrics | null {
  if (!value || typeof value !== "object") {
    return null;
  }

  const candidate =
    "sweep" in value && value.sweep && typeof value.sweep === "object"
      ? value.sweep
      : value;

  if (!candidate || typeof candidate !== "object") {
    return null;
  }

  const record = candidate as Partial<SweepMetrics>;
  if (!Array.isArray(record.cells) || !Array.isArray(record.stability)) {
    return null;
  }

  const variants =
    Array.isArray(record.variants) && record.variants.length > 0
      ? record.variants
      : [...new Set(record.cells.map((cell) => cell.variant_label))].sort();

  const parseSuccessMatrix =
    record.parse_success_matrix && typeof record.parse_success_matrix === "object"
      ? record.parse_success_matrix
      : Object.fromEntries(
          record.cells.reduce<Map<string, Record<string, number>>>((matrix, cell) => {
            const byVariant = matrix.get(cell.model_name) ?? {};
            byVariant[cell.variant_label] = cell.parse_success_rate;
            matrix.set(cell.model_name, byVariant);
            return matrix;
          }, new Map())
        );

  const variantIdByLabel =
    record.variant_id_by_label && typeof record.variant_id_by_label === "object"
      ? record.variant_id_by_label
      : Object.fromEntries(
          record.cells.map((cell) => [cell.variant_label, cell.variant_id])
        );

  const hasSweep =
    typeof record.has_sweep === "boolean" ? record.has_sweep : record.cells.length > 0;

  return {
    has_sweep: hasSweep,
    variants,
    cells: record.cells,
    stability: record.stability,
    agreement_by_variant:
      record.agreement_by_variant && typeof record.agreement_by_variant === "object"
        ? record.agreement_by_variant
        : {},
    parse_success_matrix: parseSuccessMatrix,
    variant_id_by_label: variantIdByLabel,
  };
}

async function findRunArtifact(
  runId: string,
  extension: "csv" | "json" | "sweep-summary",
  dataDir?: string
): Promise<string | null> {
  const runsDir = await resolveRunsDir(dataDir);
  const runDir = path.join(runsDir, runId);
  const candidates =
    extension === "json"
      ? [
          path.join(runDir, `${runId}_results.json`),
          path.join(runDir, "results.json"),
          path.join(runsDir, `${runId}_results.json`),
        ]
      : extension === "csv"
        ? [
            path.join(runDir, `${runId}_results.csv`),
            path.join(runDir, "results.csv"),
            path.join(runsDir, `${runId}_results.csv`),
          ]
        : [
            path.join(runDir, `${runId}_sweep_summary.json`),
            path.join(runDir, "sweep_summary.json"),
            path.join(runsDir, `${runId}_sweep_summary.json`),
          ];

  for (const candidate of candidates) {
    if (await exists(candidate)) {
      return candidate;
    }
  }

  return null;
}

function parseSummaryOverrides(raw: unknown): Record<string, Partial<ModelSummary>> {
  if (!raw || typeof raw !== "object") {
    return {};
  }

  const source = raw as Record<string, unknown>;
  const overrides: Record<string, Partial<ModelSummary>> = {};

  const summaries = source.summaries;
  if (summaries && typeof summaries === "object") {
    for (const [modelName, value] of Object.entries(summaries as Record<string, unknown>)) {
      if (!value || typeof value !== "object") {
        continue;
      }
      const record = value as Record<string, unknown>;
      overrides[modelName] = {
        exact_match_rate:
          typeof record.exact_match_rate === "number" ? record.exact_match_rate : null,
        fuzzy_match_rate:
          typeof record.fuzzy_match_rate === "number" ? record.fuzzy_match_rate : null,
      };
    }
  }

  const groundTruth =
    source.ground_truth_accuracy && typeof source.ground_truth_accuracy === "object"
      ? (source.ground_truth_accuracy as Record<string, unknown>)
      : source.accuracy && typeof source.accuracy === "object"
        ? (source.accuracy as Record<string, unknown>)
        : null;

  if (groundTruth) {
    for (const [modelName, value] of Object.entries(groundTruth)) {
      if (!value || typeof value !== "object") {
        continue;
      }
      const record = value as Record<string, unknown>;
      overrides[modelName] = {
        ...(overrides[modelName] ?? {}),
        exact_match_rate:
          typeof record.exact_match_rate === "number" ? record.exact_match_rate : null,
        fuzzy_match_rate:
          typeof record.fuzzy_match_rate === "number" ? record.fuzzy_match_rate : null,
      };
    }
  }

  return overrides;
}

function coerceAgreementMatrix(
  raw: unknown
): Record<string, Record<string, number>> | null {
  if (!raw || typeof raw !== "object") {
    return null;
  }

  const matrix: Record<string, Record<string, number>> = {};

  for (const [rowModel, rowValue] of Object.entries(raw as Record<string, unknown>)) {
    if (!rowValue || typeof rowValue !== "object") {
      continue;
    }

    const parsedRow: Record<string, number> = {};
    for (const [columnModel, score] of Object.entries(rowValue as Record<string, unknown>)) {
      if (typeof score === "number" && Number.isFinite(score)) {
        parsedRow[columnModel] = score;
      }
    }

    if (Object.keys(parsedRow).length > 0) {
      matrix[rowModel] = parsedRow;
    }
  }

  return Object.keys(matrix).length > 0 ? matrix : null;
}

async function loadJsonResults(runId: string, dataDir?: string): Promise<JsonRunBundle | null> {
  const artifactPath = await findRunArtifact(runId, "json", dataDir);
  if (!artifactPath) {
    return null;
  }

  const raw = JSON.parse(await fs.readFile(artifactPath, "utf-8")) as unknown;
  if (Array.isArray(raw)) {
    return {
      results: raw.map((record) => parseLabelResultRecord(record as Record<string, unknown>)),
      summary_overrides: {},
      agreement: null,
    };
  }
  if (!raw || typeof raw !== "object") {
    return null;
  }

  const record = raw as Record<string, unknown>;
  const results = Array.isArray(record.results)
    ? record.results.map((entry) => parseLabelResultRecord(entry as Record<string, unknown>))
    : [];

  return {
    results,
    summary_overrides: parseSummaryOverrides(record),
    agreement: coerceAgreementMatrix(record.agreement),
  };
}

async function loadCsvResults(runId: string, dataDir?: string): Promise<LabelResult[]> {
  const artifactPath = await findRunArtifact(runId, "csv", dataDir);
  if (!artifactPath) {
    return [];
  }

  const raw = await fs.readFile(artifactPath, "utf-8");
  return parseCsv(raw).map((record) => parseLabelResultRecord(record));
}

async function loadRunResults(
  runId: string,
  dataDir?: string
): Promise<JsonRunBundle> {
  const jsonResults = await loadJsonResults(runId, dataDir);
  if (jsonResults && jsonResults.results.length > 0) {
    return jsonResults;
  }
  return {
    results: await loadCsvResults(runId, dataDir),
    summary_overrides: {},
    agreement: null,
  };
}

async function loadSweepSummary(
  runId: string,
  dataDir?: string
): Promise<SweepMetrics | null> {
  const summaryPath = await findRunArtifact(runId, "sweep-summary", dataDir);
  if (!summaryPath) {
    return null;
  }

  const raw = JSON.parse(await fs.readFile(summaryPath, "utf-8")) as unknown;
  return coerceSweepSummary(raw);
}

function repoRootFromArtifacts(artifactsDir: string): string {
  return path.dirname(artifactsDir);
}

function resolveFramePath(rawPath: string, artifactsDir: string): string {
  if (path.isAbsolute(rawPath)) {
    return rawPath;
  }
  const normalized = rawPath
    .replaceAll("\\", path.sep)
    .replaceAll("/", path.sep);
  if (normalized.startsWith(`artifacts${path.sep}`)) {
    return path.resolve(repoRootFromArtifacts(artifactsDir), normalized);
  }
  return path.resolve(normalized);
}

function mimeTypeFor(filePath: string): string {
  const extension = path.extname(filePath).toLowerCase();
  if (extension === ".png") {
    return "image/png";
  }
  if (extension === ".webp") {
    return "image/webp";
  }
  return "image/jpeg";
}

async function toDataUrl(filePath: string): Promise<string | null> {
  if (!(await exists(filePath))) {
    return null;
  }
  const buffer = await fs.readFile(filePath);
  return `data:${mimeTypeFor(filePath)};base64,${buffer.toString("base64")}`;
}

async function findSegmentManifest(
  artifactsDir: string,
  videoId: string,
  segmentId: string,
  variantId?: string | null
): Promise<{ manifestPath: string; resolvedVariantId: string | null } | null> {
  const framesRoot = path.join(artifactsDir, "frames", videoId);
  if (!(await exists(framesRoot))) {
    return null;
  }

  if (variantId && variantId !== "default") {
    const explicitVariantManifest = path.join(
      framesRoot,
      variantId,
      segmentId,
      "metadata.json"
    );
    if (await exists(explicitVariantManifest)) {
      return { manifestPath: explicitVariantManifest, resolvedVariantId: variantId };
    }
  }

  const legacyManifest = path.join(framesRoot, segmentId, "metadata.json");
  if (await exists(legacyManifest)) {
    return { manifestPath: legacyManifest, resolvedVariantId: null };
  }

  const children = await fs.readdir(framesRoot, { withFileTypes: true });
  for (const child of children) {
    if (!child.isDirectory()) {
      continue;
    }
    const nestedManifest = path.join(framesRoot, child.name, segmentId, "metadata.json");
    if (await exists(nestedManifest)) {
      return { manifestPath: nestedManifest, resolvedVariantId: child.name };
    }
  }

  return null;
}

export async function listArtifactRuns(dataDir?: string): Promise<RunListItem[]> {
  const runsDir = await resolveRunsDir(dataDir);
  const entries = await fs.readdir(runsDir, { withFileTypes: true });
  const runIds = new Set<string>();

  for (const entry of entries) {
    if (entry.isDirectory()) {
      const hasArtifact =
        (await findRunArtifact(entry.name, "json", dataDir)) != null ||
        (await findRunArtifact(entry.name, "csv", dataDir)) != null;
      if (hasArtifact) {
        runIds.add(entry.name);
      }
      continue;
    }

    const match = entry.name.match(FLAT_RUN_FILE_PATTERN);
    if (match) {
      runIds.add(match[1]);
    }
  }

  const runs = await Promise.all(
    [...runIds].map(async (runId) => {
      const payload = await loadArtifactRun(runId, dataDir);
      if (!payload) {
        return null;
      }
      return {
        run_id: payload.run_id,
        created_at: payload.config.created_at,
        models: payload.models,
        prompt_version: payload.config.prompt_version,
        video_ids: payload.config.video_ids,
      };
    })
  );

  return runs
    .filter((item): item is RunListItem => item != null)
    .sort(
      (left, right) =>
        new Date(right.created_at).getTime() - new Date(left.created_at).getTime()
    );
}

export async function loadArtifactRun(
  runId: string,
  dataDir?: string
): Promise<RunPayload | null> {
  const runData = await loadRunResults(runId, dataDir);
  if (runData.results.length === 0) {
    return null;
  }

  const sweepSummary = await loadSweepSummary(runId, dataDir);
  const payload = buildRunPayload(
    runId,
    runData.results,
    sweepSummary,
    runData.summary_overrides
  );

  return {
    ...payload,
    agreement: runData.agreement ?? payload.agreement,
  };
}

export async function loadArtifactSegmentMedia(
  runId: string,
  segmentId: string,
  variantId?: string | null,
  dataDir?: string
): Promise<SegmentMedia | null> {
  const artifactsDir = await resolveArtifactsDir(dataDir);
  const runPayload = await loadArtifactRun(runId, dataDir);
  const segment = runPayload?.segments.find((entry) => entry.segment_id === segmentId);
  if (!runPayload || !segment) {
    return null;
  }

  const manifest = await findSegmentManifest(
    artifactsDir,
    segment.video_id,
    segmentId,
    variantId
  );

  if (!manifest) {
    return {
      run_id: runId,
      segment_id: segmentId,
      start_time_s: segment.start_time_s,
      end_time_s: segment.end_time_s,
      frame_timestamps_s: [],
      contact_sheet_data_url: null,
      frames: [],
      variant_id: variantId ?? null,
      variant_label: variantId ?? null,
    };
  }

  const payload = JSON.parse(await fs.readFile(manifest.manifestPath, "utf-8")) as {
    frame_paths?: string[];
    frame_timestamps_s?: number[];
    contact_sheet_path?: string | null;
  };

  const frames: FramePreview[] = await Promise.all(
    (payload.frame_paths ?? []).map(async (framePath, index) => ({
      timestamp_s: payload.frame_timestamps_s?.[index] ?? segment.start_time_s,
      data_url: await toDataUrl(resolveFramePath(framePath, artifactsDir)),
    }))
  );

  const contactSheetPath = payload.contact_sheet_path
    ? resolveFramePath(payload.contact_sheet_path, artifactsDir)
    : null;
  const variantLabel = manifest.resolvedVariantId
    ? Object.entries(runPayload.sweep?.variant_id_by_label ?? {}).find(
        ([, value]) => value === manifest.resolvedVariantId
      )?.[0] ?? manifest.resolvedVariantId
    : null;

  return {
    run_id: runId,
    segment_id: segmentId,
    start_time_s: segment.start_time_s,
    end_time_s: segment.end_time_s,
    frame_timestamps_s: payload.frame_timestamps_s ?? [],
    contact_sheet_data_url: contactSheetPath ? await toDataUrl(contactSheetPath) : null,
    frames,
    variant_id: manifest.resolvedVariantId,
    variant_label: variantLabel,
  };
}
