import { promises as fs } from "fs";
import path from "path";

import type { FramePreview, LabelResult, ModelSummary, RunListItem, RunPayload, SegmentMedia, SegmentSummary, SweepMetrics } from "./types";

const ROOT_ACTION_STOP_WORDS = new Set(["a","an","the","this","that","some","very","more","is","are","was","were","be","been","being","with","from","into","onto","upon","over","under","in","on","at","to","of","for","by","about","and","or","but","also","still","just","only","its","their","his","her"]);

async function exists(targetPath: string): Promise<boolean> {
  try { await fs.access(targetPath); return true; } catch { return false; }
}

function parseNumber(value: string | undefined): number | null {
  if (!value?.trim()) { return null; }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseInteger(value: string | undefined): number {
  const parsed = parseNumber(value);
  return parsed == null ? 0 : Math.trunc(parsed);
}

function parseBoolean(value: string | undefined): boolean {
  const normalized = (value ?? "").trim().toLowerCase();
  return normalized === "true" || normalized === "1" || normalized === "yes";
}

function parseList(value: string | undefined): string[] {
  if (!value?.trim()) { return []; }
  return value.split(";").map((item) => item.trim()).filter(Boolean);
}

function normalizeAction(action: string): string {
  let normalized = action.toLowerCase().trim();
  for (const prefix of ["the person is ", "person is ", "the user is ", "someone is "]) {
    if (normalized.startsWith(prefix)) { normalized = normalized.slice(prefix.length); break; }
  }
  return normalized.replace(/\.$/, "");
}

function extractRootPhrase(action: string): string {
  return normalizeAction(action).split(/\s+/).filter((word) => word && !ROOT_ACTION_STOP_WORDS.has(word)).join(" ");
}

function computeActionSimilarity(left: string, right: string): number {
  const a = normalizeAction(left);
  const b = normalizeAction(right);
  if (a === b) { return 1; }
  if (a.length > 3 && b.length > 3 && (a.includes(b) || b.includes(a))) { return 0.9; }
  const wordsA = new Set(a.split(/\s+/).filter(Boolean));
  const wordsB = new Set(b.split(/\s+/).filter(Boolean));
  if (wordsA.size === 0 || wordsB.size === 0) { return 0; }
  const intersection = [...wordsA].filter((word) => wordsB.has(word)).length;
  const union = new Set([...wordsA, ...wordsB]).size;
  const jaccard = union === 0 ? 0 : intersection / union;
  if (jaccard > 0.5) { return jaccard; }
  const rootA = extractRootPhrase(a);
  const rootB = extractRootPhrase(b);
  if (rootA && rootB && rootA === rootB) { return 0.8; }
  const rootWordsA = new Set(rootA.split(/\s+/).filter(Boolean));
  const rootWordsB = new Set(rootB.split(/\s+/).filter(Boolean));
  if (rootWordsA.size > 0 && rootWordsB.size > 0) {
    const rootIntersection = [...rootWordsA].filter((word) => rootWordsB.has(word)).length;
    const rootUnion = new Set([...rootWordsA, ...rootWordsB]).size;
    const rootJaccard = rootUnion === 0 ? 0 : rootIntersection / rootUnion;
    if (rootJaccard > 0.5) { return rootJaccard * 0.8; }
  }
  return 0;
}

function parseCsv(text: string): Array<Record<string, string>> {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];
    if (inQuotes) {
      if (char === '"' && next === '"') { field += '"'; index += 1; }
      else if (char === '"') { inQuotes = false; }
      else { field += char; }
      continue;
    }
    if (char === '"') { inQuotes = true; }
    else if (char === ',') { row.push(field); field = ""; }
    else if (char === '\n') { row.push(field); rows.push(row); row = []; field = ""; }
    else if (char !== '\r') { field += char; }
  }
  if (field.length > 0 || row.length > 0) { row.push(field); rows.push(row); }
  if (rows.length === 0) { return []; }
  const headers = rows[0];
  return rows.slice(1).filter((values) => values.some((value) => value.length > 0)).map((values) => {
    const record: Record<string, string> = {};
    headers.forEach((header, headerIndex) => { record[header] = values[headerIndex] ?? ""; });
    return record;
  });
}

function toLabelResult(record: Record<string, string>): LabelResult {
  return {
    run_id: record.run_id,
    video_id: record.video_id,
    segment_id: record.segment_id,
    start_time_s: parseNumber(record.start_time_s) ?? 0,
    end_time_s: parseNumber(record.end_time_s) ?? 0,
    model_name: record.model_name,
    provider: record.provider || "openrouter",
    primary_action: record.primary_action || null,
    secondary_actions: parseList(record.secondary_actions),
    description: record.description || null,
    objects: parseList(record.objects),
    environment_context: record.environment_context || null,
    confidence: parseNumber(record.confidence),
    reasoning_summary_or_notes: record.reasoning_summary_or_notes || null,
    uncertainty_flags: parseList(record.uncertainty_flags),
    parsed_success: parseBoolean(record.parsed_success),
    parse_error: record.parse_error || null,
    latency_ms: parseNumber(record.latency_ms),
    estimated_cost: parseNumber(record.estimated_cost),
    prompt_version: record.prompt_version || null,
    extraction_variant_id: record.extraction_variant_id || "",
    extraction_label: record.extraction_label || "",
    num_frames_used: parseInteger(record.num_frames_used),
    sampling_method_used: record.sampling_method_used || "",
    sweep_id: record.sweep_id || "",
    timestamp: record.timestamp || null,
  };
}

function computeModelSummary(results: LabelResult[], modelName: string): ModelSummary {
  const modelResults = results.filter((result) => result.model_name === modelName);
  const successful = modelResults.filter((result) => result.parsed_success);
  const latencies = successful.map((result) => result.latency_ms).filter((value): value is number => value != null && value > 0).sort((left, right) => left - right);
  const confidences = successful.map((result) => result.confidence).filter((value): value is number => value != null);
  const costs = successful.map((result) => result.estimated_cost).filter((value): value is number => value != null);
  return {
    model_name: modelName,
    total_segments: modelResults.length,
    successful_parses: successful.length,
    failed_parses: modelResults.length - successful.length,
    parse_success_rate: modelResults.length === 0 ? 0 : successful.length / modelResults.length,
    avg_latency_ms: latencies.length === 0 ? null : latencies.reduce((sum, value) => sum + value, 0) / latencies.length,
    median_latency_ms: latencies.length === 0 ? null : latencies[Math.floor(latencies.length / 2)],
    p95_latency_ms: latencies.length === 0 ? null : latencies[Math.floor(latencies.length * 0.95)],
    total_estimated_cost: costs.length === 0 ? null : costs.reduce((sum, value) => sum + value, 0),
    avg_confidence: confidences.length === 0 ? null : confidences.reduce((sum, value) => sum + value, 0) / confidences.length,
    consensus_alignment_rate: null,
  };
}

function computeAgreementMatrix(results: LabelResult[]): Record<string, Record<string, number>> {
  const bySegment = new Map<string, Map<string, string>>();
  const models = [...new Set(results.map((result) => result.model_name))].sort();
  for (const result of results) {
    if (!result.parsed_success || !result.primary_action) { continue; }
    const segment = bySegment.get(result.segment_id) ?? new Map<string, string>();
    segment.set(result.model_name, result.primary_action.toLowerCase().trim());
    bySegment.set(result.segment_id, segment);
  }
  const agreement: Record<string, Record<string, number>> = {};
  for (const modelA of models) {
    agreement[modelA] = {};
    for (const modelB of models) {
      if (modelA === modelB) { agreement[modelA][modelB] = 1; continue; }
      let similaritySum = 0; let total = 0;
      for (const labels of bySegment.values()) {
        const left = labels.get(modelA); const right = labels.get(modelB);
        if (!left || !right) { continue; }
        total += 1; similaritySum += computeActionSimilarity(left, right);
      }
      agreement[modelA][modelB] = total === 0 ? 0 : similaritySum / total;
    }
  }
  return agreement;
}

function variantId(result: LabelResult): string { return result.extraction_variant_id?.trim() ? result.extraction_variant_id : "default"; }
function variantLabel(result: LabelResult): string { return result.extraction_variant_id?.trim() ? result.extraction_label?.trim() || result.extraction_variant_id : "default"; }

function computeSweepMetrics(results: LabelResult[]): SweepMetrics {
  const cellsByKey = new Map<string, LabelResult[]>();
  const variantLabels = new Map<string, string>();
  const models = [...new Set(results.map((result) => result.model_name))].sort();
  for (const result of results) {
    const id = variantId(result); const label = variantLabel(result); variantLabels.set(id, label);
    const key = `${result.model_name}::${id}`;
    const current = cellsByKey.get(key) ?? []; current.push(result); cellsByKey.set(key, current);
  }
  const cells = [...cellsByKey.entries()].map(([key, cellResults]) => {
    const [modelName, currentVariantId] = key.split("::");
    const successful = cellResults.filter((result) => result.parsed_success);
    const latencies = successful.map((result) => result.latency_ms).filter((value): value is number => value != null && value > 0).sort((left, right) => left - right);
    const confidences = successful.map((result) => result.confidence).filter((value): value is number => value != null);
    const costs = successful.map((result) => result.estimated_cost).filter((value): value is number => value != null);
    return {
      model_name: modelName,
      variant_label: variantLabels.get(currentVariantId) ?? currentVariantId,
      variant_id: currentVariantId,
      total_segments: cellResults.length,
      successful_parses: successful.length,
      parse_success_rate: cellResults.length === 0 ? 0 : successful.length / cellResults.length,
      avg_latency_ms: latencies.length === 0 ? null : latencies.reduce((sum, value) => sum + value, 0) / latencies.length,
      median_latency_ms: latencies.length === 0 ? null : latencies[Math.floor(latencies.length / 2)],
      p95_latency_ms: latencies.length === 0 ? null : latencies[Math.floor(latencies.length * 0.95)],
      avg_confidence: confidences.length === 0 ? null : confidences.reduce((sum, value) => sum + value, 0) / confidences.length,
      total_estimated_cost: costs.length === 0 ? null : costs.reduce((sum, value) => sum + value, 0),
    };
  }).sort((left, right) => left.model_name.localeCompare(right.model_name) || left.variant_label.localeCompare(right.variant_label));
  const orderedVariantIds = [...variantLabels.keys()].sort();
  const stability = models.map((modelName) => {
    const segmentVariantActions = new Map<string, string>();
    for (const result of results) {
      if (result.model_name !== modelName || !result.parsed_success || !result.primary_action) { continue; }
      segmentVariantActions.set(`${result.segment_id}::${variantId(result)}`, normalizeAction(result.primary_action));
    }
    const segments = [...new Set([...segmentVariantActions.keys()].map((key) => key.split("::")[0]))].sort();
    let similaritySum = 0; let totalPairs = 0;
    for (const segmentId of segments) {
      const actions = orderedVariantIds.map((currentVariantId) => ({ variant_id: currentVariantId, action: segmentVariantActions.get(`${segmentId}::${currentVariantId}`) ?? null })).filter((entry): entry is { variant_id: string; action: string } => entry.action != null);
      for (let leftIndex = 0; leftIndex < actions.length; leftIndex += 1) {
        for (let rightIndex = leftIndex + 1; rightIndex < actions.length; rightIndex += 1) {
          totalPairs += 1; similaritySum += computeActionSimilarity(actions[leftIndex].action, actions[rightIndex].action);
        }
      }
    }
    const rankPositions: number[] = [];
    for (const currentVariantId of orderedVariantIds) {
      const ranked = models.map((currentModel) => {
        const currentResults = cellsByKey.get(`${currentModel}::${currentVariantId}`) ?? [];
        const rate = currentResults.length === 0 ? 0 : currentResults.filter((result) => result.parsed_success).length / currentResults.length;
        return { model_name: currentModel, parse_success_rate: rate };
      }).sort((left, right) => right.parse_success_rate - left.parse_success_rate);
      rankPositions.push(ranked.findIndex((entry) => entry.model_name === modelName) + 1);
    }
    const meanRank = rankPositions.reduce((sum, value) => sum + value, 0) / Math.max(rankPositions.length, 1);
    const variance = rankPositions.reduce((sum, value) => sum + (value - meanRank) ** 2, 0) / Math.max(rankPositions.length, 1);
    const rankStd = Math.sqrt(variance);
    return {
      model_name: modelName,
      self_agreement: totalPairs === 0 ? 0 : similaritySum / totalPairs,
      rank_positions: rankPositions,
      rank_stability: rankPositions.length <= 1 ? 1 : Math.max(0, 1 - rankStd / Math.max(models.length - 1, 1)),
    };
  });
  const agreementByVariant: Record<string, Record<string, number>> = {};
  for (const currentVariantId of orderedVariantIds) {
    const currentVariantResults = results.filter((result) => variantId(result) === currentVariantId);
    if (currentVariantResults.length > 0) { agreementByVariant[variantLabels.get(currentVariantId) ?? currentVariantId] = computeAgreementMatrix(currentVariantResults); }
  }
  const parseSuccessMatrix: Record<string, Record<string, number>> = {};
  const variantIdByLabel: Record<string, string> = {};
  for (const [currentVariantId, label] of variantLabels.entries()) { variantIdByLabel[label] = currentVariantId; }
  for (const cell of cells) { parseSuccessMatrix[cell.model_name] ??= {}; parseSuccessMatrix[cell.model_name][cell.variant_label] = cell.parse_success_rate; }
  return { has_sweep: results.some((result) => Boolean(result.extraction_variant_id?.trim())), variants: [...variantLabels.values()].sort(), cells, stability, agreement_by_variant: agreementByVariant, parse_success_matrix: parseSuccessMatrix, variant_id_by_label: variantIdByLabel };
}

function buildSegments(results: LabelResult[]): SegmentSummary[] {
  const bySegment = new Map<string, LabelResult[]>();
  for (const result of results) { const current = bySegment.get(result.segment_id) ?? []; current.push(result); bySegment.set(result.segment_id, current); }
  return [...bySegment.entries()].map(([segmentId, segmentResults]) => {
    const first = [...segmentResults].sort((left, right) => left.start_time_s - right.start_time_s)[0];
    return {
      segment_id: segmentId,
      video_id: first.video_id,
      video_filename: first.video_id,
      segment_index: 0,
      start_time_s: first.start_time_s,
      end_time_s: first.end_time_s,
      duration_s: first.end_time_s - first.start_time_s,
      segmentation_mode: "exported_run",
      frame_count: Math.max(...segmentResults.map((result) => result.num_frames_used ?? 0), 0),
      frame_timestamps_s: [],
      has_contact_sheet: false,
    };
  }).sort((left, right) => left.start_time_s - right.start_time_s).map((segment, index) => ({ ...segment, segment_index: index }));
}

function buildVideos(results: LabelResult[]): RunPayload["videos"] { return [...new Set(results.map((result) => result.video_id))].map((videoId) => ({ video_id: videoId, filename: videoId })); }

async function resolveArtifactsDir(): Promise<string> {
  const candidates = [process.env.VBENCH_ARTIFACTS_DIR, path.resolve(process.cwd(), "..", "..", "artifacts"), path.resolve(process.cwd(), "artifacts")].filter((candidate): candidate is string => Boolean(candidate));
  for (const candidate of candidates) { if (await exists(candidate)) { return candidate; } }
  throw new Error("Artifacts directory not found. Set VBENCH_ARTIFACTS_DIR for the frontend.");
}

function repoRootFromArtifacts(artifactsDir: string): string { return path.dirname(artifactsDir); }

async function findRunArtifact(runId: string, extension: "csv" | "json"): Promise<string | null> {
  const artifactsDir = await resolveArtifactsDir();
  const runDir = path.join(artifactsDir, "runs", runId);
  const candidates = extension === "json" ? [path.join(runDir, `${runId}_results.json`), path.join(runDir, "results.json")] : [path.join(runDir, `${runId}_results.csv`)];
  for (const candidate of candidates) { if (await exists(candidate)) { return candidate; } }
  return null;
}

async function loadCsvResults(runId: string): Promise<LabelResult[]> {
  const artifactPath = await findRunArtifact(runId, "csv");
  if (!artifactPath) { return []; }
  const raw = await fs.readFile(artifactPath, "utf-8");
  return parseCsv(raw).map(toLabelResult);
}

function buildRunPayload(runId: string, results: LabelResult[]): RunPayload {
  const models = [...new Set(results.map((result) => result.model_name))].sort();
  const videoIds = [...new Set(results.map((result) => result.video_id))];
  const createdAt = results.map((result) => result.timestamp).filter((value): value is string => Boolean(value)).sort()[0] ?? new Date().toISOString();
  return {
    run_id: runId,
    config: { models, prompt_version: results.find((result) => result.prompt_version)?.prompt_version ?? "unknown", segmentation_mode: "unknown", segmentation_config: {}, extraction_config: {}, video_ids: videoIds, created_at: createdAt },
    models,
    videos: buildVideos(results),
    summaries: Object.fromEntries(models.map((modelName) => [modelName, computeModelSummary(results, modelName)])),
    agreement: computeAgreementMatrix(results),
    segments: buildSegments(results),
    results,
    sweep: computeSweepMetrics(results),
  };
}

function resolveFramePath(rawPath: string, artifactsDir: string): string {
  if (path.isAbsolute(rawPath)) { return rawPath; }
  const normalized = rawPath.replaceAll("\\", path.sep).replaceAll("/", path.sep);
  if (normalized.startsWith(`artifacts${path.sep}`)) { return path.resolve(repoRootFromArtifacts(artifactsDir), normalized); }
  return path.resolve(normalized);
}

function mimeTypeFor(filePath: string): string {
  const extension = path.extname(filePath).toLowerCase();
  if (extension === ".png") { return "image/png"; }
  if (extension === ".webp") { return "image/webp"; }
  return "image/jpeg";
}

async function toDataUrl(filePath: string): Promise<string | null> {
  if (!(await exists(filePath))) { return null; }
  const buffer = await fs.readFile(filePath);
  return `data:${mimeTypeFor(filePath)};base64,${buffer.toString("base64")}`;
}

async function findSegmentManifest(artifactsDir: string, videoId: string, segmentId: string, variantId?: string | null): Promise<{ manifestPath: string; resolvedVariantId: string | null } | null> {
  const framesRoot = path.join(artifactsDir, "frames", videoId);
  if (!(await exists(framesRoot))) { return null; }
  if (variantId && variantId !== "default") {
    const explicitVariantManifest = path.join(framesRoot, variantId, segmentId, "metadata.json");
    if (await exists(explicitVariantManifest)) { return { manifestPath: explicitVariantManifest, resolvedVariantId: variantId }; }
  }
  const legacyManifest = path.join(framesRoot, segmentId, "metadata.json");
  if (await exists(legacyManifest)) { return { manifestPath: legacyManifest, resolvedVariantId: null }; }
  const children = await fs.readdir(framesRoot, { withFileTypes: true });
  for (const child of children) {
    if (!child.isDirectory()) { continue; }
    const nestedManifest = path.join(framesRoot, child.name, segmentId, "metadata.json");
    if (await exists(nestedManifest)) { return { manifestPath: nestedManifest, resolvedVariantId: child.name }; }
  }
  return null;
}

export async function listArtifactRuns(): Promise<RunListItem[]> {
  const artifactsDir = await resolveArtifactsDir();
  const runsDir = path.join(artifactsDir, "runs");
  if (!(await exists(runsDir))) { return []; }
  const entries = await fs.readdir(runsDir, { withFileTypes: true });
  const runs = await Promise.all(entries.filter((entry) => entry.isDirectory()).map(async (entry) => {
    const payload = await loadArtifactRun(entry.name);
    return payload ? { run_id: payload.run_id, created_at: payload.config.created_at, models: payload.models, prompt_version: payload.config.prompt_version, video_ids: payload.config.video_ids } : null;
  }));
  return runs.filter((item): item is RunListItem => item != null).sort((left, right) => new Date(right.created_at).getTime() - new Date(left.created_at).getTime());
}

export async function loadArtifactRun(runId: string): Promise<RunPayload | null> {
  const results = await loadCsvResults(runId);
  return results.length === 0 ? null : buildRunPayload(runId, results);
}

export async function loadArtifactSegmentMedia(runId: string, segmentId: string, variantId?: string | null): Promise<SegmentMedia | null> {
  const artifactsDir = await resolveArtifactsDir();
  const runPayload = await loadArtifactRun(runId);
  const segment = runPayload?.segments.find((entry) => entry.segment_id === segmentId);
  if (!runPayload || !segment) { return null; }
  const manifest = await findSegmentManifest(artifactsDir, segment.video_id, segmentId, variantId);
  if (!manifest) {
    return { run_id: runId, segment_id: segmentId, start_time_s: segment.start_time_s, end_time_s: segment.end_time_s, frame_timestamps_s: [], contact_sheet_data_url: null, frames: [], variant_id: variantId ?? null, variant_label: variantId ?? null };
  }
  const payload = JSON.parse(await fs.readFile(manifest.manifestPath, "utf-8")) as { frame_paths?: string[]; frame_timestamps_s?: number[]; contact_sheet_path?: string | null; };
  const frames: FramePreview[] = await Promise.all((payload.frame_paths ?? []).map(async (framePath, index) => ({ timestamp_s: payload.frame_timestamps_s?.[index] ?? segment.start_time_s, data_url: await toDataUrl(resolveFramePath(framePath, artifactsDir)) })));
  const contactSheetPath = payload.contact_sheet_path ? resolveFramePath(payload.contact_sheet_path, artifactsDir) : null;
  const variantLabel = manifest.resolvedVariantId ? Object.entries(runPayload.sweep?.variant_id_by_label ?? {}).find(([, value]) => value === manifest.resolvedVariantId)?.[0] ?? manifest.resolvedVariantId : null;
  return { run_id: runId, segment_id: segmentId, start_time_s: segment.start_time_s, end_time_s: segment.end_time_s, frame_timestamps_s: payload.frame_timestamps_s ?? [], contact_sheet_data_url: contactSheetPath ? await toDataUrl(contactSheetPath) : null, frames, variant_id: manifest.resolvedVariantId, variant_label: variantLabel };
}
