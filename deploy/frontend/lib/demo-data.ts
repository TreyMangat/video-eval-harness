import type { RunPayload, RunListItem } from "./types";

// Benchmark models — all receive identical input, compared head-to-head.
const MODELS = [
  "gemini-3.1-pro",
  "gpt-5.4",
  "qwen3.5-vl",
  "claude-sonnet-4.6",
];

const ACTIONS = [
  "person walking forward",
  "person opening a door",
  "person reaching for an object",
  "person typing on keyboard",
  "person standing up from chair",
  "person looking at phone",
  "person picking up cup",
  "person talking to someone",
];

const SECONDARY_ACTIONS = [
  ["looking around", "adjusting bag"],
  ["reaching for handle", "stepping forward"],
  ["extending arm", "grasping"],
  ["looking at screen", "moving fingers"],
  ["pushing off armrests", "shifting weight"],
  ["scrolling", "holding phone"],
  ["gripping handle", "lifting"],
  ["gesturing", "nodding"],
];

const OBJECTS = [
  ["backpack", "door", "hallway"],
  ["door", "handle", "wall"],
  ["table", "cup", "papers"],
  ["keyboard", "monitor", "desk"],
  ["chair", "desk", "laptop"],
  ["phone", "table", "chair"],
  ["cup", "counter", "sink"],
  ["person", "table", "chairs"],
];

const ENVIRONMENTS = [
  "indoor office hallway with fluorescent lighting",
  "small conference room with glass walls",
  "open-plan office with natural lighting",
  "kitchen/break room area",
  "reception area near entrance",
  "cubicle workspace",
  "outdoor courtyard",
  "parking garage entrance",
];

function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 16807 + 0) % 2147483647;
    return s / 2147483647;
  };
}

function generateDemoRun(
  runId: string,
  createdAt: string,
  models: string[],
  videoName: string,
  numSegments: number,
  seed: number
): RunPayload {
  const rand = seededRandom(seed);
  const videoId = `vid_${runId.slice(0, 8)}`;

  const segments = Array.from({ length: numSegments }, (_, i) => ({
    segment_id: `${videoId}_seg${String(i).padStart(4, "0")}`,
    video_id: videoId,
    video_filename: videoName,
    segment_index: i,
    start_time_s: i * 10,
    end_time_s: (i + 1) * 10,
    duration_s: 10,
    segmentation_mode: "fixed_window",
    frame_count: 8,
    frame_timestamps_s: Array.from({ length: 8 }, (_, j) => i * 10 + j * 1.25),
    has_contact_sheet: false,
  }));

  const results = segments.flatMap((seg) =>
    models.map((model) => {
      const actionIdx = Math.floor(rand() * ACTIONS.length);
      // Realistic latency/cost/confidence per frontier model
      const latencyBase =
        model.includes("pro")
          ? 2200
          : model.includes("gpt")
            ? 1800
            : model.includes("claude")
              ? 1600
              : 1400; // qwen
      const costBase =
        model.includes("pro")
          ? 0.003
          : model.includes("gpt")
            ? 0.0025
            : model.includes("claude")
              ? 0.002
              : 0.0008; // qwen
      const confBase =
        model.includes("pro")
          ? 0.91
          : model.includes("gpt")
            ? 0.89
            : model.includes("claude")
              ? 0.88
              : 0.85; // qwen

      const parseFail = rand() < 0.03;

      return {
        run_id: runId,
        video_id: videoId,
        segment_id: seg.segment_id,
        start_time_s: seg.start_time_s,
        end_time_s: seg.end_time_s,
        model_name: model,
        provider: "openrouter",
        primary_action: parseFail ? null : ACTIONS[actionIdx],
        secondary_actions: parseFail ? [] : SECONDARY_ACTIONS[actionIdx],
        description: parseFail
          ? null
          : `The person is ${ACTIONS[actionIdx]} in a ${ENVIRONMENTS[actionIdx % ENVIRONMENTS.length]}.`,
        objects: parseFail ? [] : OBJECTS[actionIdx],
        environment_context: parseFail
          ? null
          : ENVIRONMENTS[actionIdx % ENVIRONMENTS.length],
        confidence: parseFail ? null : Math.min(1, confBase + (rand() - 0.5) * 0.15),
        reasoning_summary_or_notes: parseFail
          ? null
          : "Based on body posture and movement direction across frames.",
        uncertainty_flags: rand() < 0.15 ? ["low_visibility"] : [],
        parsed_success: !parseFail,
        parse_error: parseFail ? "JSON extraction failed: no valid object found" : null,
        latency_ms: latencyBase + (rand() - 0.5) * latencyBase * 0.6,
        estimated_cost: costBase * (0.8 + rand() * 0.4),
        prompt_version: "concise",
      };
    })
  );

  // Build summaries
  const summaries: RunPayload["summaries"] = {};
  for (const model of models) {
    const modelResults = results.filter((r) => r.model_name === model);
    const successful = modelResults.filter((r) => r.parsed_success);
    const latencies = successful.map((r) => r.latency_ms!).sort((a, b) => a - b);
    const costs = successful.map((r) => r.estimated_cost!);
    const confidences = successful
      .map((r) => r.confidence!)
      .filter((c) => c != null);

    summaries[model] = {
      model_name: model,
      total_segments: modelResults.length,
      successful_parses: successful.length,
      failed_parses: modelResults.length - successful.length,
      parse_success_rate:
        modelResults.length > 0 ? successful.length / modelResults.length : 0,
      avg_latency_ms:
        latencies.length > 0
          ? latencies.reduce((a, b) => a + b, 0) / latencies.length
          : null,
      median_latency_ms:
        latencies.length > 0
          ? latencies[Math.floor(latencies.length / 2)]
          : null,
      p95_latency_ms:
        latencies.length > 0
          ? latencies[Math.floor(latencies.length * 0.95)]
          : null,
      total_estimated_cost:
        costs.length > 0 ? costs.reduce((a, b) => a + b, 0) : null,
      avg_confidence:
        confidences.length > 0
          ? confidences.reduce((a, b) => a + b, 0) / confidences.length
          : null,
      consensus_alignment_rate: null,
    };
  }

  // Build agreement matrix
  const agreement: Record<string, Record<string, number>> = {};
  for (const m1 of models) {
    agreement[m1] = {};
    for (const m2 of models) {
      if (m1 === m2) {
        agreement[m1][m2] = 1;
      } else {
        let matches = 0;
        let total = 0;
        for (const seg of segments) {
          const r1 = results.find(
            (r) => r.segment_id === seg.segment_id && r.model_name === m1
          );
          const r2 = results.find(
            (r) => r.segment_id === seg.segment_id && r.model_name === m2
          );
          if (r1?.primary_action && r2?.primary_action) {
            total++;
            if (r1.primary_action === r2.primary_action) matches++;
          }
        }
        agreement[m1][m2] = total > 0 ? matches / total : 0;
      }
    }
  }

  return {
    run_id: runId,
    config: {
      models,
      prompt_version: "concise",
      segmentation_mode: "fixed_window",
      segmentation_config: { mode: "fixed_window", window_size_s: 10, stride_s: null },
      extraction_config: { num_frames: 8, method: "uniform" },
      model_configs: Object.fromEntries(
        models.map((model) => [model, {}])
      ),
      video_ids: [videoId],
      created_at: createdAt,
    },
    models,
    videos: [
      {
        video_id: videoId,
        source_path: `/data/${videoName}`,
        filename: videoName,
        duration_s: numSegments * 10,
        width: 1920,
        height: 1080,
        fps: 30,
        codec: "h264",
        file_size_bytes: numSegments * 2_500_000,
        ingested_at: createdAt,
      },
    ],
    summaries,
    agreement,
    segments,
    results,
  };
}

export const DEMO_RUNS: RunPayload[] = [
  generateDemoRun(
    "run_a1b2c3d4e5f6",
    "2026-03-18T14:30:00Z",
    ["gemini-3.1-pro", "gpt-5.4", "qwen3.5-vl", "claude-sonnet-4.6"],
    "office_walkthrough.mp4",
    6,
    42
  ),
  generateDemoRun(
    "run_x7y8z9w0v1u2",
    "2026-03-17T09:15:00Z",
    ["gemini-3.1-pro", "gpt-5.4", "claude-sonnet-4.6"],
    "kitchen_activity.mp4",
    8,
    137
  ),
  generateDemoRun(
    "run_m3n4o5p6q7r8",
    "2026-03-16T16:45:00Z",
    ["gemini-3.1-pro", "gpt-5.4"],
    "meeting_room.mp4",
    4,
    999
  ),
];

export const DEMO_RUN_LIST: RunListItem[] = DEMO_RUNS.map((run) => ({
  run_id: run.run_id,
  created_at: run.config.created_at,
  models: run.models,
  prompt_version: run.config.prompt_version,
  video_ids: run.config.video_ids,
}));
