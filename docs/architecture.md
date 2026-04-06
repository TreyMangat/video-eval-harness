# VBench architecture

## Overview

VBench is a harness that runs the same video content through multiple AI models in parallel, collects structured responses, and measures how they compare. It is not a pipeline — every model sees identical frames and identical prompts, so comparisons are apples-to-apples.

## The 5-stage pipeline

1. **Ingest** — Adapters normalize video sources (local files, directories, manifests, UCF101, EPIC-KITCHENS, Ego4D, Build.ai). FFprobe extracts metadata.
2. **Segment** — Videos are chopped into temporal windows. Fixed-window is the default (10s coarse, 3s dense). Auto-windowing scales with video duration.
3. **Extract frames** — Uniform or keyframe sampling via FFmpeg. Default 8 frames per segment for coarse mode, 4 for dense. Contact sheets optional.
4. **Label** — Each (segment, model) pair is sent through a provider via ThreadPoolExecutor. Jinja2 prompt templates enforce structured JSON output. Disk-based response cache enables resume on failure.
5. **Evaluate** — Pairwise agreement matrices, ground-truth accuracy (fuzzy + exact), per-model summaries, optional LLM-judge agreement scoring.

## Storage

- **MongoDB Atlas** — primary metadata store. Collections: `videos`, `segments`, `extracted_frames`, `label_results`, `runs`. A factory pattern (`storage_factory.create_storage`) falls back to SQLite for local dev without a MongoDB URI.
- **Filesystem** — frames, uploaded videos, exported run artifacts (JSON / CSV / Parquet).
- **diskcache** — response cache keyed by `(model, prompt_hash, input_hash, variant_id)` to prevent duplicate API calls across runs.

## Providers

Three provider backends all implement the same `BaseProvider` interface:

- **OpenRouter** (primary) — one API key unlocks 10+ models. Handles retries, rate limiting, cost extraction from usage metadata. Supports image and base64 video inputs.
- **OpenAI native** — direct GPT API for users who have an OpenAI key.
- **Gemini native** — direct Google Gemini API.

## Deployment

- **Modal** — FastAPI backend with autoscaling serverless containers. `min_containers=1` keeps one container warm for latency. 900s function timeout for long benchmark runs. Volume mounts for frame files.
- **Vercel** — Next.js 15 frontend with SSR. Dynamic Open Graph images generated via the `opengraph-image.tsx` file convention. Static build reads committed JSON from `data/` at build time; runtime API calls go to the Modal backend.
- **MongoDB Atlas** — free M0 tier, us-west-2 region to minimize latency to Modal workers.

## Cost protection

Three layers prevent accidentally expensive runs:

1. **Auto-windowing** — window size scales with video duration
2. **`--max-segments` cap** — hard limit with uniform subsampling
3. **Budget guard** — confirmation prompt when API calls exceed 500

Server-side limits in `src/video_eval_harness/limits.py` enforce public-API caps (max file size, max clip duration, allowed model whitelist).
