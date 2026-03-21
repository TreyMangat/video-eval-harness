# Changelog

All notable changes to VBench are documented here.

## [v0.3.0](https://github.com/TreyMangat/video-eval-harness/releases/tag/v0.3.0) — 2026-03-21

### Sweep system complete
- **`vbench test-suite`**: one-command benchmark runner with best-practice defaults (fast models, 2 variants, max 25 segments, auto-windowing, cost confirmation)
- **`vbench estimate`**: preview API calls, cost, and time without running anything
- **`vbench compare`**: side-by-side run comparison with green/red deltas for parse rate, latency, confidence, agreement, and cost
- **`vbench export-sweep-summary`**: pre-computed sweep metrics as JSON for dashboards
- **`--max-segments`** flag: hard cap on segment count with uniform subsampling
- **Budget guard**: confirmation prompt when API calls exceed 500
- **Auto-windowing**: window size scales with video duration (<60s: 10s, 60-300s: 30s, 300-1800s: 60s, >1800s: 120s)
- **Human-readable run IDs**: `run_20260321_cooking_30s_sweep_a7f4` instead of `run_36bed372cb33`
- **`--name` flag** on run-benchmark, sweep, and test-suite for custom run IDs
- **`--action-vocabulary`** flag: constrain labels to a fixed taxonomy (Ego4D-style)
- **`--ground-truth`** flag: evaluate against known labels with exact/fuzzy match metrics

### Prompt engineering
- **`action_label`** template: constrains primary_action to concise verb phrases (max 5 words, no articles). Improved cross-model agreement from 0% to 100% on test content.
- **`action_label_v2`** template: adds "use the most general verb" instruction. Self-agreement improved 10-22% to 63-85% on real content. Cross-model agreement up 40-60%.

### Models
- Added **Llama 4 Maverick** (`meta-llama/llama-4-maverick`) to frontier tier
- Added fast tier: **Gemini 3 Flash**, **GPT-5.4 Mini**, **Qwen 3.5-27B**
- Removed Claude Sonnet 4.6 from default set (required model-specific prompt hack)

### Dataset adapters
- **Build.ai Egocentric-10K** adapter: local-first scanning with tar auto-extraction, JSON sidecar metadata, factory/worker filtering
- **`vbench download-dataset buildai-10k`**: fetch a single worker's shard from Hugging Face
- **Ego4D** adapter: parses ego4d.json manifest, auto-loads ground truth annotations
- **`--adapter`** flag on run-benchmark, sweep, list-videos for dataset-aware ingestion

### Parser improvements
- Truncated JSON repair: recovers primary_action from responses cut off by max_tokens
- Qwen parse rate: 67% to 100% on frontier benchmark

### Test content
- Added real video clips: Sintel action footage, Big Buck Bunny, trimmed Build.ai factory clips
- `scripts/download-test-clips.sh` for downloading CC-licensed test content

## [v0.2.0](https://github.com/TreyMangat/video-eval-harness/releases/tag/v0.2.0) — 2026-03-20

### Sweep foundation
- **Sweep orchestrator** (`sweep.py`): `SweepConfig`, `SweepAxis`, `ExtractionVariant`, `SweepOrchestrator` with deterministic variant IDs
- **Sweep-aware metrics**: `CellMetrics` (model x variant), `ModelStabilityScore` (self-agreement, rank stability), per-variant agreement matrices
- **`vbench sweep`** command: extraction sweep across frame counts and sampling methods
- **`--sweep`** flag on run-benchmark with `--frames` and `--methods` overrides
- **`--dry-run`** for sweep plan preview without API calls
- **`--model-filter`** for running a subset of models

### Evaluation
- Fuzzy action matching with tiered similarity (exact, Jaccard, root phrase)
- Agreement matrix using continuous similarity scores
- Verbosity stats and failure analysis per model
- Ground truth accuracy: exact match, fuzzy match, mean similarity

### Infrastructure
- Docker support with `docker-compose.prod.yml`
- GitHub Actions CI pipeline
- Response caching with `diskcache` for resume support
- JSON export alongside CSV/Parquet

### Schema
- Added sweep fields to `SegmentLabelResult`: `extraction_variant_id`, `extraction_label`, `num_frames_used`, `sampling_method_used`, `sweep_id`
- `GroundTruthLabel` schema for ground truth annotations
- `ModelRunSummary` for per-model statistics

## [v0.1.0](https://github.com/TreyMangat/video-eval-harness/releases/tag/v0.1.0) — 2026-03-20

### Initial release
- **Full pipeline**: `vbench run-benchmark` — ingest, segment, extract frames, label with models, summarize
- **OpenRouter provider**: unified API key for GPT, Gemini, Claude, Qwen, and more
- **Native providers**: OpenAI and Gemini direct API support
- **SQLite storage**: metadata, segments, label results, run configs
- **Streamlit viewer**: interactive result exploration
- **3 prompt templates**: concise, rich, strict_json
- **Segmentation**: fixed-window and scene-heuristic strategies
- **Frame extraction**: uniform sampling with configurable frame count
- **Structured output parsing**: handles JSON in markdown blocks, surrounding text, direct responses
- **Rich CLI output**: tables, progress bars, colored status
- **Pydantic schemas**: `VideoMetadata`, `Segment`, `ExtractedFrames`, `SegmentLabelResult`, `RunConfig`
- **Deterministic IDs**: SHA-256 based video and segment IDs
- **Config-driven**: YAML configs for models, benchmark settings, and prompts
