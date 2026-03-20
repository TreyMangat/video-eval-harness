# VBench - Multi-Model Video Segmentation & Labeling Benchmark Harness

[![CI](https://github.com/TreyMangat/video-eval-harness/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/TreyMangat/video-eval-harness/actions/workflows/ci.yml)

A local-first, developer-friendly evaluation harness for comparing multiple frontier vision-language models on temporal video understanding tasks. Designed for egocentric action recognition, but applicable to any video labeling workflow.

## What It Does

1. **Ingests** local videos (mp4, avi, mov, mkv, webm)
2. **Segments** them into temporal windows (fixed-window or scene-boundary heuristic)
3. **Extracts** representative frames from each segment
4. **Labels** each segment by sending the same frames + prompt to multiple models via OpenRouter
5. **Compares** model outputs: parse success, latency, cost, agreement, and optional ground-truth accuracy
6. **Exports** results to CSV/Parquet and provides a Streamlit viewer for exploration

All through a single API key (OpenRouter) and a clean CLI.

## Quick start (3 commands)

```bash
# Install
pip install -e ".[dev]"

# Smoke test (3 fast models, ~$0.03)
vbench run-benchmark path/to/video.mp4 --config configs/benchmark_fast.yaml

# View results
streamlit run src/video_eval_harness/viewer.py
```

### Prerequisites

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed and on PATH
- An [OpenRouter](https://openrouter.ai/) API key

### Configure

```bash
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY=your-key-here
```

Edit `configs/models.yaml` to enable/disable models. Edit `configs/benchmark.yaml` to adjust segmentation, extraction, and prompt settings.

### Run a Benchmark

```bash
# Full pipeline on a single video (4 frontier models)
vbench run-benchmark path/to/video.mp4

# Full pipeline on a directory of videos
vbench run-benchmark path/to/videos/

# With overrides
vbench run-benchmark video.mp4 --window 15 --num-frames 4 --prompt rich

# Run only specific models
vbench run-benchmark video.mp4 --model-filter gemini-3.1-pro,gpt-5.4

# With ground truth evaluation
vbench run-benchmark video.mp4 --ground-truth annotations.json

# Using Ego4D dataset adapter (auto-loads ground truth from manifest)
vbench run-benchmark path/to/clips/ --adapter ego4d --manifest path/to/ego4d.json
```

## Sweep mode

Sweep benchmarks extraction strategies (frame count x sampling method) alongside models:

```bash
# Preview the matrix without API calls
vbench sweep path/to/video.mp4 --dry-run

# Run full sweep (3 models x 6 extraction configs, ~$0.50)
vbench sweep path/to/video.mp4 --config configs/benchmark_fast.yaml

# Compare two runs
vbench compare <run_id_1> <run_id_2>
```

### View Results

```bash
# List all runs
vbench inspect-run

# Inspect a specific run
vbench inspect-run run_abc123def456

# Evaluate and export (CSV, Parquet, and JSON by default)
vbench evaluate run_abc123def456
vbench export run_abc123def456 --output ./exports

# Streamlit viewer (requires [ui] extras)
streamlit run src/video_eval_harness/viewer.py
```

## Public Visual Test Path

If you want a public, browser-based version instead of the local CLI plus Streamlit flow:

1. deploy the Modal backend in [`deploy/modal/app.py`](deploy/modal/app.py)
2. point the Next.js frontend in [`deploy/frontend`](deploy/frontend) at that Modal URL with `MODAL_API_BASE_URL`
3. paste a public or pre-signed clip URL into the dashboard's live benchmark form

The full step-by-step guide is in [`deploy/README.md`](deploy/README.md).

## CLI Commands

| Command | Description |
|---------|-------------|
| `vbench run-benchmark <path>` | Full pipeline: ingest → segment → extract → label → summarize |
| `vbench sweep <path>` | Extraction sweep: benchmark across frame counts and sampling methods |
| `vbench compare <run_a> <run_b>` | Side-by-side run comparison with deltas |
| `vbench evaluate <run_id>` | Evaluate and summarize a previous run |
| `vbench export <run_id>` | Export results to CSV/Parquet/JSON |
| `vbench inspect-run [run_id]` | List all runs or inspect a specific run |
| `vbench ingest <path>` | Ingest video(s) and extract metadata |
| `vbench segment` | Segment ingested videos into temporal windows |
| `vbench extract-frames` | Extract representative frames from segments |
| `vbench label` | Run model labeling on extracted segments |
| `vbench list-videos` | List all ingested videos |
| `vbench version` | Show version |

## Architecture

```
src/video_eval_harness/
├── cli.py              # Typer CLI with all commands
├── config.py           # YAML config loading + Pydantic settings
├── schemas.py          # Core data models (VideoMetadata, Segment, SegmentLabelResult, etc.)
├── storage.py          # SQLite storage + artifact directory management
├── caching.py          # Disk-based response cache (diskcache)
├── log.py              # Rich-powered logging
├── viewer.py           # Streamlit result viewer
├── sweep.py            # Multi-config extraction sweep orchestrator
├── adapters/           # Data source adapters
│   ├── dataset_base.py # BaseAdapter interface
│   ├── local_files.py  # Single file adapter
│   ├── directory.py    # Directory scanner adapter
│   ├── manifest.py     # CSV/JSON manifest adapter
│   └── ego4d.py        # Ego4D dataset adapter with ground truth
├── segmentation/       # Temporal segmentation strategies
│   ├── base.py         # BaseSegmenter interface
│   ├── fixed_window.py # Fixed-duration windows with optional overlap
│   └── scene_heuristic.py # Histogram-based shot boundary detection
├── extraction/         # Frame extraction
│   └── frames.py       # Uniform sampling + contact sheet generation
├── prompting/          # Prompt template system
│   └── templates.py    # Jinja2 templates (concise, rich, action_label, strict_json)
├── labeling/           # Model inference orchestration
│   ├── runner.py       # Concurrent multi-model labeling with resume
│   └── normalization.py # JSON extraction + response parsing
├── providers/          # Model provider backends
│   ├── base.py         # BaseProvider interface
│   └── openrouter.py   # OpenRouter implementation (retries, rate limits)
├── evaluation/         # Metrics and summaries
│   ├── metrics.py      # Agreement matrix, ground truth accuracy, sweep metrics
│   └── summaries.py    # Rich tables, DataFrame export, CSV/Parquet/JSON
└── utils/              # Shared utilities
    ├── ffmpeg.py       # ffprobe metadata + frame extraction
    ├── ids.py          # Deterministic ID generation
    └── time_utils.py   # Time formatting
```

## Configuration

### models.yaml

Defines which models to benchmark. All models use OpenRouter model IDs:

```yaml
models:
  gemini-3.1-pro:
    model_id: "google/gemini-3.1-pro-preview"
    provider: openrouter
    max_tokens: 2048
    temperature: 0.1
    supports_images: true

  gpt-5.4:
    model_id: "openai/gpt-5.4"
    provider: openrouter
    max_tokens: 2048
    temperature: 0.1
    supports_images: true
```

Native providers (`openai`, `gemini`) are also supported — set the corresponding API key and `provider` field. See [OpenRouter Models](https://openrouter.ai/models) for available model IDs.

### benchmark.yaml

Controls the benchmark pipeline:

```yaml
name: "default"
models:              # Which models to run (keys from models.yaml)
  - gemini-3.1-pro
  - gpt-5.4
  - qwen3.5-vl
  - claude-sonnet-4.6

prompt_version: "action_label"  # action_label | concise | rich | strict_json

segmentation:
  mode: fixed_window
  window_size_s: 10.0
  stride_s: null             # null = no overlap
  min_segment_s: 2.0

extraction:
  num_frames: 8
  method: uniform
  image_format: jpg
  image_quality: 85
  max_dimension: 1280
  generate_contact_sheet: false
```

### Prompt Templates

Five built-in templates:

- **action_label** — Constrains primary_action to concise verb phrases (max 5 words, no articles). Best for agreement across models. Default.
- **claude_action_label** — Like action_label but with stronger constraints for Claude models that tend to focus on animated sub-elements rather than the dominant visual.
- **concise** — Compact labeling prompt, narrative primary_action strings
- **rich** — Detailed egocentric video analysis prompt
- **strict_json** — Minimal, forces raw JSON output

Custom templates can be added in `configs/prompts.yaml`:

```yaml
templates:
  my_custom:
    template: |
      Analyze {{ num_frames }} frames from {{ start_time }}s to {{ end_time }}s.
      Return JSON with primary_action, description, confidence.
```

## Key Design Decisions

### Provider Abstraction

OpenRouter is the primary provider — one API key accesses GPT-4o, Gemini, Claude, Qwen, Llama, and more. The provider layer is abstracted so native providers (OpenAI, Google) can be added later without changing the pipeline.

### Frame-Based Input

The first version works by extracting frames from video segments and sending them as images. This is the common denominator across all vision-language models. Direct video input can be added per-provider as an extension path.

### Caching & Resume

- **Response cache**: API responses are cached by (model, prompt_hash, input_hash). Re-running the same benchmark skips already-cached calls.
- **Resume support**: The labeling runner checks SQLite before each request. Interrupted runs resume where they left off.
- **Frame cache**: Extracted frames are saved to disk and reused across runs.

### Structured Output Parsing

Models are prompted to return JSON. The parser handles:
- Direct JSON responses
- JSON wrapped in markdown code blocks
- JSON embedded in surrounding text
- Parse failures are recorded without crashing the run

### Reproducibility

Every run saves:
- Full config snapshot (models, prompt, segmentation, extraction)
- Raw model responses
- Parsed/normalized results
- Timestamps and run ID

## Output Schema

Each model response is normalized into `SegmentLabelResult`:

```python
{
    "run_id": "run_abc123",
    "video_id": "vid_cooking_demo_f8a2b1c3",
    "segment_id": "vid_cooking_demo_f8a2b1c3_seg0003",
    "start_time_s": 30.0,
    "end_time_s": 40.0,
    "model_name": "gemini-3.1-pro",
    "provider": "openrouter",
    "primary_action": "chopping vegetables",
    "secondary_actions": ["holding knife", "looking at cutting board"],
    "description": "Person is chopping vegetables on a wooden cutting board",
    "objects": ["knife", "cutting board", "vegetables", "bowl"],
    "environment_context": "indoor kitchen with natural lighting",
    "confidence": 0.92,
    "reasoning_summary_or_notes": "Clear chopping motion visible across frames",
    "uncertainty_flags": [],
    "parsed_success": true,
    "latency_ms": 2340.5,
    "estimated_cost": 0.0023,
    "prompt_version": "action_label"
}
```

## Evaluation Metrics

### Without Ground Truth
- Parse success rate per model
- Latency statistics (avg, median, P95)
- Cost estimation
- Pairwise primary-action agreement matrix (fuzzy matching with tiered similarity)
- Confidence distribution
- Cross-run comparison via `vbench compare`

### With Ground Truth
- Exact match rate
- Fuzzy match rate (tiered similarity >= 0.5)
- Mean similarity score
- Per-model accuracy breakdown

### Sweep Metrics
- Model x extraction variant matrix (parse rate, latency, confidence)
- Model stability across extraction variants (self-agreement, rank stability)
- Per-variant agreement matrices

## Extending the System

### Adding a New Provider

Implement `BaseProvider`:

```python
from video_eval_harness.providers.base import BaseProvider, ProviderResponse

class MyProvider(BaseProvider):
    provider_name = "my_provider"

    def complete(self, model_id, prompt, image_paths=None, max_tokens=2048, temperature=0.1):
        # Your implementation
        return ProviderResponse(text=..., model=model_id, ...)

    def list_models(self):
        return [...]
```

### Adding a Dataset Adapter

Implement `BaseAdapter`:

```python
from video_eval_harness.adapters.dataset_base import BaseAdapter, VideoEntry

class MyDatasetAdapter(BaseAdapter):
    def list_videos(self) -> list[VideoEntry]:
        # Return list of VideoEntry objects
        return [VideoEntry(path=Path("..."), metadata={...})]

    def name(self) -> str:
        return "my_dataset"
```

### Using Ego4D Data

The `Ego4DAdapter` parses the Ego4D JSON manifest and works with partial downloads:

```bash
# Run benchmark on locally-downloaded Ego4D clips
# Ground truth is auto-loaded from the manifest
vbench run-benchmark path/to/clips/ --adapter ego4d --manifest path/to/ego4d.json
```

The adapter:
1. Parses `ego4d.json` for video metadata and temporal action annotations
2. Returns `VideoEntry` objects only for clips found in the local directory
3. Extracts ground truth as `GroundTruthLabel` objects automatically
4. Skips missing videos with a warning (the full dataset is ~5TB)

## Research Note: ARC-AGI Evaluation Patterns

The spec mentioned investigating "Poetiq / ARC AGI" evaluation harness patterns. Research findings:

- **Poetiq ARC-AGI Solver** uses an iterative refinement loop pattern (generate → verify → refine) rather than single-pass evaluation
- The ARC Prize evaluation infrastructure emphasizes that harness design matters as much as model capability — identical models show ±20% variance depending on scaffolding
- Key patterns adopted: config-driven model selection, cost tracking alongside accuracy, structured JSON task format, and custom evaluation harness over general frameworks
- The EleutherAI `lm-evaluation-harness` was considered but is too text-focused for multimodal video tasks

This harness draws architectural inspiration from these patterns (modular provider abstraction, config-driven evaluation, reproducible runs with cost tracking) while being purpose-built for video understanding tasks.

## Artifacts Layout

```
artifacts/
├── vbench.db           # SQLite database (metadata, results)
├── cache/              # Disk cache for API responses
├── frames/             # Extracted frames organized by video/segment
│   └── <video_id>/
│       └── <segment_id>/
│           ├── frame_000.jpg
│           ├── frame_001.jpg
│           └── ...
└── runs/               # Per-run exports
    └── <run_id>/
        ├── <run_id>_results.csv
        ├── <run_id>_results.parquet
        └── <run_id>_results.json
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/ tests/
```

## License

MIT
