# VBench — Video Model Benchmark Harness

[![CI](https://github.com/TreyMangat/video-eval-harness/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/TreyMangat/video-eval-harness/actions/workflows/ci.yml)

Compare how frontier vision-language models interpret the same video content. Drop in clips, pick models, see who agrees — and who doesn't.

## Quick start

Prerequisites: Python 3.10+, FFmpeg on PATH, an [OpenRouter](https://openrouter.ai/) API key.

```bash
pip install -e ".[dev]"
cp .env.example .env   # add your OPENROUTER_API_KEY
vbench test-suite       # benchmarks test_videos/ with 3 fast models
```

View results:

```bash
streamlit run src/video_eval_harness/viewer.py   # Streamlit on http://localhost:8501
```

## What it measures

- **Agreement**: do models label the same action for the same video segment?
- **Stability**: does changing frame count or sampling method change the ranking?
- **Cost**: what does each model cost per segment?
- **Latency**: how fast does each model respond?
- **Confidence**: how sure is each model about its answer?

## Models (March 2026)

| Model | Tier | OpenRouter ID |
|-------|------|---------------|
| Gemini 3.1 Pro | Frontier | `google/gemini-3.1-pro-preview` |
| GPT-5.4 | Frontier | `openai/gpt-5.4` |
| Qwen 3.5-397B | Frontier | `qwen/qwen3.5-397b-a17b` |
| Llama 4 Maverick | Frontier | `meta-llama/llama-4-maverick` |
| Gemini 3 Flash | Fast | `google/gemini-3-flash-preview` |
| GPT-5.4 Mini | Fast | `openai/gpt-5.4-mini` |
| Qwen 3.5-27B | Fast | `qwen/qwen3.5-27b` |

Edit `configs/models.yaml` to add or remove models. All use OpenRouter by default; native OpenAI and Gemini providers are also supported.

## Sweep mode

Compare models across extraction variants (frame count x sampling method):

```bash
# Preview the sweep plan without making API calls
vbench sweep test_videos/ --config configs/benchmark_fast.yaml --frames 4,8 --methods uniform --dry-run

# Run it
vbench sweep test_videos/ --config configs/benchmark_fast.yaml --frames 4,8 --methods uniform

# Compare two runs side-by-side
vbench compare run_20260321_cooking_30s_a7f4 run_20260321_6videos_sweep_b8e2
```

## CLI commands

| Command | Description |
|---------|-------------|
| `vbench test-suite` | Recommended way to run benchmarks (fast models, 2 variants, max 25 segments) |
| `vbench run-benchmark <path>` | Full pipeline: ingest -> segment -> extract -> label -> summarize |
| `vbench sweep <path>` | Extraction sweep: benchmark across frame counts and sampling methods |
| `vbench estimate <path>` | Preview API calls, cost, and time without running anything |
| `vbench compare <run_a> <run_b>` | Side-by-side run comparison with deltas |
| `vbench evaluate <run_id>` | Evaluate and summarize a previous run |
| `vbench export <run_id>` | Export results to CSV/Parquet/JSON |
| `vbench export-sweep-summary <run_id>` | Export pre-computed sweep metrics as JSON for dashboards |
| `vbench inspect-run [run_id]` | List all runs or inspect a specific run |
| `vbench download-dataset buildai-10k` | Download a Build.ai Egocentric-10K shard from Hugging Face |
| `vbench list-videos` | List ingested videos or scan a dataset adapter's directory |
| `vbench version` | Show version |

## Configuration

### models.yaml

```yaml
models:
  gemini-3.1-pro:
    model_id: "google/gemini-3.1-pro-preview"
    provider: openrouter
    max_tokens: 2048
    temperature: 0.1
    supports_images: true
    tier: frontier

  llama-4-maverick:
    model_id: "meta-llama/llama-4-maverick"
    provider: openrouter
    max_tokens: 2048
    temperature: 0.1
    supports_images: true
    tier: frontier
```

### benchmark.yaml

```yaml
name: "default"
models:
  - gemini-3.1-pro
  - gpt-5.4
  - qwen3.5-vl
  - llama-4-maverick

prompt_version: "action_label"

segmentation:
  mode: fixed_window
  window_size_s: 10.0

extraction:
  num_frames: 8
  method: uniform
```

### Prompt templates

| Template | Description |
|----------|-------------|
| `action_label` | Default. Constrains primary_action to concise verb phrases (max 5 words). |
| `action_label_v2` | Generalized labels ("fighting" not "fighting with spear"). Better cross-model agreement. |
| `concise` | Narrative primary_action strings. Lower agreement but more detail. |
| `rich` | Detailed egocentric video analysis prompt. |
| `strict_json` | Minimal, forces raw JSON output. |

## Build.ai Egocentric-10K

```bash
# Download one worker's shard (~50-100 egocentric videos)
vbench download-dataset buildai-10k --output data/egocentric-10k

# List what's in it
vbench list-videos --adapter buildai --data-dir data/egocentric-10k

# Run a sweep with auto-windowing
vbench sweep data/egocentric-10k --adapter buildai --data-dir data/egocentric-10k \
    --config configs/benchmark_fast.yaml --max-segments 20
```

The adapter auto-extracts tar shards, reads paired JSON metadata sidecars, and supports factory/worker filtering.

## Ego4D

```bash
# Ground truth is auto-loaded from the manifest
vbench run-benchmark path/to/clips/ --adapter ego4d --manifest path/to/ego4d.json
```

## Cost protection

Three layers prevent accidental expensive runs:

1. **Auto-windowing**: window size scales with video duration (10s for <60s clips, 60s for 5-30min, 120s for >30min)
2. **`--max-segments`**: hard cap on segment count with uniform subsampling
3. **Budget guard**: confirmation prompt when API calls exceed 500

Preview before committing:

```bash
vbench estimate test_videos/ --sweep --frames 4,8 --methods uniform
```

## Architecture

```
src/video_eval_harness/
├── cli.py                  # Typer CLI (16 commands)
├── config.py               # YAML config loading, Pydantic settings
├── schemas.py              # Core data models
├── sweep.py                # Multi-config extraction sweep orchestrator
├── storage.py              # SQLite storage + artifact directory
├── caching.py              # Disk-based response cache
├── log.py                  # Rich-powered logging
├── viewer.py               # Streamlit result viewer
├── adapters/               # Data source adapters
│   ├── dataset_base.py     # BaseAdapter interface + VideoEntry
│   ├── local_files.py      # Single file adapter
│   ├── directory.py        # Directory scanner
│   ├── manifest.py         # CSV/JSON manifest adapter
│   ├── ego4d.py            # Ego4D dataset with ground truth
│   └── build_ai.py         # Build.ai Egocentric-10K (WebDataset tars)
├── segmentation/           # Temporal segmentation
│   ├── fixed_window.py     # Fixed-duration windows with optional overlap
│   └── scene_heuristic.py  # Histogram-based shot boundary detection
├── extraction/             # Frame extraction
│   └── frames.py           # Uniform/keyframe sampling + contact sheets
├── prompting/              # Prompt template system
│   └── templates.py        # 6 Jinja2 templates + vocabulary injection
├── labeling/               # Model inference orchestration
│   ├── runner.py           # Concurrent multi-model labeling with resume
│   └── normalization.py    # JSON extraction + truncated response repair
├── providers/              # Model provider backends
│   ├── base.py             # BaseProvider interface
│   ├── openrouter.py       # OpenRouter (retries, rate limits)
│   ├── openai_native.py    # Native OpenAI API
│   └── gemini_native.py    # Native Google Gemini API
├── evaluation/             # Metrics and summaries
│   ├── metrics.py          # Agreement, ground truth, sweep metrics
│   └── summaries.py        # Rich tables, DataFrame export, CSV/Parquet/JSON
└── utils/
    ├── ffmpeg.py           # ffprobe metadata + frame extraction
    ├── ids.py              # Human-readable run/video/segment ID generation
    └── time_utils.py       # Time formatting
```

## Development

```bash
pip install -e ".[dev]"

# Tests
py -3.12 -m pytest -q             # 117 tests

# Lint
py -3.12 -m ruff check src/ tests/

# Streamlit viewer
streamlit run src/video_eval_harness/viewer.py
```

## License

MIT
