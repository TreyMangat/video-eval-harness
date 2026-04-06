# VBench

> **Multi-model video benchmark harness.** Compare how 10 frontier AI models interpret the same video content — agreement, accuracy, cost, and latency.

[![CI](https://github.com/TreyMangat/video-eval-harness/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/TreyMangat/video-eval-harness/actions/workflows/ci.yml)

**[Live dashboard](https://video-eval-harness-qu4m.vercel.app)**

![VBench dashboard](docs/screenshot-dashboard.png)

## What this is

I benchmarked 10 frontier vision-language models on real-world video clips to answer one question: **do frontier AI models actually agree on what's happening in a video, and which is worth paying for?**

The harness feeds identical frames and prompts to every model in parallel, then measures:

- **Agreement** — do models label the same action for the same segment?
- **Accuracy** — fuzzy and exact match against ground-truth labels on UCF101 and EPIC-KITCHENS
- **Cost** — per segment, per 1000 segments
- **Latency** — end-to-end response time
- **Stability** — how rankings shift when extraction parameters change

## Key findings

On 3 benchmark runs totaling 32 videos and 420+ model responses:

**Gemini 3.1 Pro is the most consistent frontier model.** 94% fuzzy match on UCF101 (tied for top), 70% on EPIC-KITCHENS (outright winner). Highest exact-match rate on both benchmarks. Expensive on some runs, but it's the only model that holds up across easy and hard video types.

**Rankings flip dramatically between easy and hard benchmarks.** Llama 4 Maverick ties for #1 on UCF101 at 100% fuzzy match — then collapses to 45% on EPIC-KITCHENS, the 2nd-worst result of any model tested. That's an overfitting signal: the model performs well on clean third-person action videos but falls apart on egocentric kitchen footage.

**Qwen 3.5-9B is the best budget choice.** At $0.005 per run with 90% UCF101 and 60% EPIC-KITCHENS accuracy, it beats models that cost 20x more. If you're cost-sensitive and don't need peak accuracy, this is the pick.

**Model agreement stays high but accuracy spreads.** On UCF101, accuracy is compressed between 88-100% and models broadly agree. On EPIC-KITCHENS, accuracy spreads from 25% (grok-4.1-fast) to 70% (gemini-3.1-pro / qwen3.5-122b) and agreement drops. Models agree on easy cases and disagree on hard ones.

**Exact-match is the more discriminating metric.** On UCF101 fuzzy match, 6 models tie at 94%. On exact-match, only `gemini-3-flash` and `gemini-3.1-pro` hit 88%. When you need to pick a model on easy data, look at exact-match, not fuzzy.

### UCF101 Full Benchmark

10 models, 12 videos, 16 segments, $0.91 total cost.

| Model | Fuzzy match | Exact match | Cost | Latency |
|-------|------------|------------|------|---------|
| llama-4-maverick | 100% | 81% | $0.010 | 6.3s |
| gemini-3-flash | 100% | 88% | $0.081 | 3.2s |
| gpt-5.4-mini | 94% | 62% | $0.025 | 1.1s |
| qwen3.5-27b | 94% | 81% | $0.050 | 19.6s |
| qwen3.5-122b-a10b | 94% | 81% | $0.066 | 12.0s |
| gpt-5.4 | 94% | 81% | $0.094 | 3.2s |
| gemini-3.1-pro | 94% | 88% | $0.459 | 15.4s |
| qwen3.5-vl | 93% | 80% | $0.109 | 23.7s |
| qwen3.5-9b | 90% | 70% | $0.005 | 12.5s |
| grok-4.1-fast | 88% | 75% | $0.012 | 4.4s |

### EPIC-KITCHENS Full Benchmark

10 models, 20 videos, 20 segments, $0.33 total cost.

| Model | Fuzzy match | Exact match | Cost | Latency |
|-------|------------|------------|------|---------|
| gemini-3.1-pro | 70% | 35% | $0.031 | 15.1s |
| qwen3.5-122b-a10b | 70% | 20% | $0.186 | 21.0s |
| gemini-3-flash | 65% | 15% | — | — |
| qwen3.5-vl | 61% | 22% | $0.057 | 26.4s |
| gpt-5.4 | 60% | 20% | — | — |
| gpt-5.4-mini | 60% | 30% | — | — |
| qwen3.5-9b | 60% | 40% | $0.013 | 16.4s |
| qwen3.5-27b | 53% | 16% | $0.017 | 51.8s |
| llama-4-maverick | 45% | 10% | — | — |
| grok-4.1-fast | 25% | 0% | $0.021 | 10.2s |

See the [methodology](docs/methodology.md) for how these numbers are computed, or explore the [live dashboard](https://video-eval-harness-qu4m.vercel.app) to see every model response per segment.

## Featured runs

- [UCF101 Full Benchmark](https://video-eval-harness-qu4m.vercel.app/report/run_20260322_ucf101-full-benchmark_95e2) — 10 models, 12 videos, 16 segments, $0.91 total cost
- [EPIC-KITCHENS Full Benchmark](https://video-eval-harness-qu4m.vercel.app/report/run_20260322_epic-kitchens-full-benchmark_6872) — 10 models, 20 videos, 20 segments, $0.33 total cost
- [UCF101 Fast Models](https://video-eval-harness-qu4m.vercel.app/report/run_20260323_ucf101-fast-models_acb2) — 4 fast models, same UCF101 videos

## Architecture overview

```
Video → Ingest → Segment → Extract frames → Label (N models in parallel) → Evaluate → Export
                                                ↓
                                    OpenRouter / OpenAI / Gemini
```

- **Backend:** Python 3.12, Typer CLI, FastAPI, concurrent model labeling via ThreadPoolExecutor
- **Storage:** MongoDB Atlas for metadata, filesystem for frames/videos, diskcache for response cache
- **Inference:** OpenRouter as the primary provider (10 models via one key), with native OpenAI and Gemini provider fallbacks
- **API:** FastAPI deployed on Modal with autoscaling serverless containers
- **Frontend:** Next.js 15 + React 19 + Recharts, deployed on Vercel with dynamic Open Graph images per run
- **Tests:** 135+ pytest tests, ruff for linting, GitHub Actions CI

~9,400 lines of Python + ~15,400 lines of TypeScript.

See [`docs/architecture.md`](docs/architecture.md) for a detailed breakdown.

---

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
├── storage.py              # SQLite storage (local fallback)
├── mongo_storage.py        # MongoDB storage (production)
├── storage_factory.py      # Auto-selects storage backend
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
python3 -m pytest -q             # 135+ tests

# Lint
python3 -m ruff check src/ tests/

# Streamlit viewer
streamlit run src/video_eval_harness/viewer.py
```

## License

MIT
