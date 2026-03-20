# CLAUDE.md — VBench Project Guide

## Project Overview

VBench is a multi-model video segmentation and labeling benchmark harness. It ingests videos, segments them, extracts frames, sends frames to multiple vision-language models via OpenRouter, and compares the results. The codebase lives in `src/video_eval_harness/`.

## Your Role

You own the **pipeline logic, evaluation, and orchestration** layers. Codex (on a separate branch) owns extraction utilities, storage, frontend, and tests. You are working on the `claude-dev` branch. Codex works on the `codex-dev` branch. Neither agent touches the other's branch.

## Architecture — Know What You Own

```
YOU OWN:                              CODEX OWNS:
├── cli.py                            ├── extraction/frames.py
├── config.py                         ├── storage.py
├── schemas.py                        ├── caching.py
├── sweep.py (new)                    ├── viewer.py
├── sweep_metrics.py (new)            ├── utils/ffmpeg.py
├── labeling/runner.py                ├── utils/ids.py
├── labeling/normalization.py         ├── utils/time_utils.py
├── prompting/templates.py            ├── tests/
├── evaluation/metrics.py             └── .github/workflows/
├── evaluation/summaries.py
├── providers/base.py
├── providers/openrouter.py
├── segmentation/base.py
├── segmentation/fixed_window.py
├── segmentation/scene_heuristic.py
└── adapters/
```

**If you need to change a file Codex owns**, describe what you need changed and I will relay it. Do not modify those files directly.

## Current Priority: Sweep Feature

We are adding multi-config benchmark sweeps. The extraction strategy (num_frames × sampling_method) becomes a benchmarked variable alongside the model. Implementation order:

1. `schemas.py` — Add sweep fields to SegmentLabelResult (P0, do first)
2. `sweep.py` — SweepConfig, SweepAxis, ExtractionVariant, SweepOrchestrator (P0)
3. `config.py` — Wire parse_sweep_config into config loading (P0)
4. `labeling/runner.py` — Accept variant param, tag results (P0, after 1-3)
5. `cli.py` — Add --sweep flag and vbench sweep command (P1)
6. `evaluation/` — Sweep-aware metrics and export (P1)
7. `caching.py` — Include variant_id in cache key (P1, coordinate with Codex)

## Code Standards

### Python
- Python 3.10+ with type hints on all function signatures
- Use `from __future__ import annotations` in every file
- Pydantic for config validation, dataclasses for internal data structures
- Async where the labeling pipeline uses it; sync is fine for config/eval
- f-strings, not .format() or %

### Naming
- Files: snake_case
- Classes: PascalCase
- Functions/methods: snake_case
- Constants: UPPER_SNAKE_CASE
- IDs: deterministic hashes using hashlib.sha256, truncated to 8-12 hex chars

### Imports
- Standard library first, blank line, third-party, blank line, local imports
- No wildcard imports
- Prefer explicit imports over module-level imports

### Error Handling
- Never silently swallow exceptions in the pipeline
- Parse failures get recorded in the result (parsed_success=False), not raised
- API failures get retried (existing retry logic in providers/openrouter.py)
- Config errors should fail fast with clear messages before any API calls

## Testing Rules

### Before Any Commit
1. Run `ruff check src/ tests/` — zero warnings
2. Run `pytest tests/` — all pass
3. If you created new modules, write at least smoke tests

### What to Test
- Config parsing: both sweep and non-sweep YAML produce correct objects
- Sweep matrix: cardinality is correct (models × variants)
- Variant IDs: deterministic and unique
- Schema changes: new fields have defaults, old data still loads
- Evaluation: metrics compute correctly with synthetic data

### What NOT to Test (Codex handles these)
- Frame extraction with real FFmpeg
- SQLite migrations
- Frontend rendering
- CI pipeline

## Branch Discipline

### Your Branch: `claude-dev`
```bash
# Before starting work each session:
git fetch origin
git rebase origin/main

# Commit frequently with descriptive messages:
git commit -m "feat(sweep): add ExtractionVariant with deterministic variant_id"
git commit -m "feat(schemas): add sweep fields to SegmentLabelResult"
git commit -m "fix(config): handle missing sweep block gracefully"

# Never force push. Never merge codex-dev yourself.
```

### Merge Criteria — ALL must be true before merging to main:
1. `ruff check` passes with zero warnings
2. `pytest` passes with zero failures
3. No import errors when running `python -c "from video_eval_harness.sweep import SweepConfig"`
4. Feature is complete enough to be useful on its own (no half-wired imports)
5. No changes to files owned by Codex

### Merge Sequence for Sweep Feature:
- **Merge 1 (schema alignment):** schemas.py changes merge first so both branches share the same data model
- **Merge 2 (orchestrator):** sweep.py + config.py after Codex has merged storage.py changes
- **Merge 3 (pipeline):** labeling/runner.py changes after extraction/frames.py is updated by Codex
- **Merge 4 (evaluation + CLI):** everything else

## Things to Never Do

- Don't add dependencies without telling me. If you need a new pip package, say so and stop.
- Don't modify `storage.py` — Codex owns the SQLite layer
- Don't create frontend files (no .jsx, .tsx, .html, .css, .svelte)
- Don't make API calls during tests (mock the OpenRouter provider)
- Don't change the existing SegmentLabelResult field names — only add new optional fields
- Don't put secrets, API keys, or .env contents in any committed file

## Useful Context

### Models (already configured in models.yaml)
- `google/gemini-3.1-pro-preview` — frontier, $2/$12 per 1M tokens
- `openai/gpt-5.4` — frontier, $2.50/$15 per 1M tokens
- `qwen/qwen3.5-397b-a17b` — frontier, ~$0.20/$1.56 per 1M tokens
- `anthropic/claude-sonnet-4.6` — frontier

### Key Data Flow
```
Video → Segments → Frames → (sent to model as images) → JSON response → parsed → SegmentLabelResult → SQLite + CSV
```

### Sweep adds this dimension:
```
Video → Segments → [for each ExtractionVariant: Frames] → [for each Model: label] → tagged results → grouped evaluation
```
