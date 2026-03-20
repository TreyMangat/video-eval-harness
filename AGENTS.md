# AGENTS.md — VBench Project Guide

## Project Overview

VBench is a multi-model video segmentation and labeling benchmark harness. It ingests videos, segments them, extracts frames, sends frames to multiple vision-language models via OpenRouter, and compares the results. The codebase lives in `src/video_eval_harness/`.

## Your Role

You own the **extraction utilities, storage, frontend, tests, and CI** layers. Claude Code (on a separate branch) owns pipeline orchestration, evaluation logic, schemas, providers, and CLI. You are working on the `codex-dev` branch. Claude Code works on the `claude-dev` branch. Neither agent touches the other's branch.

## Architecture — Know What You Own

```
YOU OWN:                              CLAUDE CODE OWNS:
├── extraction/frames.py              ├── cli.py
├── storage.py                        ├── config.py
├── caching.py                        ├── schemas.py
├── viewer.py                         ├── sweep.py (new)
├── utils/ffmpeg.py                   ├── sweep_metrics.py (new)
├── utils/ids.py                      ├── labeling/runner.py
├── utils/time_utils.py               ├── labeling/normalization.py
├── tests/                            ├── prompting/templates.py
├── .github/workflows/                ├── evaluation/metrics.py
└── deployment/                       ├── evaluation/summaries.py
                                      ├── providers/
                                      ├── segmentation/
                                      └── adapters/
```

**If you need to change a file Claude Code owns**, describe what you need changed and I will relay it. Do not modify those files directly.

## Current Priority: Sweep Support + Deployment

Two parallel tracks:

### Track 1: Sweep Infrastructure (support Claude Code's sweep feature)

1. `storage.py` — Add sweep columns to SQLite schema (P0, do first)
2. `extraction/frames.py` — Accept ExtractionVariant params, cache per variant (P0)
3. `tests/` — Sweep config parsing, metrics, extraction variant tests (P1)
4. `viewer.py` — Integrate sweep dashboard visualization (P1)
5. Benchmark config validator (P1)

### Track 2: Deployment (your independent track)

Continue deployment work as planned. This does not block or get blocked by the sweep feature.

## Code Standards

### Python

- Python 3.10+ with type hints on all function signatures
- Use `from __future__ import annotations` in every file
- Pydantic for config validation, dataclasses for internal data structures
- f-strings, not .format() or %

### Naming

- Files: snake_case
- Classes: PascalCase
- Functions/methods: snake_case
- Constants: UPPER_SNAKE_CASE

### Testing

- pytest for all tests
- Tests go in `tests/` directory mirroring `src/` structure
- Name test files `test_<module>.py`
- Name test functions `test_<behavior_being_tested>()`
- Use fixtures for shared setup
- Mock external dependencies (OpenRouter API, FFmpeg subprocess calls)
- Never make real API calls in tests — use `@pytest.mark.integration` for those and skip in CI

### Frontend

- If Streamlit: keep viewer.py self-contained, use st.cache_data for data loading
- If React: single-file components, Tailwind for styling, no separate CSS files
- Dashboard must load data from `artifacts/runs/<run_id>/` or `<sweep_id>/` exports

## SQLite Schema Changes for Sweep

Add these columns to `segment_label_results`. Use migration pattern (check before adding):

```sql
-- Check if columns exist before adding
ALTER TABLE segment_label_results ADD COLUMN extraction_variant_id TEXT DEFAULT '';
ALTER TABLE segment_label_results ADD COLUMN extraction_label TEXT DEFAULT '';
ALTER TABLE segment_label_results ADD COLUMN num_frames_used INTEGER DEFAULT 0;
ALTER TABLE segment_label_results ADD COLUMN sampling_method_used TEXT DEFAULT '';
ALTER TABLE segment_label_results ADD COLUMN sweep_id TEXT DEFAULT '';
```

Add composite index:

```sql
CREATE INDEX IF NOT EXISTS idx_sweep_results
ON segment_label_results(sweep_id, model_name, extraction_variant_id);
```

Add query method:

```python
def get_results_by_sweep(self, sweep_id: str) -> list[dict]:
    """Return all results for a sweep, ordered by model then variant."""
```

## Frame Extraction Changes for Sweep

`extraction/frames.py` needs to:

- Accept `num_frames` and `method` as parameters (or an ExtractionVariant-like object)
- Cache frames per variant: `artifacts/frames/<video_id>/<variant_id>/<segment_id>/`
- Return cached frames if they exist for that variant
- The `keyframe` method should use histogram-based shot boundary detection to pick visually distinct frames
- The `uniform` method samples evenly across the segment duration (existing behavior)

## Testing Rules

### Before Any Commit

1. Run `ruff check src/ tests/` — zero warnings
2. Run `pytest tests/` — all pass
3. New code must have corresponding tests

### Tests to Write for Sweep

```python
# tests/test_sweep.py
test_sweep_config_parsing_with_sweep_block()
test_sweep_config_parsing_without_sweep_block()
test_sweep_matrix_cardinality()           # 3 × 2 × 4 = 24
test_extraction_variant_id_deterministic()
test_extraction_variant_id_unique()

# tests/test_sweep_metrics.py
test_cell_metrics_parse_success_rate()
test_stability_perfect_agreement()
test_stability_no_agreement()
test_agreement_matrix_per_variant()
test_variant_impact_sensitive_model()

# tests/test_extraction_variants.py
test_uniform_4_frames()
test_uniform_16_frames()
test_keyframe_extraction()
test_frame_cache_per_variant()

# tests/test_storage_sweep.py
test_sweep_columns_migration()
test_insert_sweep_result()
test_get_results_by_sweep()
test_backwards_compatible_non_sweep()
```

## Branch Discipline

### Your Branch: `codex-dev`

```bash
# Before starting work:
git fetch origin
git rebase origin/main

# Commit messages:
git commit -m "feat(storage): add sweep columns to SQLite schema"
git commit -m "feat(extraction): support variant-aware frame caching"
git commit -m "test(sweep): add sweep config and metrics test suite"

# Never force push. Never merge claude-dev yourself.
```

### Merge Criteria — ALL must be true:

1. `ruff check` passes clean
2. `pytest` passes with zero failures
3. No import errors on any module you changed
4. Feature is self-contained (no dangling imports to unwritten code)
5. No changes to files owned by Claude Code

### Merge Sequence:

- **Merge 1:** storage.py schema changes — merge early so Claude Code's runner can write to new columns
- **Merge 2:** extraction/frames.py variant support — merge after schema is in main
- **Merge 3:** tests + viewer — merge after sweep orchestrator is in main
- Always rebase on main before merging. Resolve conflicts in your files only.

## Things to Never Do

- Don't add pip dependencies without telling me first
- Don't modify `schemas.py`, `config.py`, `cli.py`, or anything in `labeling/`, `evaluation/`, `providers/`, `segmentation/`, `adapters/`, or `prompting/`
- Don't define new Pydantic models — Claude Code owns data modeling. Use the types defined in schemas.py
- Don't make real API calls in tests
- Don't put secrets, API keys, or .env contents in any committed file
- Don't create migration scripts that drop or rename existing columns

## Useful Context

### Key Interfaces You Consume

From `schemas.py` (Claude Code owns, you read):

```python
@dataclass
class SegmentLabelResult:
    run_id: str
    video_id: str
    segment_id: str
    model_name: str
    primary_action: str
    parsed_success: bool
    latency_ms: float
    estimated_cost: float
    confidence: float
    # NEW sweep fields (optional, default empty):
    extraction_variant_id: str = ""
    extraction_label: str = ""
    num_frames_used: int = 0
    sampling_method_used: str = ""
    sweep_id: str = ""
```

From `sweep.py` (Claude Code owns, you may import):

```python
@dataclass(frozen=True)
class ExtractionVariant:
    num_frames: int
    method: SamplingMethod  # "uniform" | "keyframe"
    variant_id: str         # deterministic hash property
    label: str              # e.g. "uniform_8f" property
```

### Data Flow

```
Video → Segments → Frames → Model API → JSON → SegmentLabelResult → SQLite + CSV
                     ↑
              You own this step
              (extraction/frames.py)
```
