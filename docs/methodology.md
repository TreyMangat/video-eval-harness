# VBench methodology

## The comparison is apples-to-apples

Every model receives:
- Identical frames (same video, same segment, same extraction parameters)
- Identical prompt (same Jinja2 template with the same context variables)
- Identical output constraints (strict JSON schema)

This is not a pipeline. VBench does not chain model outputs together. The point is to measure how different models interpret the same input, not to build the best possible classifier.

## Segmentation

Videos are split into fixed-duration windows (default 10 seconds for coarse mode). Auto-windowing scales the window size with video duration so a 30-second clip and a 30-minute clip both produce manageable segment counts.

## Frame extraction

Default: 8 frames per segment, uniformly spaced. Alternative: keyframe-based sampling via OpenCV. Frames are cached per extraction variant so sweep runs don't duplicate work.

## Prompts

Every labeling run uses one of 7 Jinja2 templates. The default is `action_label_optimized`, which was selected after testing all 7 templates against UCF101 ground truth. It produces the highest mean similarity across all 10 models tested (+21% improvement over the baseline `concise` template).

Key rule: `primary_action` must be a concise verb phrase (max 5 words) so that string-based agreement matching is meaningful.

## Agreement metric

Pairwise agreement between models is computed via fuzzy string matching on the `primary_action` field. Each pair of models gets a mean similarity score across all segments where both models successfully parsed a response.

An LLM judge is an optional secondary metric: a third model (default `gpt-5.4-mini`) is asked whether two responses refer to the same action. LLM judge rankings are stable across judge models (verified by running the same data through `gpt-5.4`, `gpt-5.4-mini`, and `gemini-3-flash` as judges — identical rankings in all three).

## Accuracy metric

When ground-truth labels are available (UCF101, EPIC-KITCHENS, Ego4D manifests), VBench computes:

- **Exact match rate** — the model's `primary_action` exactly matches the ground truth after normalization
- **Fuzzy match rate** — similarity score above a threshold (default 0.5)
- **Mean similarity** — average similarity score across all evaluated segments

Fuzzy and exact match diverge the most on easy benchmarks where many models land close to 100%. Exact-match is the discriminating metric in that regime.

## Stability metric

Sweep mode varies extraction parameters (frame count, sampling method) and measures how model rankings shift. A model with `rank_stability = 1.0` keeps the same relative rank across all variants; a model with `rank_stability = 0.0` swaps rank on every variant change.

This is how VBench answers: "is this model's advantage real, or an artifact of a lucky extraction setting?"

## Cost tracking

OpenRouter returns cost in the `usage.total_cost` field on every completion. VBench uses this directly rather than estimating from tokens and rate cards. Cost tracking is per-request, aggregated per-model, per-run.

## What VBench does NOT measure

- **Human preference** — no Elo-style pairwise human ranking
- **Reasoning quality** — `reasoning_summary_or_notes` is captured but not scored
- **Calibration** — `confidence` values are captured but not evaluated against correctness
- **Video understanding beyond labeling** — no VQA, no temporal grounding, no scene graph extraction

These are intentional limitations. VBench is a labeling benchmark, not a general-purpose video understanding evaluation suite.
