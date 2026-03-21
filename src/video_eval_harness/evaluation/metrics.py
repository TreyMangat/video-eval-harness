"""Evaluation metrics for model comparison."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .llm_judge import LLMJudge

from ..log import get_logger
from ..schemas import GroundTruthLabel, ModelRunSummary, SegmentLabelResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Sweep-aware metric containers
# ---------------------------------------------------------------------------

@dataclass
class CellMetrics:
    """Metrics for a single (model x extraction_variant) cell in the sweep matrix."""

    model_name: str
    variant_label: str
    variant_id: str
    total_segments: int = 0
    successful_parses: int = 0
    parse_success_rate: float = 0.0
    avg_latency_ms: Optional[float] = None
    median_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    avg_confidence: Optional[float] = None
    total_estimated_cost: Optional[float] = None


@dataclass
class ModelStabilityScore:
    """How stable a model's outputs are across extraction variants.

    self_agreement: fraction of (segment, variant_a, variant_b) triples where
    the model's primary_action matches across the two variants.
    rank_stability: 1.0 means the model keeps the same rank (by parse rate)
    across all variants; 0.0 means it swaps constantly.
    """

    model_name: str
    self_agreement: float = 0.0
    rank_positions: list[int] = field(default_factory=list)
    rank_stability: float = 0.0


def compute_model_summary(results: list[SegmentLabelResult], model_name: str) -> ModelRunSummary:
    """Compute summary statistics for a single model's results."""
    model_results = [r for r in results if r.model_name == model_name]

    if not model_results:
        return ModelRunSummary(model_name=model_name)

    total = len(model_results)
    successful = sum(1 for r in model_results if r.parsed_success)
    failed = total - successful

    latencies = [r.latency_ms for r in model_results if r.latency_ms is not None and r.latency_ms > 0]
    confidences = [r.confidence for r in model_results if r.confidence is not None]
    costs = [r.estimated_cost for r in model_results if r.estimated_cost is not None]

    return ModelRunSummary(
        model_name=model_name,
        total_segments=total,
        successful_parses=successful,
        failed_parses=failed,
        parse_success_rate=successful / total if total > 0 else 0.0,
        avg_latency_ms=float(np.mean(latencies)) if latencies else None,
        median_latency_ms=float(np.median(latencies)) if latencies else None,
        p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else None,
        total_estimated_cost=sum(costs) if costs else None,
        avg_confidence=float(np.mean(confidences)) if confidences else None,
    )


def compute_agreement_matrix(
    results: list[SegmentLabelResult],
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Compute pairwise primary_action agreement between models.

    Uses :func:`compute_action_similarity` for fuzzy matching.
    The returned value per pair is the mean similarity score across
    all segments where both models produced a result.

    Args:
        results: Label results to compare.
        threshold: Not used for scoring (kept for API compat). The matrix
            now reports continuous similarity rather than binary agreement.

    Returns a dict of {model_a: {model_b: mean_similarity}}.
    """
    # Group by segment
    by_segment: dict[str, dict[str, str]] = defaultdict(dict)
    for r in results:
        if r.parsed_success and r.primary_action:
            by_segment[r.segment_id][r.model_name] = r.primary_action.lower().strip()

    models = sorted({r.model_name for r in results})
    agreement: dict[str, dict[str, float]] = {}

    for m1 in models:
        agreement[m1] = {}
        for m2 in models:
            if m1 == m2:
                agreement[m1][m2] = 1.0
                continue
            sim_sum = 0.0
            total = 0
            for seg_labels in by_segment.values():
                if m1 in seg_labels and m2 in seg_labels:
                    total += 1
                    sim_sum += compute_action_similarity(seg_labels[m1], seg_labels[m2])
            agreement[m1][m2] = sim_sum / total if total > 0 else 0.0

    return agreement


def compute_ground_truth_accuracy(
    results: list[SegmentLabelResult],
    ground_truth: list[GroundTruthLabel],
) -> dict[str, dict[str, float]]:
    """Compute accuracy against ground truth labels.

    Matches by segment_id first, then falls back to video_id (for datasets
    like UCF101 where each clip has a single whole-video GT label).

    Returns {model_name: {metric: value}}.
    """
    gt_by_segment = {gt.segment_id: gt for gt in ground_truth if gt.segment_id}
    gt_by_video = {gt.video_id: gt for gt in ground_truth if gt.video_id}

    models = sorted({r.model_name for r in results})
    metrics: dict[str, dict[str, float]] = {}

    for model in models:
        model_results = [r for r in results if r.model_name == model and r.parsed_success]
        exact_match = 0
        fuzzy_match = 0
        sim_sum = 0.0
        total = 0

        for r in model_results:
            gt = gt_by_segment.get(r.segment_id) or gt_by_video.get(r.video_id)
            if gt is None:
                continue
            total += 1
            pred = _normalize_action((r.primary_action or ""))
            truth = _normalize_action(gt.primary_action)

            sim = compute_action_similarity(pred, truth)
            sim_sum += sim
            if pred == truth:
                exact_match += 1
            if sim >= 0.5:
                fuzzy_match += 1

        metrics[model] = {
            "exact_match_rate": exact_match / total if total > 0 else 0.0,
            "fuzzy_match_rate": fuzzy_match / total if total > 0 else 0.0,
            "mean_similarity": sim_sum / total if total > 0 else 0.0,
            "evaluated_segments": total,
        }

    return metrics


def compute_action_similarity(a: str, b: str) -> float:
    """Compute similarity between two action strings using tiered matching.

    Returns a float 0.0–1.0:
        Tier 1   (exact):    normalized strings identical → 1.0
        Tier 1.5 (synonym):  canonical verb match → 0.7–0.9 based on object overlap
        Tier 2   (overlap):  word-set Jaccard > 0.33 → Jaccard score
        Tier 3   (root):     head-noun overlap after stripping modifiers → 0.8
        Otherwise → 0.0
    """
    na = _normalize_action(a)
    nb = _normalize_action(b)

    # Tier 1: exact match
    if na == nb:
        return 1.0

    # Substring containment (legacy, still useful)
    if len(na) > 3 and len(nb) > 3 and (na in nb or nb in na):
        return 0.9

    # Tier 1.5: verb synonym match
    words_a_list = na.split()
    words_b_list = nb.split()
    if words_a_list and words_b_list:
        canon_a = _canonicalize_verb(words_a_list[0])
        canon_b = _canonicalize_verb(words_b_list[0])
        if canon_a == canon_b:
            # Same canonical verb — check object overlap
            obj_a = set(words_a_list[1:]) - _STOP_WORDS
            obj_b = set(words_b_list[1:]) - _STOP_WORDS
            if obj_a & obj_b:  # shared object words
                return 0.9
            elif not obj_a and not obj_b:
                return 0.85
            else:
                return 0.7

    # Tokenize
    words_a = set(na.split())
    words_b = set(nb.split())

    if not words_a or not words_b:
        return 0.0

    # Tier 2: Jaccard similarity on word sets
    intersection = words_a & words_b
    union = words_a | words_b
    jaccard = len(intersection) / len(union) if union else 0.0
    if jaccard > 0.33:
        return jaccard

    # Tier 3: extract head noun phrase (verb + nouns) and compare
    root_a = _extract_root_phrase(na)
    root_b = _extract_root_phrase(nb)
    if root_a and root_b and root_a == root_b:
        return 0.8

    # Root phrase Jaccard as fallback
    root_words_a = set(root_a.split()) if root_a else set()
    root_words_b = set(root_b.split()) if root_b else set()
    if root_words_a and root_words_b:
        root_intersection = root_words_a & root_words_b
        root_union = root_words_a | root_words_b
        root_jaccard = len(root_intersection) / len(root_union) if root_union else 0.0
        if root_jaccard > 0.5:
            return root_jaccard * 0.8  # Discount slightly vs full-word match

    return 0.0


# Words that are modifiers/articles — stripped when extracting root phrases
_STOP_WORDS = frozenset({
    "a", "an", "the", "this", "that", "some", "very", "more",
    "is", "are", "was", "were", "be", "been", "being",
    "with", "from", "into", "onto", "upon", "over", "under",
    "in", "on", "at", "to", "of", "for", "by", "about",
    "and", "or", "but", "also", "still", "just", "only",
    "its", "their", "his", "her",
})

# Verb synonym map — canonicalizes action verbs so semantically equivalent
# descriptions (e.g. "cutting vegetables" vs "chopping vegetables") score high.
_VERB_SYNONYMS: dict[str, str] = {
    "loading": "placing", "feeding": "placing", "putting": "placing",
    "inserting": "placing", "setting": "placing",
    "cutting": "slicing", "chopping": "slicing", "dicing": "slicing",
    "trimming": "slicing",
    "grabbing": "picking up", "taking": "picking up", "grasping": "picking up",
    "retrieving": "picking up",
    "walking": "moving", "running": "moving", "jogging": "moving",
    "strolling": "moving",
    "looking": "observing", "watching": "observing", "examining": "observing",
    "inspecting": "observing", "viewing": "observing",
    "talking": "speaking", "saying": "speaking", "discussing": "speaking",
    "displaying": "showing", "presenting": "showing",
    "assembling": "building", "constructing": "building",
    "stirring": "mixing", "blending": "mixing", "whisking": "mixing",
}


def _canonicalize_verb(word: str) -> str:
    """Map a verb to its canonical form, or return it unchanged."""
    return _VERB_SYNONYMS.get(word, word)


def _extract_root_phrase(s: str) -> str:
    """Extract the core action phrase by removing modifiers and articles.

    'displaying a video test pattern' → 'displaying test pattern'
    'showing colorful television test bars' → 'showing television test bars'
    """
    words = s.split()
    root = [w for w in words if w not in _STOP_WORDS]
    return " ".join(root)


def _normalized_match(a: str, b: str, threshold: float = 0.5) -> bool:
    """Check if two action strings match using similarity threshold."""
    return compute_action_similarity(a, b) >= threshold


def compute_verbosity_stats(
    results: list[SegmentLabelResult],
) -> dict[str, dict[str, float]]:
    """Compare output verbosity/richness across models.

    Returns {model_name: {metric: value}}.
    """
    models = sorted({r.model_name for r in results})
    stats: dict[str, dict[str, float]] = {}

    for model in models:
        mr = [r for r in results if r.model_name == model and r.parsed_success]
        if not mr:
            stats[model] = {}
            continue

        desc_lens = [len(r.description or "") for r in mr]
        action_lens = [len(r.primary_action or "") for r in mr]
        obj_counts = [len(r.objects) for r in mr]
        secondary_counts = [len(r.secondary_actions) for r in mr]
        uncertainty_counts = [len(r.uncertainty_flags) for r in mr]

        stats[model] = {
            "avg_description_length": float(np.mean(desc_lens)) if desc_lens else 0.0,
            "avg_action_length": float(np.mean(action_lens)) if action_lens else 0.0,
            "avg_objects_count": float(np.mean(obj_counts)) if obj_counts else 0.0,
            "avg_secondary_actions": float(np.mean(secondary_counts)) if secondary_counts else 0.0,
            "avg_uncertainty_flags": float(np.mean(uncertainty_counts)) if uncertainty_counts else 0.0,
        }

    return stats


def compute_failure_analysis(
    results: list[SegmentLabelResult],
) -> dict[str, list[dict[str, str]]]:
    """Analyze parse failures per model.

    Returns {model_name: [{segment_id, error, raw_snippet}]}.
    """
    models = sorted({r.model_name for r in results})
    failures: dict[str, list[dict[str, str]]] = {}

    for model in models:
        model_failures = []
        for r in results:
            if r.model_name == model and not r.parsed_success:
                raw_snippet = (r.raw_response_text or "")[:200]
                model_failures.append({
                    "segment_id": r.segment_id,
                    "error": r.parse_error or "Unknown error",
                    "raw_snippet": raw_snippet,
                })
        if model_failures:
            failures[model] = model_failures

    return failures


def _normalize_action(s: str) -> str:
    """Normalize an action string for comparison."""
    s = s.lower().strip()
    # Remove common prefixes
    for prefix in ["the person is ", "person is ", "the user is ", "someone is "]:
        if s.startswith(prefix):
            s = s[len(prefix):]
    # Strip articles and common filler
    s = s.strip().rstrip(".")
    return s


# ---------------------------------------------------------------------------
# LLM-as-judge metrics
# ---------------------------------------------------------------------------

def compute_agreement_matrix_llm(
    results: list[SegmentLabelResult],
    judge: "LLMJudge",
) -> dict[str, dict[str, float]]:
    """Compute agreement using LLM judge instead of string matching.

    Same interface as :func:`compute_agreement_matrix` — returns
    ``{model_a: {model_b: mean_similarity}}``.  Exploits symmetry to halve
    the number of judge calls.
    """
    by_segment: dict[str, dict[str, str]] = defaultdict(dict)
    for r in results:
        if r.parsed_success and r.primary_action:
            by_segment[r.segment_id][r.model_name] = r.primary_action.strip()

    models = sorted({r.model_name for r in results})
    agreement: dict[str, dict[str, float]] = {}

    for m1 in models:
        agreement[m1] = {}
        for m2 in models:
            if m1 == m2:
                agreement[m1][m2] = 1.0
                continue
            # Exploit symmetry
            if m2 in agreement and m1 in agreement[m2]:
                agreement[m1][m2] = agreement[m2][m1]
                continue
            sims: list[float] = []
            for seg_labels in by_segment.values():
                if m1 in seg_labels and m2 in seg_labels:
                    result = judge.judge_pair(seg_labels[m1], seg_labels[m2])
                    sims.append(result.get("similarity", 0.0))
            agreement[m1][m2] = sum(sims) / len(sims) if sims else 0.0

    return agreement


def compute_ground_truth_accuracy_llm(
    results: list[SegmentLabelResult],
    ground_truth: list[GroundTruthLabel],
    judge: "LLMJudge",
) -> dict[str, dict[str, float]]:
    """Score accuracy using LLM judge.

    Same interface as :func:`compute_ground_truth_accuracy`.
    """
    gt_by_segment = {gt.segment_id: gt for gt in ground_truth if gt.segment_id}
    gt_by_video = {gt.video_id: gt for gt in ground_truth if gt.video_id}

    models = sorted({r.model_name for r in results})
    metrics: dict[str, dict[str, float]] = {}

    for model in models:
        model_results = [r for r in results if r.model_name == model and r.parsed_success]
        correct = 0
        sim_sum = 0.0
        total = 0

        for r in model_results:
            gt = gt_by_segment.get(r.segment_id) or gt_by_video.get(r.video_id)
            if gt is None:
                continue
            total += 1
            result = judge.judge_accuracy(r.primary_action or "", gt.primary_action)
            sim_sum += result.get("similarity", 0.0)
            if result.get("correct", False):
                correct += 1

        metrics[model] = {
            "llm_accuracy": correct / total if total > 0 else 0.0,
            "llm_mean_similarity": sim_sum / total if total > 0 else 0.0,
            "evaluated_segments": total,
        }

    return metrics


# ---------------------------------------------------------------------------
# Sweep-aware metrics
# ---------------------------------------------------------------------------

def compute_sweep_metrics(
    results: list[SegmentLabelResult],
) -> dict:
    """Compute sweep-aware metrics from a full result set.

    Returns a dict with:
        cells: list[CellMetrics] — one per (model x variant) pair
        stability: list[ModelStabilityScore] — one per model
        agreement_by_variant: dict[variant_label, agreement_matrix]
    """
    # Partition results by (model, variant)
    cells: list[CellMetrics] = []
    by_model_variant: dict[tuple[str, str], list[SegmentLabelResult]] = defaultdict(list)
    for r in results:
        key = (r.model_name, r.extraction_variant_id or "default")
        by_model_variant[key].append(r)

    # Variant labels lookup
    variant_labels: dict[str, str] = {}
    for r in results:
        vid = r.extraction_variant_id or "default"
        if vid not in variant_labels:
            variant_labels[vid] = r.extraction_label or "default"

    # Compute per-cell metrics
    for (model, vid), cell_results in by_model_variant.items():
        total = len(cell_results)
        successful = sum(1 for r in cell_results if r.parsed_success)
        latencies = [r.latency_ms for r in cell_results if r.latency_ms is not None and r.latency_ms > 0]
        confidences = [r.confidence for r in cell_results if r.confidence is not None]
        costs = [r.estimated_cost for r in cell_results if r.estimated_cost is not None]

        cells.append(CellMetrics(
            model_name=model,
            variant_label=variant_labels.get(vid, vid),
            variant_id=vid,
            total_segments=total,
            successful_parses=successful,
            parse_success_rate=successful / total if total > 0 else 0.0,
            avg_latency_ms=float(np.mean(latencies)) if latencies else None,
            median_latency_ms=float(np.median(latencies)) if latencies else None,
            p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else None,
            avg_confidence=float(np.mean(confidences)) if confidences else None,
            total_estimated_cost=sum(costs) if costs else None,
        ))

    # Model stability: self-agreement across variants for same segments
    models = sorted({r.model_name for r in results})
    variant_ids = sorted(variant_labels.keys())
    stability: list[ModelStabilityScore] = []

    for model in models:
        # Group by (segment_id, variant_id) -> primary_action
        seg_var_action: dict[tuple[str, str], str] = {}
        for r in results:
            if r.model_name == model and r.parsed_success and r.primary_action:
                vid = r.extraction_variant_id or "default"
                seg_var_action[(r.segment_id, vid)] = _normalize_action(r.primary_action)

        # Self-agreement: for each segment, mean similarity across variant pairs
        segments = sorted({s for s, _ in seg_var_action.keys()})
        sim_sum = 0.0
        total_pairs = 0
        for seg in segments:
            actions = [(v, seg_var_action.get((seg, v))) for v in variant_ids]
            actions = [(v, a) for v, a in actions if a is not None]
            for i in range(len(actions)):
                for j in range(i + 1, len(actions)):
                    total_pairs += 1
                    sim_sum += compute_action_similarity(actions[i][1], actions[j][1])

        self_agreement = sim_sum / total_pairs if total_pairs > 0 else 0.0

        # Rank stability: rank models by parse_success_rate per variant, check consistency
        rank_positions = []
        for vid in variant_ids:
            # Rank all models in this variant by parse rate
            variant_rates = []
            for m in models:
                cell_r = by_model_variant.get((m, vid), [])
                if cell_r:
                    rate = sum(1 for r in cell_r if r.parsed_success) / len(cell_r)
                else:
                    rate = 0.0
                variant_rates.append((m, rate))
            variant_rates.sort(key=lambda x: x[1], reverse=True)
            rank = next(i for i, (m, _) in enumerate(variant_rates, 1) if m == model)
            rank_positions.append(rank)

        # Rank stability: 1 - normalized std of ranks (1.0 = perfectly stable)
        if len(rank_positions) > 1:
            max_rank = len(models)
            rank_std = float(np.std(rank_positions))
            rank_stability = max(0.0, 1.0 - rank_std / max(max_rank - 1, 1))
        else:
            rank_stability = 1.0

        stability.append(ModelStabilityScore(
            model_name=model,
            self_agreement=self_agreement,
            rank_positions=rank_positions,
            rank_stability=rank_stability,
        ))

    # Pairwise agreement per variant
    agreement_by_variant: dict[str, dict[str, dict[str, float]]] = {}
    for vid in variant_ids:
        variant_results = [r for r in results if (r.extraction_variant_id or "default") == vid]
        if variant_results:
            agreement_by_variant[variant_labels.get(vid, vid)] = compute_agreement_matrix(variant_results)

    return {
        "cells": cells,
        "stability": stability,
        "agreement_by_variant": agreement_by_variant,
    }
