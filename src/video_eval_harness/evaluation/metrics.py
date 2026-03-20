"""Evaluation metrics for model comparison."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from ..log import get_logger
from ..schemas import GroundTruthLabel, ModelRunSummary, SegmentLabelResult

logger = get_logger(__name__)


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
) -> dict[str, dict[str, float]]:
    """Compute pairwise primary_action agreement between models.

    Returns a dict of {model_a: {model_b: agreement_rate}}.
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
            matches = 0
            total = 0
            for seg_labels in by_segment.values():
                if m1 in seg_labels and m2 in seg_labels:
                    total += 1
                    if _normalized_match(seg_labels[m1], seg_labels[m2]):
                        matches += 1
            agreement[m1][m2] = matches / total if total > 0 else 0.0

    return agreement


def compute_ground_truth_accuracy(
    results: list[SegmentLabelResult],
    ground_truth: list[GroundTruthLabel],
) -> dict[str, dict[str, float]]:
    """Compute accuracy against ground truth labels.

    Returns {model_name: {metric: value}}.
    """
    gt_map = {gt.segment_id: gt for gt in ground_truth}
    models = sorted({r.model_name for r in results})
    metrics: dict[str, dict[str, float]] = {}

    for model in models:
        model_results = [r for r in results if r.model_name == model and r.parsed_success]
        exact_match = 0
        normalized_match = 0
        total = 0

        for r in model_results:
            gt = gt_map.get(r.segment_id)
            if gt is None:
                continue
            total += 1
            pred = (r.primary_action or "").lower().strip()
            truth = gt.primary_action.lower().strip()

            if pred == truth:
                exact_match += 1
            if _normalized_match(pred, truth):
                normalized_match += 1

        metrics[model] = {
            "exact_match_rate": exact_match / total if total > 0 else 0.0,
            "normalized_match_rate": normalized_match / total if total > 0 else 0.0,
            "evaluated_segments": total,
        }

    return metrics


def _normalized_match(a: str, b: str) -> bool:
    """Check if two action strings match after normalization."""
    a = _normalize_action(a)
    b = _normalize_action(b)
    if a == b:
        return True
    # Check substring containment for near-matches
    if len(a) > 3 and len(b) > 3:
        if a in b or b in a:
            return True
    return False


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
