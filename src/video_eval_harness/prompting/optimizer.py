"""Prompt template optimization via accuracy-driven iteration."""

from __future__ import annotations

from typing import Optional

from ..log import get_logger

logger = get_logger(__name__)

META_PROMPT = """\
You are optimizing a prompt template for video action labeling.
A VLM receives frames from a video segment and must output JSON with a
"primary_action" field: a concise gerund verb phrase (max 5 words, lowercase).

Current template (being sent to the VLM):
---
{current_template}
---

Current accuracy against ground truth: {accuracy:.0%}
Common errors (model output -> ground truth):
{error_examples}

Generate an IMPROVED prompt template that addresses these errors.
The template MUST:
1. Use Jinja2 syntax with variables: {{{{ num_frames }}}}, {{{{ start_time }}}}, {{{{ end_time }}}}, {{{{ duration }}}}
2. Instruct the model to return ONLY a JSON object
3. Include "primary_action" as a required field
4. Keep the same JSON schema (primary_action, secondary_actions, description, objects, environment_context, confidence, reasoning_summary_or_notes, uncertainty_flags)

Return ONLY the template text, no commentary.
"""


class PromptOptimizer:
    """Iterate on prompt templates using accuracy as the objective."""

    def __init__(
        self,
        base_template: str,
        test_results: list,
        ground_truth: list,
        provider: object,
        meta_model: str = "openai/gpt-5.4-mini",
    ):
        self.base_template = base_template
        self.test_results = test_results
        self.ground_truth = ground_truth
        self.provider = provider
        self.meta_model = meta_model
        self.history: list[dict] = []

    def analyze_errors(
        self,
        results: list,
        gt_labels: list,
    ) -> list[dict]:
        """Compare results against ground truth and categorize errors."""
        from ..evaluation.metrics import _normalize_action, compute_action_similarity

        gt_by_video = {gt.video_id: gt for gt in gt_labels if gt.video_id}
        gt_by_segment = {gt.segment_id: gt for gt in gt_labels if gt.segment_id}

        errors = []
        for r in results:
            if not r.parsed_success or not r.primary_action:
                continue
            gt = gt_by_segment.get(r.segment_id) or gt_by_video.get(r.video_id)
            if gt is None:
                continue

            pred = _normalize_action(r.primary_action)
            truth = _normalize_action(gt.primary_action)
            sim = compute_action_similarity(pred, truth)

            if sim < 0.8:
                error_type = "wrong_verb"
                pred_words = pred.split()
                truth_words = truth.split()
                if pred_words and truth_words and pred_words[0] == truth_words[0]:
                    error_type = "wrong_object"
                elif len(pred.split()) > 5:
                    error_type = "too_detailed"
                elif len(pred.split()) <= 1:
                    error_type = "too_vague"

                errors.append({
                    "segment_id": r.segment_id,
                    "model": r.model_name,
                    "predicted": pred,
                    "ground_truth": truth,
                    "similarity": sim,
                    "error_type": error_type,
                })

        return errors

    def generate_variant(self, current_template: str, accuracy: float, errors: list[dict]) -> str:
        """Use an LLM to generate an improved prompt template variant."""
        error_lines = []
        for e in errors[:10]:
            error_lines.append(f'  "{e["predicted"]}" -> should be "{e["ground_truth"]}" ({e["error_type"]})')
        error_str = "\n".join(error_lines) if error_lines else "  (no specific errors)"

        meta_prompt = META_PROMPT.format(
            current_template=current_template,
            accuracy=accuracy,
            error_examples=error_str,
        )

        response = self.provider.complete(
            model_id=self.meta_model,
            prompt=meta_prompt,
            max_tokens=2048,
            temperature=0.7,
        )

        if not response.success:
            logger.warning(f"Meta-prompt failed: {response.error}")
            return current_template

        return response.text.strip()

    def run_loop(
        self,
        iterations: int = 3,
        evaluate_fn: Optional[object] = None,
    ) -> dict:
        """Main optimization loop.

        Args:
            iterations: Number of improvement iterations.
            evaluate_fn: Optional callable(template_str) -> {"accuracy": float, "results": list}.
                If not provided, returns after generating variants without evaluation.

        Returns:
            {"best_template": str, "best_accuracy": float, "history": list}
        """
        current_template = self.base_template
        best_accuracy = 0.0

        # Analyze errors from existing results
        errors = self.analyze_errors(self.test_results, self.ground_truth)
        if not errors:
            logger.info("No errors found — current template may be optimal")
            return {
                "best_template": current_template,
                "best_accuracy": 1.0,
                "history": [],
            }

        best_accuracy = 1.0 - (len(errors) / max(len(self.test_results), 1))
        logger.info(f"Starting accuracy: ~{best_accuracy:.0%} ({len(errors)} errors)")

        for i in range(iterations):
            logger.info(f"Iteration {i + 1}/{iterations}")

            variant = self.generate_variant(current_template, best_accuracy, errors)

            self.history.append({
                "iteration": i + 1,
                "accuracy_before": best_accuracy,
                "errors_analyzed": len(errors),
                "variant_preview": variant[:200],
            })

            if evaluate_fn is not None:
                result = evaluate_fn(variant)
                new_accuracy = result.get("accuracy", 0.0)
                if new_accuracy > best_accuracy:
                    logger.info(f"  Improvement: {best_accuracy:.0%} -> {new_accuracy:.0%}")
                    best_accuracy = new_accuracy
                    current_template = variant
                    errors = self.analyze_errors(result.get("results", []), self.ground_truth)
                else:
                    logger.info(f"  No improvement ({new_accuracy:.0%} <= {best_accuracy:.0%})")
            else:
                # Without evaluation, just keep the generated variant
                current_template = variant
                logger.info("  Generated variant (no evaluation fn provided)")

        return {
            "best_template": current_template,
            "best_accuracy": best_accuracy,
            "history": self.history,
        }
