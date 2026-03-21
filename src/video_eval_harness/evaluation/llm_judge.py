"""LLM-as-judge evaluation for semantic similarity scoring."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..providers.openrouter import OpenRouterProvider

from ..log import get_logger

logger = get_logger(__name__)

JUDGE_MODEL = "openai/gpt-5.4-mini"  # ~$0.001 per judgment

PAIRWISE_PROMPT = """\
You are comparing two action labels generated from the same video segment.
Determine if they describe the same action, even if worded differently.

Label A: "{label_a}"
Label B: "{label_b}"

Respond with ONLY a JSON object:
{{"same_action": true/false, "similarity": 0.0 to 1.0, "brief_reason": "one sentence"}}

Scoring guide:
- 1.0: identical actions, different wording
- 0.8-0.9: same core action, minor difference in scope
- 0.5-0.7: related actions, same general activity
- 0.2-0.4: loosely related
- 0.0: completely different actions
"""

ACCURACY_PROMPT = """\
You are evaluating a model's action label against a ground truth label for a video segment.

Model output: "{prediction}"
Ground truth: "{ground_truth}"

Respond with ONLY a JSON object:
{{"correct": true/false, "similarity": 0.0 to 1.0, "brief_reason": "one sentence"}}

"correct" means the model identified the same action, even with different wording.
"""


class LLMJudge:
    """Use a cheap LLM to judge semantic similarity between action labels."""

    def __init__(self, provider: "OpenRouterProvider", model: str = JUDGE_MODEL):
        self.provider = provider
        self.model = model
        self.call_count = 0
        self.total_cost = 0.0

    def judge_pair(self, label_a: str, label_b: str) -> dict:
        """Compare two labels.

        Returns {"same_action": bool, "similarity": float, "brief_reason": str}.
        """
        prompt = PAIRWISE_PROMPT.format(label_a=label_a, label_b=label_b)
        response = self.provider.complete(
            model_id=self.model, prompt=prompt,
            max_tokens=150, temperature=0.0,
        )
        self.call_count += 1
        self.total_cost += response.estimated_cost or 0.0
        return self._parse(response.text)

    def judge_accuracy(self, prediction: str, ground_truth: str) -> dict:
        """Score a prediction against ground truth.

        Returns {"correct": bool, "similarity": float, "brief_reason": str}.
        """
        prompt = ACCURACY_PROMPT.format(prediction=prediction, ground_truth=ground_truth)
        response = self.provider.complete(
            model_id=self.model, prompt=prompt,
            max_tokens=150, temperature=0.0,
        )
        self.call_count += 1
        self.total_cost += response.estimated_cost or 0.0
        return self._parse(response.text)

    def _parse(self, text: str) -> dict:
        """Parse judge response, handling markdown wrapping."""
        from ..labeling.normalization import extract_json_from_text

        try:
            parsed = extract_json_from_text(text)
            if parsed is not None:
                return parsed
            return json.loads(text)
        except Exception:
            logger.warning(f"Judge parse failed: {text[:200]}")
            return {"same_action": False, "similarity": 0.0, "brief_reason": "parse_error"}

    def stats(self) -> dict:
        """Return usage statistics for this judge session."""
        return {"calls": self.call_count, "total_cost_usd": round(self.total_cost, 6)}
