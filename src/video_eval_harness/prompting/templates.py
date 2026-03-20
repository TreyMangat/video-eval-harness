"""Prompt template system using Jinja2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from jinja2 import Template

from ..log import get_logger
from ..schemas import Segment

logger = get_logger(__name__)

# --- Built-in templates ---

CONCISE_LABEL_TEMPLATE = """\
You are analyzing a sequence of {{ num_frames }} frames extracted from a video segment.

Segment info:
- Start time: {{ start_time }}s
- End time: {{ end_time }}s
- Duration: {{ duration }}s

Analyze these frames and identify the actions, objects, and context visible.

Respond with ONLY a JSON object in this exact format:
{
  "primary_action": "the main action occurring in this segment",
  "secondary_actions": ["other actions if any"],
  "description": "concise natural language description of what is happening",
  "objects": ["notable objects or tools visible"],
  "environment_context": "brief description of the environment/setting",
  "confidence": 0.0 to 1.0,
  "reasoning_summary_or_notes": "brief reasoning for your classification",
  "uncertainty_flags": ["list any aspects you are uncertain about"]
}

Be concise, specific, and honest about uncertainty. Do not fabricate details not visible in the frames.
"""

RICH_LABEL_TEMPLATE = """\
You are an expert video analyst examining {{ num_frames }} sequential frames from a {{ duration }}-second \
segment of an egocentric (first-person) video.

Temporal context:
- This segment spans {{ start_time }}s to {{ end_time }}s of the full video.
- Frames are uniformly sampled across this window.

Your task:
1. Carefully examine each frame in sequence to understand the temporal flow.
2. Identify the primary action the camera wearer is performing.
3. Note any secondary or concurrent actions.
4. List notable objects, tools, or items being interacted with.
5. Describe the environment/setting.
6. Assess your confidence in the labeling.

Important guidelines:
- Focus on ACTIONS, not static descriptions.
- If the action is ambiguous, state the most likely action and flag uncertainty.
- Consider temporal progression across frames.
- Do not hallucinate objects or actions not visible.

Respond with ONLY a JSON object:
{
  "primary_action": "most likely primary action",
  "secondary_actions": ["other concurrent or sequential actions"],
  "description": "2-3 sentence description of what is happening across the frames",
  "objects": ["objects, tools, or items visible or being used"],
  "environment_context": "setting description (indoor/outdoor, room type, etc.)",
  "confidence": 0.85,
  "reasoning_summary_or_notes": "brief explanation of your reasoning and what frames show",
  "uncertainty_flags": ["list aspects with low confidence"]
}
"""

ACTION_LABEL_TEMPLATE = """\
You are analyzing {{ num_frames }} frames from a {{ duration }}-second video segment \
({{ start_time }}s–{{ end_time }}s).

Identify the single main action visible across the frames.

Output rules:
- primary_action MUST be a concise verb phrase, maximum 5 words, lowercase, no articles.
  Good: "chopping vegetables", "walking forward", "opening door", "displaying test pattern"
  Bad: "a person is chopping some vegetables", "the door is being opened slowly"
- Put ALL narrative detail in the "description" field, not primary_action.
- secondary_actions: short verb phrases only, same rules as primary_action.
- objects: nouns only, no descriptions.
- confidence: your honest certainty from 0.0 to 1.0.

Respond with ONLY a JSON object:
{
  "primary_action": "verb phrase, max 5 words",
  "secondary_actions": ["other actions if any"],
  "description": "detailed natural language description of what is happening",
  "objects": ["notable objects visible"],
  "environment_context": "brief setting description",
  "confidence": 0.85,
  "reasoning_summary_or_notes": "brief reasoning",
  "uncertainty_flags": ["uncertain aspects if any"]
}
"""

CLAUDE_ACTION_LABEL_TEMPLATE = """\
You are analyzing {{ num_frames }} frames from a {{ duration }}-second video segment \
({{ start_time }}s–{{ end_time }}s).

Identify the single main action visible across the frames.

CRITICAL output rules for primary_action:
- MUST be a concise verb phrase, maximum 5 words, lowercase, no articles.
- Describe the DOMINANT VISUAL, not minor animated elements.
- If the scene shows a static display (test pattern, title card, menu screen), the primary \
action is "displaying [what]" — NOT a description of small moving elements within it.
- Do NOT add qualifiers, adjectives, or sub-elements. Strip to the core verb + object.
  Good: "displaying test pattern", "chopping vegetables", "walking forward"
  Bad: "displaying countdown timer test pattern", "counting down digits sequentially"
- Put ALL detail (timers, counters, specific objects, motion) in "description", not primary_action.
- secondary_actions: short verb phrases only, same rules.
- objects: nouns only, no descriptions.

Respond with ONLY a JSON object:
{
  "primary_action": "verb phrase, max 5 words",
  "secondary_actions": ["other actions if any"],
  "description": "detailed natural language description of what is happening",
  "objects": ["notable objects visible"],
  "environment_context": "brief setting description",
  "confidence": 0.85,
  "reasoning_summary_or_notes": "brief reasoning",
  "uncertainty_flags": ["uncertain aspects if any"]
}
"""

STRICT_JSON_TEMPLATE = """\
Analyze {{ num_frames }} frames from a video segment ({{ start_time }}s-{{ end_time }}s).
Output ONLY valid JSON matching this schema exactly. No markdown, no explanation, no preamble.
{"primary_action":"string","secondary_actions":["string"],"description":"string","objects":["string"],\
"environment_context":"string","confidence":float,"reasoning_summary_or_notes":"string","uncertainty_flags":["string"]}
"""


BUILTIN_TEMPLATES = {
    "concise": CONCISE_LABEL_TEMPLATE,
    "rich": RICH_LABEL_TEMPLATE,
    "action_label": ACTION_LABEL_TEMPLATE,
    "claude_action_label": CLAUDE_ACTION_LABEL_TEMPLATE,
    "strict_json": STRICT_JSON_TEMPLATE,
}


class PromptBuilder:
    """Build prompts from templates with segment context."""

    def __init__(self, templates: Optional[dict[str, str]] = None):
        self.templates = dict(BUILTIN_TEMPLATES)
        if templates:
            self.templates.update(templates)

    @classmethod
    def from_config(cls, prompts_config: dict) -> "PromptBuilder":
        """Load templates from prompts config dict."""
        custom = {}
        for name, cfg in prompts_config.get("templates", {}).items():
            if "template" in cfg:
                custom[name] = cfg["template"]
            elif "file" in cfg:
                custom[name] = Path(cfg["file"]).read_text(encoding="utf-8")
        return cls(templates=custom)

    def build(
        self,
        template_name: str,
        segment: Segment,
        num_frames: int,
        extra_context: Optional[dict[str, Any]] = None,
    ) -> str:
        """Render a prompt template with segment context."""
        if template_name not in self.templates:
            raise ValueError(
                f"Unknown template '{template_name}'. Available: {list(self.templates.keys())}"
            )

        template = Template(self.templates[template_name])
        ctx = {
            "start_time": segment.start_time_s,
            "end_time": segment.end_time_s,
            "duration": segment.duration_s,
            "num_frames": num_frames,
            "segment_index": segment.segment_index,
            "video_id": segment.video_id,
        }
        if extra_context:
            ctx.update(extra_context)

        return template.render(**ctx)

    def list_templates(self) -> list[str]:
        return list(self.templates.keys())
