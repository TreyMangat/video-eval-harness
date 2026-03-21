"""Adapter for the UCF101 action recognition dataset.

UCF101 organizes videos by action category in a flat directory structure::

    UCF101/
    +-- ApplyEyeMakeup/
    |   +-- v_ApplyEyeMakeup_g01_c01.avi
    |   +-- v_ApplyEyeMakeup_g01_c02.avi
    +-- CuttingInKitchen/
    |   +-- v_CuttingInKitchen_g01_c01.avi
    +-- Hammering/
    |   +-- ...
    +-- ... (101 categories)

The label is the folder name, converted from CamelCase to natural language.
Ground truth is derived directly from the directory structure — no CSV needed.

Usage::

    adapter = UCF101Adapter("data/UCF101", categories=["Hammering", "Knitting"])
    videos = adapter.list_videos()
    gt = adapter.load_ground_truth()
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from ..log import get_logger
from ..schemas import GroundTruthLabel
from .dataset_base import BaseAdapter, VideoEntry
from .local_files import VIDEO_EXTENSIONS

logger = get_logger(__name__)

# UCF101 may contain .avi files not in the default set
_UCF_EXTENSIONS = VIDEO_EXTENSIONS | {".avi"}


def camel_to_label(name: str) -> str:
    """Convert CamelCase category name to a natural language label.

    ``CuttingInKitchen`` -> ``cutting in kitchen``
    ``ApplyEyeMakeup`` -> ``apply eye makeup``
    ``HandStandPushups`` -> ``hand stand pushups``
    """
    words = re.sub(r"([A-Z])", r" \1", name).strip().lower()
    return words


def ucf101_label_to_phrase(folder_name: str) -> str:
    """Convert CamelCase UCF101 folder name to a gerund verb phrase.

    More aggressive normalization than :func:`camel_to_label` — attempts to
    convert the first word to present participle (gerund) form so the label
    is comparable with VLM action descriptions.

    ``ApplyEyeMakeup`` → ``applying eye makeup``
    ``CuttingInKitchen`` → ``cutting in kitchen``
    ``HorseRiding`` → ``horse riding``
    ``Hammering`` → ``hammering``
    ``PlayingGuitar`` → ``playing guitar``
    """
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", folder_name).split()
    if not words:
        return folder_name.lower()

    phrase_words = [w.lower() for w in words]
    first = phrase_words[0]

    # Already a gerund — leave it
    if first.endswith("ing"):
        return " ".join(phrase_words)

    # If any later word is already a gerund (e.g. "HorseRiding"), the first
    # word is a noun modifier — don't convert it.
    if any(w.endswith("ing") for w in phrase_words[1:]):
        return " ".join(phrase_words)

    # Simple gerund conversion heuristics
    if first.endswith("e") and not first.endswith("ee"):
        # "make" → "making", "ride" → "riding", but "see" stays "seeing"
        phrase_words[0] = first[:-1] + "ing"
    elif (
        len(first) >= 3
        and first[-1] not in "aeiouywx"
        and first[-2] in "aeiou"
        and first[-3] not in "aeiou"
    ):
        # CVC pattern: "put" → "putting", "cut" → "cutting", "run" → "running"
        phrase_words[0] = first + first[-1] + "ing"
    else:
        phrase_words[0] = first + "ing"

    return " ".join(phrase_words)


class UCF101Adapter(BaseAdapter):
    """Adapter for the UCF101 action recognition dataset.

    Parameters
    ----------
    data_dir:
        Path to the extracted UCF101 root directory.
    categories:
        Optional list of category folder names to include
        (e.g. ``["CuttingInKitchen", "Hammering"]``). If None, all
        categories are included.
    limit_per_category:
        Maximum clips per category. Keeps runs small for benchmarking.
    """

    def __init__(
        self,
        data_dir: str | Path,
        categories: Optional[list[str]] = None,
        limit_per_category: int = 5,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.categories = categories
        self.limit_per_category = limit_per_category

        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"UCF101 directory not found: {self.data_dir}")

    def list_videos(self) -> list[VideoEntry]:
        """Scan category folders and return video entries."""
        entries: list[VideoEntry] = []

        category_dirs = sorted(
            d for d in self.data_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

        for cat_dir in category_dirs:
            category = cat_dir.name

            if self.categories and category not in self.categories:
                continue

            label = camel_to_label(category)
            clips = sorted(
                f for f in cat_dir.iterdir()
                if f.is_file() and f.suffix.lower() in _UCF_EXTENSIONS
            )

            for clip in clips[: self.limit_per_category]:
                entries.append(VideoEntry(
                    path=clip,
                    video_id=clip.stem,
                    metadata={
                        "category": category,
                        "ground_truth_action": label,
                    },
                ))

        logger.info(
            "UCF101: found %d videos across %d categories in %s",
            len(entries),
            len({e.metadata["category"] for e in entries if e.metadata}),
            self.data_dir,
        )
        return entries

    def load_ground_truth(self, video_id: Optional[str] = None) -> list[GroundTruthLabel]:
        """Generate ground truth labels from the directory structure.

        Each video gets a single ground truth label covering the full clip.
        The primary_action is derived from the category folder name.

        Parameters
        ----------
        video_id:
            If provided, only return the label for this video.
        """
        entries = self.list_videos()
        labels: list[GroundTruthLabel] = []

        for entry in entries:
            if video_id and entry.video_id != video_id:
                continue

            action = entry.metadata["ground_truth_action"] if entry.metadata else ""

            # Create a label that covers the entire video.
            # The segment_id will be matched at comparison time by
            # compute_ground_truth_accuracy using the segment's video_id.
            labels.append(GroundTruthLabel(
                video_id=entry.video_id or "",
                segment_id="",  # matched by video_id, not segment_id
                start_time_s=0.0,
                end_time_s=0.0,
                primary_action=action,
                source="ucf101",
            ))

        logger.info("UCF101: generated %d ground truth labels", len(labels))
        return labels

    def name(self) -> str:
        return "ucf101"
