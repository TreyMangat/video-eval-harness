"""EPIC-KITCHENS-100 dataset adapter.

Expects::

    data_dir/
    +-- segments/             # Trimmed .mp4 clips
    |   +-- P01_01_take_001.mp4
    |   +-- ...
    +-- EPIC_100_train.csv    # Official annotations (optional)
    +-- ground_truth.json     # VBench-format ground truth

Ground truth can be auto-generated from the narration column in the CSV,
converting imperative verbs to gerund form: ``take plate`` -> ``taking plate``.

Usage::

    adapter = EpicKitchensAdapter("data/epic_kitchens_subset")
    videos = adapter.list_videos()
    gt = adapter.load_ground_truth()
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

from ..log import get_logger
from ..schemas import GroundTruthLabel
from .dataset_base import BaseAdapter, VideoEntry
from .local_files import VIDEO_EXTENSIONS

logger = get_logger(__name__)

# Common EPIC-KITCHENS verbs and their gerund forms
_VERB_GERUND_MAP: dict[str, str] = {
    "take": "taking",
    "put": "putting",
    "open": "opening",
    "close": "closing",
    "wash": "washing",
    "cut": "cutting",
    "mix": "mixing",
    "pour": "pouring",
    "move": "moving",
    "remove": "removing",
    "turn-on": "turning on",
    "turn-off": "turning off",
    "dry": "drying",
    "shake": "shaking",
    "stir": "stirring",
    "throw": "throwing",
    "squeeze": "squeezing",
    "check": "checking",
    "adjust": "adjusting",
    "scoop": "scooping",
    "peel": "peeling",
    "flip": "flipping",
    "insert": "inserting",
    "pick-up": "picking up",
    "set-down": "setting down",
}


def narration_to_gerund(narration: str) -> str:
    """Convert EPIC-KITCHENS narration to gerund verb phrase.

    ``take plate`` -> ``taking plate``
    ``open fridge`` -> ``opening fridge``
    ``wash hands`` -> ``washing hands``
    ``turn-on tap`` -> ``turning on tap``
    """
    narration = narration.strip().lower()
    if not narration:
        return narration

    # Try direct match on first word / hyphenated compound
    words = narration.split()
    first = words[0]

    # Check hyphenated verbs first (e.g. "turn-on")
    if len(words) >= 2:
        compound = f"{words[0]}-{words[1]}"
        if compound in _VERB_GERUND_MAP:
            return _VERB_GERUND_MAP[compound] + " " + " ".join(words[2:])

    if first in _VERB_GERUND_MAP:
        return _VERB_GERUND_MAP[first] + (" " + " ".join(words[1:]) if len(words) > 1 else "")

    # Generic gerund conversion
    if first.endswith("e") and not first.endswith("ee"):
        words[0] = first[:-1] + "ing"
    elif (
        len(first) >= 3
        and first[-1] not in "aeiouywx"
        and first[-2] in "aeiou"
        and first[-3] not in "aeiou"
    ):
        words[0] = first + first[-1] + "ing"
    else:
        words[0] = first + "ing"

    return " ".join(words)


class EpicKitchensAdapter(BaseAdapter):
    """Adapter for the EPIC-KITCHENS-100 dataset.

    Parameters
    ----------
    data_dir:
        Path to the dataset root directory.
    manifest:
        Optional path to a CSV annotation file (EPIC_100_train.csv).
        If not provided, looks for it in ``data_dir/``.
    limit:
        Maximum clips to return. Keeps runs manageable.
    """

    def __init__(
        self,
        data_dir: str | Path,
        manifest: str | Path | None = None,
        limit: int = 50,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.limit = limit

        # Find segments directory
        self.segments_dir = self.data_dir / "segments"
        if not self.segments_dir.is_dir():
            # Fall back to root dir
            self.segments_dir = self.data_dir

        # Load annotations CSV if available
        self.annotations: dict[str, str] = {}
        csv_path = Path(manifest) if manifest else None
        if csv_path is None:
            for candidate in ["EPIC_100_train.csv", "annotations.csv"]:
                p = self.data_dir / candidate
                if p.exists():
                    csv_path = p
                    break

        if csv_path and csv_path.exists():
            self._load_annotations(csv_path)

    def _load_annotations(self, csv_path: Path) -> None:
        """Load narration annotations from EPIC-KITCHENS CSV."""
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # EPIC CSV has fields: narration_id, participant_id, video_id,
                # narration_timestamp, start_timestamp, stop_timestamp, narration, verb, noun, ...
                narration_id = row.get("narration_id", "")
                narration = row.get("narration", "")
                if narration_id and narration:
                    self.annotations[narration_id] = narration

        logger.info("EPIC-KITCHENS: loaded %d annotations from %s", len(self.annotations), csv_path)

    def list_videos(self) -> list[VideoEntry]:
        """Scan segments directory for video clips."""
        entries: list[VideoEntry] = []

        clips = sorted(
            f for f in self.segments_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        )

        for clip in clips[: self.limit]:
            video_id = clip.stem
            narration = self.annotations.get(video_id, "")
            action = narration_to_gerund(narration) if narration else ""

            entries.append(VideoEntry(
                path=clip,
                video_id=video_id,
                metadata={
                    "narration": narration,
                    "ground_truth_action": action,
                    "dataset": "epic_kitchens",
                },
            ))

        logger.info("EPIC-KITCHENS: found %d video clips in %s", len(entries), self.segments_dir)
        return entries

    def load_ground_truth(self, video_id: Optional[str] = None) -> list[GroundTruthLabel]:
        """Load ground truth from ground_truth.json or generate from annotations."""
        # Try VBench-format JSON first
        gt_path = self.data_dir / "ground_truth.json"
        if gt_path.exists():
            with open(gt_path, encoding="utf-8") as f:
                data = json.load(f)

            labels = []
            if isinstance(data, list):
                for entry in data:
                    if video_id and entry.get("video_id") != video_id:
                        continue
                    labels.append(GroundTruthLabel(
                        video_id=entry.get("video_id", ""),
                        segment_id=entry.get("segment_id", ""),
                        start_time_s=entry.get("start_time_s", 0.0),
                        end_time_s=entry.get("end_time_s", 0.0),
                        primary_action=entry["primary_action"],
                        source="epic_kitchens",
                    ))
            logger.info("EPIC-KITCHENS: loaded %d ground truth labels from %s", len(labels), gt_path)
            return labels

        # Fall back to generating from annotations
        entries = self.list_videos()
        labels = []
        for entry in entries:
            if video_id and entry.video_id != video_id:
                continue
            action = entry.metadata.get("ground_truth_action", "") if entry.metadata else ""
            if not action:
                continue
            labels.append(GroundTruthLabel(
                video_id=entry.video_id or "",
                segment_id="",
                start_time_s=0.0,
                end_time_s=0.0,
                primary_action=action,
                source="epic_kitchens",
            ))

        logger.info("EPIC-KITCHENS: generated %d ground truth labels from annotations", len(labels))
        return labels

    def name(self) -> str:
        return "epic_kitchens"
