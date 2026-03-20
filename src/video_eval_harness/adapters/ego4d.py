"""Adapter for the Ego4D dataset.

Ego4D ships a JSON manifest (``ego4d.json``) containing video metadata and
temporal action annotations.  Users typically download only a subset of clips,
so the adapter silently skips manifest entries whose video file is not found
locally.

Usage::

    adapter = Ego4DAdapter("path/to/ego4d.json", "path/to/clips/")
    videos = adapter.list_videos()
    gt = adapter.load_ground_truth("some_video_uid")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..log import get_logger
from ..schemas import GroundTruthLabel
from .dataset_base import BaseAdapter, VideoEntry
from .local_files import VIDEO_EXTENSIONS

logger = get_logger(__name__)


class Ego4DAdapter(BaseAdapter):
    """Adapter for Ego4D dataset manifests.

    Parameters
    ----------
    manifest_path:
        Path to ``ego4d.json`` manifest file.
    video_dir:
        Directory containing the downloaded video clips.
    """

    def __init__(self, manifest_path: str | Path, video_dir: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        self.video_dir = Path(video_dir)

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Ego4D manifest not found: {self.manifest_path}")
        if not self.video_dir.is_dir():
            raise NotADirectoryError(f"Video directory not found: {self.video_dir}")

        self._manifest: Optional[dict] = None

    def _load_manifest(self) -> dict:
        if self._manifest is None:
            with open(self.manifest_path, encoding="utf-8") as f:
                self._manifest = json.load(f)
        return self._manifest

    def list_videos(self) -> list[VideoEntry]:
        """Parse the manifest and return entries for locally-available clips."""
        manifest = self._load_manifest()
        videos_data = manifest.get("videos", [])

        entries: list[VideoEntry] = []
        skipped = 0

        for video in videos_data:
            video_uid = video.get("video_uid", "")
            if not video_uid:
                continue

            # Ego4D clips are typically stored as <video_uid>.mp4
            video_path = self._find_video_file(video_uid)
            if video_path is None:
                skipped += 1
                continue

            metadata = {
                "duration_sec": video.get("duration_sec"),
                "scenario": ", ".join(video.get("scenarios", [])),
            }

            entries.append(VideoEntry(
                path=video_path,
                video_id=video_uid,
                metadata=metadata,
            ))

        if skipped:
            logger.warning(
                "Ego4D: skipped %d videos not found in %s", skipped, self.video_dir,
            )
        logger.info("Ego4D: found %d / %d videos locally", len(entries), len(videos_data))
        return entries

    def _find_video_file(self, video_uid: str) -> Optional[Path]:
        """Look for a video file matching *video_uid* in ``video_dir``."""
        for ext in VIDEO_EXTENSIONS:
            candidate = self.video_dir / f"{video_uid}{ext}"
            if candidate.exists():
                return candidate
        return None

    def load_ground_truth(self, video_id: Optional[str] = None) -> list[GroundTruthLabel]:
        """Extract temporal action annotations from the manifest.

        Parameters
        ----------
        video_id:
            If provided, only return annotations for this video.
            Otherwise return annotations for all videos.

        Returns
        -------
        List of :class:`GroundTruthLabel` objects derived from the manifest's
        ``annotations`` section.
        """
        manifest = self._load_manifest()
        labels: list[GroundTruthLabel] = []

        # Build video duration lookup
        video_durations: dict[str, float] = {}
        for video in manifest.get("videos", []):
            uid = video.get("video_uid", "")
            if uid:
                video_durations[uid] = float(video.get("duration_sec", 0.0))

        # Ego4D annotations live under various keys depending on the benchmark.
        # For temporal action annotations (moments/narrations), the common
        # structures are:
        #   - video["annotations"][*]["labels"][*]
        #   - top-level "annotations" list with "video_uid" references
        # We support both layouts.

        # Layout 1: top-level "annotations" list (Ego4D v2 narrations)
        for annotation in manifest.get("annotations", []):
            ann_vid = annotation.get("video_uid", "")
            if video_id and ann_vid != video_id:
                continue

            for label_entry in annotation.get("labels", []):
                primary = label_entry.get("label", label_entry.get("primary_action", ""))
                if not primary:
                    continue

                start = float(label_entry.get("start_time", label_entry.get("start_time_s", 0.0)))
                end = float(label_entry.get("end_time", label_entry.get("end_time_s", 0.0)))
                segment_id = label_entry.get("segment_id", f"{ann_vid}_seg{int(start):04d}")

                labels.append(GroundTruthLabel(
                    video_id=ann_vid,
                    segment_id=segment_id,
                    start_time_s=start,
                    end_time_s=end,
                    primary_action=primary,
                    secondary_actions=label_entry.get("secondary_actions", []),
                    description=label_entry.get("description"),
                    source="ego4d",
                ))

        # Layout 2: per-video nested annotations
        for video in manifest.get("videos", []):
            vid_uid = video.get("video_uid", "")
            if video_id and vid_uid != video_id:
                continue

            for annotation in video.get("annotations", []):
                for label_entry in annotation.get("labels", []):
                    primary = label_entry.get("label", label_entry.get("primary_action", ""))
                    if not primary:
                        continue

                    start = float(label_entry.get("start_time", label_entry.get("start_time_s", 0.0)))
                    end = float(label_entry.get("end_time", label_entry.get("end_time_s", 0.0)))
                    segment_id = label_entry.get("segment_id", f"{vid_uid}_seg{int(start):04d}")

                    labels.append(GroundTruthLabel(
                        video_id=vid_uid,
                        segment_id=segment_id,
                        start_time_s=start,
                        end_time_s=end,
                        primary_action=primary,
                        secondary_actions=label_entry.get("secondary_actions", []),
                        description=label_entry.get("description"),
                        source="ego4d",
                    ))

        logger.info(
            "Ego4D: loaded %d ground truth labels%s",
            len(labels),
            f" for {video_id}" if video_id else "",
        )
        return labels

    def name(self) -> str:
        return "ego4d"
