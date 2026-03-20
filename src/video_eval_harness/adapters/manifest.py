"""Adapter for loading videos from a CSV/JSON manifest file.

Supports manifests like:
  video_path,label,split
  /data/vid1.mp4,cooking,train
  /data/vid2.mp4,walking,test

Or JSON:
  [{"path": "/data/vid1.mp4", "metadata": {"label": "cooking"}}]
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

from .dataset_base import BaseAdapter, VideoEntry
from .local_files import VIDEO_EXTENSIONS


class ManifestAdapter(BaseAdapter):
    """Adapter that loads video entries from a CSV or JSON manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        path_column: str = "video_path",
        base_dir: Optional[str | Path] = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.path_column = path_column
        self.base_dir = Path(base_dir) if base_dir else self.manifest_path.parent

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

    def list_videos(self) -> list[VideoEntry]:
        suffix = self.manifest_path.suffix.lower()
        if suffix == ".json":
            return self._load_json()
        elif suffix in (".csv", ".tsv"):
            return self._load_csv()
        else:
            raise ValueError(f"Unsupported manifest format: {suffix}. Use .csv, .tsv, or .json")

    def _load_csv(self) -> list[VideoEntry]:
        entries = []
        delimiter = "\t" if self.manifest_path.suffix.lower() == ".tsv" else ","

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                path_val = row.get(self.path_column)
                if not path_val:
                    continue

                video_path = Path(path_val)
                if not video_path.is_absolute():
                    video_path = self.base_dir / video_path

                if not video_path.exists():
                    continue
                if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue

                metadata = {k: v for k, v in row.items() if k != self.path_column}
                entries.append(VideoEntry(
                    path=video_path,
                    video_id=row.get("video_id"),
                    metadata=metadata if metadata else None,
                ))
        return entries

    def _load_json(self) -> list[VideoEntry]:
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = data.get("videos", data.get("data", []))

        entries = []
        for item in data:
            if isinstance(item, str):
                path_val = item
                metadata = None
                video_id = None
            elif isinstance(item, dict):
                path_val = item.get("path", item.get("video_path"))
                metadata = item.get("metadata")
                video_id = item.get("video_id")
            else:
                continue

            if not path_val:
                continue

            video_path = Path(path_val)
            if not video_path.is_absolute():
                video_path = self.base_dir / video_path

            if not video_path.exists():
                continue
            if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            entries.append(VideoEntry(
                path=video_path,
                video_id=video_id,
                metadata=metadata,
            ))
        return entries

    def name(self) -> str:
        return "manifest"
