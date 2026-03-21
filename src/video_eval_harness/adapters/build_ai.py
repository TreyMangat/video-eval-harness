"""Adapter for Build AI's Egocentric-10K dataset.

The dataset is hosted on Hugging Face at ``builddotai/Egocentric-10K`` in
WebDataset format — tar shards containing paired ``.mp4`` and ``.json``
files organized as ``factory_XXX/workers/worker_XXX/partNN.tar``.

This adapter supports two modes:

1. **Local-first**: point ``data_dir`` at a directory of already-extracted
   clips.  The adapter scans for ``.mp4`` files and reads paired ``.json``
   sidecars for metadata.

2. **Tar extraction**: if the directory contains ``.tar`` shards, they are
   extracted in-place on first scan.

Usage::

    adapter = BuildAIAdapter("/data/egocentric-10k")
    videos = adapter.list_videos()
    meta = adapter.load_metadata(videos[0].path)
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Optional

from ..log import get_logger
from .dataset_base import BaseAdapter, VideoEntry
from .local_files import VIDEO_EXTENSIONS

logger = get_logger(__name__)


class BuildAIAdapter(BaseAdapter):
    """Adapter for locally-downloaded Build AI Egocentric-10K data.

    Parameters
    ----------
    data_dir:
        Root directory containing extracted clips or tar shards.
    factory_filter:
        Optional list of factory IDs to include (e.g. ``["factory_001"]``).
    worker_filter:
        Optional list of worker IDs to include (e.g. ``["worker_001"]``).
    """

    def __init__(
        self,
        data_dir: str | Path,
        factory_filter: Optional[list[str]] = None,
        worker_filter: Optional[list[str]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.factory_filter = factory_filter
        self.worker_filter = worker_filter

        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"Build.ai data directory not found: {self.data_dir}")

    def list_videos(self) -> list[VideoEntry]:
        """Scan data_dir for .mp4 files, extracting tars if needed."""
        # Extract any tar shards first
        self._extract_tars()

        entries: list[VideoEntry] = []
        for mp4 in sorted(self.data_dir.rglob("*.mp4")):
            if mp4.suffix.lower() not in VIDEO_EXTENSIONS:
                continue

            # Apply factory/worker filters based on path
            if not self._matches_filters(mp4):
                continue

            video_id = mp4.stem
            metadata = self.load_metadata(mp4)

            entries.append(VideoEntry(
                path=mp4,
                video_id=video_id,
                metadata=metadata,
            ))

        logger.info("Build.ai: found %d videos in %s", len(entries), self.data_dir)
        return entries

    def load_metadata(self, video_path: Path | str) -> Optional[dict]:
        """Read the paired .json sidecar for a video file.

        Returns the parsed JSON dict, or None if no sidecar exists.
        """
        video_path = Path(video_path)
        json_path = video_path.with_suffix(".json")
        if not json_path.exists():
            return None

        try:
            with open(json_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read metadata for %s: %s", video_path.name, e)
            return None

    def name(self) -> str:
        return "buildai"

    def _matches_filters(self, path: Path) -> bool:
        """Check if a video path matches the factory/worker filters."""
        parts = path.relative_to(self.data_dir).parts

        if self.factory_filter:
            if not any(f in parts for f in self.factory_filter):
                return False

        if self.worker_filter:
            if not any(w in parts for w in self.worker_filter):
                return False

        return True

    def _extract_tars(self) -> None:
        """Extract any .tar shards found in data_dir."""
        tar_files = list(self.data_dir.rglob("*.tar"))
        if not tar_files:
            return

        for tar_path in tar_files:
            extract_dir = tar_path.parent
            marker = tar_path.with_suffix(".extracted")

            if marker.exists():
                continue

            logger.info("Extracting %s ...", tar_path.name)
            try:
                with tarfile.open(tar_path, "r:*") as tf:
                    # Filter to only .mp4 and .json files for safety
                    members = [
                        m for m in tf.getmembers()
                        if m.name.endswith((".mp4", ".json")) and not m.name.startswith(("/", ".."))
                    ]
                    tf.extractall(path=extract_dir, members=members)

                # Mark as extracted so we don't re-extract
                marker.touch()
                logger.info("Extracted %d files from %s", len(members), tar_path.name)
            except (tarfile.TarError, OSError) as e:
                logger.error("Failed to extract %s: %s", tar_path.name, e)


def download_buildai_shard(
    output_dir: str | Path,
    factory: str = "factory_001",
    worker: str = "worker_001",
    repo_id: str = "builddotai/Egocentric-10K",
    hf_token: Optional[str] = None,
) -> Path:
    """Download a single worker's shard from Hugging Face.

    Uses ``huggingface_hub`` to fetch the tar file for one factory/worker
    pair. This gets ~50-100 videos without downloading the full dataset.

    Returns the path to the downloaded tar file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Downloading Build.ai data requires 'huggingface-hub'. "
            "Install with: pip install huggingface-hub"
        ) from exc

    import os

    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tar filenames use a no-underscore prefix: factory001_worker001_part00.tar
    prefix = f"{factory.replace('_', '')}_{worker.replace('_', '')}"
    shard_path = f"{factory}/workers/{worker}/{prefix}_part00.tar"

    logger.info("Downloading %s from %s ...", shard_path, repo_id)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=shard_path,
        repo_type="dataset",
        token=token,
        local_dir=str(output_dir),
    )

    return Path(local_path)
