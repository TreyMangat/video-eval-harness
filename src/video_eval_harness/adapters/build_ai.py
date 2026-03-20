"""Adapter for Build AI's Egocentric-10K dataset on Hugging Face."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from .dataset_base import BaseAdapter, VideoEntry


class BuildAIEgocentricAdapter(BaseAdapter):
    """Materialize Build AI Egocentric-10K videos locally for benchmarking.

    The dataset is gated on Hugging Face, so callers typically need to:
    1. accept the dataset access conditions on Hugging Face
    2. provide an HF token via ``HF_TOKEN`` or the ``hf_token`` argument
    """

    def __init__(
        self,
        cache_dir: str | Path,
        repo_id: str = "builddotai/Egocentric-10K",
        factories: Optional[list[str]] = None,
        workers: Optional[list[str]] = None,
        limit: int = 10,
        hf_token: Optional[str] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.repo_id = repo_id
        self.factories = factories or []
        self.workers = workers or []
        self.limit = limit
        self.hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def list_videos(self) -> list[VideoEntry]:
        try:
            from datasets import Features, Value, load_dataset
        except ImportError as exc:
            raise RuntimeError(
                "Build AI ingestion requires the optional 'datasets' package. "
                "Install it with: pip install datasets huggingface-hub"
            ) from exc

        features = Features(
            {
                "mp4": Value("binary"),
                "json": {
                    "factory_id": Value("string"),
                    "worker_id": Value("string"),
                    "video_index": Value("int64"),
                    "duration_sec": Value("float64"),
                    "width": Value("int64"),
                    "height": Value("int64"),
                    "fps": Value("float64"),
                    "size_bytes": Value("int64"),
                    "codec": Value("string"),
                },
                "__key__": Value("string"),
                "__url__": Value("string"),
            }
        )

        load_kwargs = {
            "streaming": True,
            "features": features,
        }
        data_files = self._build_data_files()
        if data_files:
            load_kwargs["data_files"] = data_files
        if self.hf_token:
            load_kwargs["token"] = self.hf_token

        dataset = load_dataset(self.repo_id, **load_kwargs)
        iterable = dataset["train"] if hasattr(dataset, "__getitem__") and "train" in dataset else dataset

        entries: list[VideoEntry] = []
        for idx, item in enumerate(iterable):
            if self.limit > 0 and idx >= self.limit:
                break

            key = item.get("__key__") or f"build_ai_{idx:05d}"
            metadata = item.get("json") or {}
            video_bytes = self._coerce_video_bytes(item.get("mp4"))
            if not video_bytes:
                continue

            factory_id = metadata.get("factory_id") or "factory_unknown"
            worker_id = metadata.get("worker_id") or "worker_unknown"
            target_dir = self.cache_dir / factory_id / worker_id
            target_dir.mkdir(parents=True, exist_ok=True)

            video_path = target_dir / f"{key}.mp4"
            if not video_path.exists():
                video_path.write_bytes(video_bytes)

            metadata_path = target_dir / f"{key}.json"
            if metadata and not metadata_path.exists():
                metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            entry_metadata = dict(metadata)
            if item.get("__url__"):
                entry_metadata["source_url"] = item["__url__"]
            entry_metadata["dataset_repo_id"] = self.repo_id

            entries.append(
                VideoEntry(
                    path=video_path,
                    video_id=key,
                    metadata=entry_metadata,
                )
            )

        return entries

    def name(self) -> str:
        return "build_ai_egocentric_10k"

    def _build_data_files(self) -> list[str]:
        if not self.factories and not self.workers:
            return []

        if self.workers and not self.factories:
            raise ValueError("workers can only be specified when factories are also specified")

        if not self.workers:
            return [f"{factory}/**/*.tar" for factory in self.factories]

        return [
            f"{factory}/workers/{worker}/*.tar"
            for factory in self.factories
            for worker in self.workers
        ]

    @staticmethod
    def _coerce_video_bytes(value: object) -> bytes:
        if value is None:
            return b""
        if isinstance(value, bytes):
            return value
        if isinstance(value, bytearray):
            return bytes(value)
        if isinstance(value, memoryview):
            return value.tobytes()
        if isinstance(value, dict):
            payload = value.get("bytes")
            if isinstance(payload, bytes):
                return payload
        return b""
