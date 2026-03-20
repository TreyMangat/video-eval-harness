"""Configuration loading and management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Environment-driven application settings."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    openrouter_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""
    vbench_artifacts_dir: str = "./artifacts"
    vbench_log_level: str = "INFO"
    vbench_max_concurrency: int = 4
    ffmpeg_path: str = "ffmpeg"


class ModelConfig(BaseModel):
    """Configuration for a single model."""

    model_config = {"protected_namespaces": ()}

    name: str
    provider: str = "openrouter"
    model_id: str
    max_tokens: int = 2048
    temperature: float = 0.1
    supports_images: bool = True
    supports_video: bool = False
    notes: Optional[str] = None


class SegmentationConfig(BaseModel):
    """Segmentation parameters."""

    mode: str = "fixed_window"
    window_size_s: float = 10.0
    stride_s: Optional[float] = None  # None means no overlap (stride = window)
    min_segment_s: float = 1.0


class ExtractionConfig(BaseModel):
    """Frame extraction parameters."""

    num_frames: int = 8
    method: str = "uniform"  # uniform, keyframe
    image_format: str = "jpg"
    image_quality: int = 85
    max_dimension: int = 1280
    generate_contact_sheet: bool = False
    contact_sheet_cols: int = 4


class BenchmarkConfig(BaseModel):
    """Full benchmark configuration."""

    name: str = "default"
    models: list[str] = Field(default_factory=list)
    prompt_version: str = "default"
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and return its contents."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_models_config(path: str | Path) -> dict[str, ModelConfig]:
    """Load model configurations from YAML."""
    data = load_yaml(path)
    models = {}
    for name, cfg in data.get("models", {}).items():
        cfg["name"] = name
        models[name] = ModelConfig(**cfg)
    return models


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    """Load benchmark configuration from YAML."""
    data = load_yaml(path)
    return BenchmarkConfig(**data)


def load_prompts_config(path: str | Path) -> dict[str, Any]:
    """Load prompt templates from YAML."""
    return load_yaml(path)


def get_settings() -> AppSettings:
    """Get application settings, loading from .env if present."""
    return AppSettings()


def get_artifacts_dir(settings: Optional[AppSettings] = None) -> Path:
    """Get the artifacts directory path."""
    if settings is None:
        settings = get_settings()
    path = Path(settings.vbench_artifacts_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path
