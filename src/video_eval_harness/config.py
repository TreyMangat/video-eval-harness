"""Configuration loading and management."""

from __future__ import annotations

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
    tier: str = "frontier"
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
    """Load benchmark configuration from YAML.

    If the YAML contains an ``extraction.sweep`` block, use
    :func:`parse_sweep_config` internally but return the underlying
    :class:`BenchmarkConfig` for backwards compatibility.  Callers that
    need the full sweep config should use :func:`load_sweep_config`.
    """
    data = load_yaml(path)
    extraction = data.get("extraction", {})
    if "sweep" in extraction:
        from .sweep import parse_sweep_config
        return parse_sweep_config(data).benchmark
    return BenchmarkConfig(**data)


def load_sweep_config(path: str | Path):
    """Load benchmark config with sweep support.

    Returns a :class:`~video_eval_harness.sweep.SweepConfig`.  If no
    ``extraction.sweep`` block is present the single extraction config
    is wrapped as a 1x1 axis for uniform handling.
    """
    from .sweep import parse_sweep_config
    data = load_yaml(path)
    return parse_sweep_config(data)


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


def setup_providers(
    active_models: dict[str, ModelConfig],
    settings: Optional[AppSettings] = None,
) -> dict[str, Any]:
    """Create provider instances for all providers needed by the active models.

    Returns a dict of {provider_name: provider_instance}.
    """
    from .providers import OpenRouterProvider, OpenAINativeProvider, GeminiNativeProvider

    if settings is None:
        settings = get_settings()

    needed_providers = {m.provider for m in active_models.values()}
    providers: dict[str, Any] = {}

    if "openrouter" in needed_providers:
        if not settings.openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY not set. Required for OpenRouter models.")
        providers["openrouter"] = OpenRouterProvider(api_key=settings.openrouter_api_key)

    if "openai" in needed_providers:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set. Required for native OpenAI models.")
        providers["openai"] = OpenAINativeProvider(api_key=settings.openai_api_key)

    if "gemini" in needed_providers:
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY not set. Required for native Gemini models.")
        providers["gemini"] = GeminiNativeProvider(api_key=settings.google_api_key)

    return providers


def validate_run_config(
    model_names: list[str],
    models_cfg: dict[str, ModelConfig],
    settings: Optional[AppSettings] = None,
    sweep_axis: Optional[Any] = None,
) -> list[str]:
    """Pre-flight validation before any API calls.

    Checks:
        1. All model names in the benchmark exist in models.yaml.
        2. Required API keys are set for the providers needed.
        3. FFmpeg is callable (needed for frame extraction).
        4. Sweep axis values are valid (if provided).

    Returns a list of error messages.  Empty list means all checks passed.
    """
    import shutil

    if settings is None:
        settings = get_settings()

    errors: list[str] = []

    # 1. Model name cross-reference
    missing = [m for m in model_names if m not in models_cfg]
    if missing:
        errors.append(f"Models not found in models.yaml: {', '.join(missing)}")

    # 2. API keys for needed providers
    active = {k: v for k, v in models_cfg.items() if k in model_names}
    needed_providers = {m.provider for m in active.values()}

    provider_keys = {
        "openrouter": ("OPENROUTER_API_KEY", settings.openrouter_api_key),
        "openai": ("OPENAI_API_KEY", settings.openai_api_key),
        "gemini": ("GOOGLE_API_KEY", settings.google_api_key),
    }
    for prov in needed_providers:
        if prov in provider_keys:
            env_name, value = provider_keys[prov]
            if not value:
                errors.append(f"{env_name} not set (required for {prov} models)")

    # 3. FFmpeg availability
    ffmpeg = shutil.which(settings.ffmpeg_path)
    if ffmpeg is None:
        errors.append(
            f"FFmpeg not found (looked for '{settings.ffmpeg_path}'). "
            "Install with: winget install ffmpeg  or  scoop install ffmpeg"
        )

    # 4. Sweep axis validation
    if sweep_axis is not None:
        valid_methods = {"uniform", "keyframe"}
        if hasattr(sweep_axis, "num_frames"):
            for nf in sweep_axis.num_frames:
                if nf < 1 or nf > 128:
                    errors.append(f"Invalid num_frames value: {nf} (must be 1-128)")
        if hasattr(sweep_axis, "methods"):
            for method in sweep_axis.methods:
                if method not in valid_methods:
                    errors.append(f"Invalid sampling method: '{method}' (valid: {', '.join(sorted(valid_methods))})")

    return errors
