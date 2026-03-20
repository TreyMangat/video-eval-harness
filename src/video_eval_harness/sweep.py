"""Extraction sweep support for multi-config benchmarking.

Sweeps the cartesian product of frame counts and sampling methods so you can
answer: "Does model A still beat model B when you change frame sampling?"
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Any, Optional

from .config import BenchmarkConfig, ExtractionConfig


@dataclass(frozen=True)
class ExtractionVariant:
    """A single extraction configuration in a sweep."""

    num_frames: int
    method: str
    # Shared across all variants in a sweep
    image_format: str = "jpg"
    image_quality: int = 85
    max_dimension: int = 1280
    generate_contact_sheet: bool = False
    contact_sheet_cols: int = 4

    @property
    def variant_id(self) -> str:
        """Deterministic hash of this variant's key parameters."""
        raw = f"{self.method}:{self.num_frames}:{self.image_format}:{self.image_quality}:{self.max_dimension}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    @property
    def label(self) -> str:
        """Human-readable label, e.g. 'uniform_8f'."""
        return f"{self.method}_{self.num_frames}f"

    def to_extraction_config(self) -> ExtractionConfig:
        """Convert to an ExtractionConfig for the frame extractor."""
        return ExtractionConfig(
            num_frames=self.num_frames,
            method=self.method,
            image_format=self.image_format,
            image_quality=self.image_quality,
            max_dimension=self.max_dimension,
            generate_contact_sheet=self.generate_contact_sheet,
            contact_sheet_cols=self.contact_sheet_cols,
        )


@dataclass(frozen=True)
class SweepAxis:
    """Defines the axes to sweep across."""

    num_frames: list[int]
    methods: list[str]

    @property
    def variants(self) -> list[ExtractionVariant]:
        """Cartesian product of num_frames x methods."""
        return [
            ExtractionVariant(num_frames=nf, method=m)
            for nf, m in itertools.product(self.num_frames, self.methods)
        ]


@dataclass
class SweepConfig:
    """Benchmark config extended with optional sweep support.

    If ``axis`` is None, this is a non-sweep run with a single variant
    derived from ``base_extraction``.
    """

    benchmark: BenchmarkConfig
    axis: Optional[SweepAxis]
    # Shared image settings carried from YAML
    image_format: str = "jpg"
    image_quality: int = 85
    max_dimension: int = 1280
    generate_contact_sheet: bool = False
    contact_sheet_cols: int = 4

    @property
    def is_sweep(self) -> bool:
        return self.axis is not None

    @property
    def variants(self) -> list[ExtractionVariant]:
        """All extraction variants to run.

        For non-sweep configs, returns a single variant matching the
        benchmark's extraction config.
        """
        if self.axis is not None:
            # Apply shared image settings to each variant
            return [
                ExtractionVariant(
                    num_frames=v.num_frames,
                    method=v.method,
                    image_format=self.image_format,
                    image_quality=self.image_quality,
                    max_dimension=self.max_dimension,
                    generate_contact_sheet=self.generate_contact_sheet,
                    contact_sheet_cols=self.contact_sheet_cols,
                )
                for v in self.axis.variants
            ]
        # Non-sweep: wrap the single extraction config
        ext = self.benchmark.extraction
        return [
            ExtractionVariant(
                num_frames=ext.num_frames,
                method=ext.method,
                image_format=ext.image_format,
                image_quality=ext.image_quality,
                max_dimension=ext.max_dimension,
                generate_contact_sheet=ext.generate_contact_sheet,
                contact_sheet_cols=ext.contact_sheet_cols,
            )
        ]

    @property
    def sweep_id(self) -> str:
        """Deterministic ID for the full sweep (hash of all variant IDs)."""
        ids = sorted(v.variant_id for v in self.variants)
        raw = ":".join(ids)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


@dataclass
class SweepPlanItem:
    """A single item in the sweep execution plan."""

    model_name: str
    variant: ExtractionVariant


class SweepOrchestrator:
    """Generates the (model x variant) run matrix for a sweep."""

    def __init__(self, sweep_config: SweepConfig):
        self.config = sweep_config

    def plan(self) -> list[SweepPlanItem]:
        """Return the full (model x variant) execution plan.

        Useful for dry-run previews.
        """
        items = []
        for variant in self.config.variants:
            for model_name in self.config.benchmark.models:
                items.append(SweepPlanItem(model_name=model_name, variant=variant))
        return items

    def estimate_api_calls(self, num_segments: int) -> dict[str, Any]:
        """Estimate total API calls and breakdown for cost preview.

        Returns:
            Dict with total_calls, per_variant breakdown, models, variants.
        """
        variants = self.config.variants
        models = self.config.benchmark.models
        calls_per_variant = num_segments * len(models)
        total = calls_per_variant * len(variants)

        return {
            "total_calls": total,
            "num_segments": num_segments,
            "num_models": len(models),
            "num_variants": len(variants),
            "models": list(models),
            "variants": [
                {
                    "variant_id": v.variant_id,
                    "label": v.label,
                    "num_frames": v.num_frames,
                    "method": v.method,
                    "calls": calls_per_variant,
                }
                for v in variants
            ],
        }


def parse_sweep_config(raw: dict[str, Any]) -> SweepConfig:
    """Parse a raw YAML dict into a SweepConfig.

    Handles both sweep and non-sweep YAML formats:

    Non-sweep (backwards compatible):
        extraction:
          num_frames: 8
          method: uniform
          ...

    Sweep:
        extraction:
          sweep:
            num_frames: [4, 8, 16]
            methods: [uniform, keyframe]
          image_format: jpg
          image_quality: 85
          ...
    """
    extraction_raw = raw.get("extraction", {})
    sweep_block = extraction_raw.get("sweep")

    # Shared image settings (used by both sweep and non-sweep)
    image_format = extraction_raw.get("image_format", "jpg")
    image_quality = extraction_raw.get("image_quality", 85)
    max_dimension = extraction_raw.get("max_dimension", 1280)
    generate_contact_sheet = extraction_raw.get("generate_contact_sheet", False)
    contact_sheet_cols = extraction_raw.get("contact_sheet_cols", 4)

    if sweep_block:
        # Sweep mode: extraction.sweep.num_frames and extraction.sweep.methods
        num_frames_list = sweep_block.get("num_frames", [8])
        methods_list = sweep_block.get("methods", ["uniform"])

        if isinstance(num_frames_list, int):
            num_frames_list = [num_frames_list]
        if isinstance(methods_list, str):
            methods_list = [methods_list]

        axis = SweepAxis(num_frames=num_frames_list, methods=methods_list)

        # Build a BenchmarkConfig without extraction (sweep replaces it)
        bench_raw = {k: v for k, v in raw.items() if k != "extraction"}
        bench_raw["extraction"] = {
            "num_frames": num_frames_list[0],
            "method": methods_list[0],
            "image_format": image_format,
            "image_quality": image_quality,
            "max_dimension": max_dimension,
            "generate_contact_sheet": generate_contact_sheet,
            "contact_sheet_cols": contact_sheet_cols,
        }
        benchmark = BenchmarkConfig(**bench_raw)

        return SweepConfig(
            benchmark=benchmark,
            axis=axis,
            image_format=image_format,
            image_quality=image_quality,
            max_dimension=max_dimension,
            generate_contact_sheet=generate_contact_sheet,
            contact_sheet_cols=contact_sheet_cols,
        )
    else:
        # Non-sweep: standard extraction config
        benchmark = BenchmarkConfig(**raw)
        return SweepConfig(
            benchmark=benchmark,
            axis=None,
            image_format=image_format,
            image_quality=image_quality,
            max_dimension=max_dimension,
            generate_contact_sheet=generate_contact_sheet,
            contact_sheet_cols=contact_sheet_cols,
        )
