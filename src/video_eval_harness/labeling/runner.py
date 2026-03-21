"""Labeling runner - orchestrates sending segments to models."""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..sweep import SweepConfig

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from ..caching import ResponseCache
from ..config import ModelConfig
from ..log import get_logger
from ..prompting.templates import PromptBuilder
from ..providers.base import BaseProvider
from ..schemas import ExtractedFrames, Segment, SegmentLabelResult
from ..storage import Storage
from .normalization import parse_model_response

logger = get_logger(__name__)


class LabelingRunner:
    """Run labeling across multiple models for a set of segments."""

    def __init__(
        self,
        providers: dict[str, BaseProvider],
        models: dict[str, ModelConfig],
        prompt_builder: PromptBuilder,
        storage: Storage,
        cache: ResponseCache,
        prompt_version: str = "concise",
        max_concurrency: int = 4,
    ):
        self.providers = providers
        self.models = models
        self.prompt_builder = prompt_builder
        self.storage = storage
        self.cache = cache
        self.prompt_version = prompt_version
        self.max_concurrency = max_concurrency

    def run(
        self,
        run_id: str,
        segments: list[Segment],
        frames_map: dict[str, ExtractedFrames],
        model_names: Optional[list[str]] = None,
    ) -> list[SegmentLabelResult]:
        """Run all configured models against all segments.

        Args:
            run_id: Unique run identifier.
            segments: List of segments to label.
            frames_map: Mapping of segment_id -> ExtractedFrames.
            model_names: Optional subset of models to run. Defaults to all.

        Returns:
            List of all label results.
        """
        if model_names is None:
            model_names = list(self.models.keys())

        # Build work items: (segment, model_name)
        work_items = []
        for seg in segments:
            for model_name in model_names:
                # Skip if already computed (resume support)
                if self.storage.has_result(run_id, seg.segment_id, model_name):
                    logger.debug(f"Skipping {seg.segment_id} x {model_name} (already exists)")
                    continue
                work_items.append((seg, model_name))

        if not work_items:
            logger.info("All results already computed. Nothing to do.")
            existing = self.storage.get_run_results(run_id)
            return existing

        total = len(work_items)
        logger.info(f"Running {total} labeling tasks ({len(segments)} segments x {len(model_names)} models)")

        results: list[SegmentLabelResult] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            transient=False,
        ) as progress:
            task = progress.add_task("Labeling", total=total)

            # Use thread pool for concurrent API calls
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
                futures = {}
                for seg, model_name in work_items:
                    frames = frames_map.get(seg.segment_id)
                    future = pool.submit(
                        self._label_segment, run_id, seg, frames, model_name
                    )
                    futures[future] = (seg.segment_id, model_name)

                for future in concurrent.futures.as_completed(futures):
                    seg_id, mname = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        self.storage.save_label_result(result)
                        status = "ok" if result.parsed_success else "parse_fail"
                        progress.update(task, advance=1, description=f"[{mname}] {seg_id}: {status}")
                    except Exception as e:
                        logger.error(f"Unexpected error for {seg_id} x {mname}: {e}")
                        progress.update(task, advance=1)

        # Include previously cached results
        all_results = self.storage.get_run_results(run_id)
        return all_results

    def run_sweep(
        self,
        run_id: str,
        segments: list[Segment],
        video_paths: dict[str, "Path"],
        sweep_config: "SweepConfig",
        model_names: Optional[list[str]] = None,
    ) -> list[SegmentLabelResult]:
        """Run a sweep: outer loop over extraction variants, inner loop over models.

        For each variant, frames are extracted once and then all models
        are run against those frames.  Results are tagged with sweep
        metadata fields.

        Args:
            run_id: Unique run identifier.
            segments: List of segments to label.
            video_paths: Mapping of video_id -> Path for frame extraction.
            sweep_config: The sweep configuration with variants.
            model_names: Optional subset of models. Defaults to sweep config.

        Returns:
            List of all label results across all variants.
        """
        from ..extraction import FrameExtractor

        if model_names is None:
            model_names = list(self.models.keys())

        variants = sweep_config.variants
        sid = sweep_config.sweep_id
        total_variants = len(variants)

        logger.info(
            f"Sweep {sid}: {total_variants} variants x {len(model_names)} models "
            f"x {len(segments)} segments = {total_variants * len(model_names) * len(segments)} tasks"
        )

        all_results: list[SegmentLabelResult] = []

        for vi, variant in enumerate(variants, 1):
            logger.info(f"  Variant {vi}/{total_variants}: {variant.label} ({variant.variant_id})")

            # Extract frames for this variant
            ext_cfg = variant.to_extraction_config()
            extractor = FrameExtractor(ext_cfg, self.storage)
            frames_map: dict[str, ExtractedFrames] = {}
            for seg in segments:
                vpath = video_paths.get(seg.video_id)
                if vpath is not None:
                    frames_map[seg.segment_id] = extractor.extract(seg, vpath)

            # Build work items for this variant
            work_items = []
            for seg in segments:
                for model_name in model_names:
                    if self.storage.has_result(run_id, seg.segment_id, model_name, variant.variant_id):
                        logger.debug(f"Skipping {seg.segment_id} x {model_name} x {variant.label} (exists)")
                        continue
                    work_items.append((seg, model_name))

            if not work_items:
                logger.info(f"  All results for {variant.label} already computed.")
                continue

            total = len(work_items)
            variant_results: list[SegmentLabelResult] = []

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                transient=False,
            ) as progress:
                task = progress.add_task(f"[{variant.label}]", total=total)

                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
                    futures = {}
                    for seg, model_name in work_items:
                        frames = frames_map.get(seg.segment_id)
                        future = pool.submit(
                            self._label_segment,
                            run_id, seg, frames, model_name,
                            variant_id=variant.variant_id,
                            extraction_label=variant.label,
                            num_frames_used=variant.num_frames,
                            sampling_method_used=variant.method,
                            sweep_id=sid,
                        )
                        futures[future] = (seg.segment_id, model_name)

                    for future in concurrent.futures.as_completed(futures):
                        seg_id, mname = futures[future]
                        try:
                            result = future.result()
                            variant_results.append(result)
                            self.storage.save_label_result(result)
                            status = "ok" if result.parsed_success else "parse_fail"
                            progress.update(
                                task, advance=1,
                                description=f"[{variant.label}|{mname}] {seg_id}: {status}",
                            )
                        except Exception as e:
                            logger.error(f"Error {seg_id} x {mname} x {variant.label}: {e}")
                            progress.update(task, advance=1)

            all_results.extend(variant_results)

        # Return everything including previously stored results
        return self.storage.get_run_results(run_id)

    def _label_segment(
        self,
        run_id: str,
        segment: Segment,
        frames: Optional[ExtractedFrames],
        model_name: str,
        variant_id: str = "",
        extraction_label: str = "",
        num_frames_used: int = 0,
        sampling_method_used: str = "",
        sweep_id: str = "",
    ) -> SegmentLabelResult:
        """Label a single segment with a single model."""
        model_cfg = self.models[model_name]
        provider = self.providers[model_cfg.provider]

        # Build prompt
        num_frames = frames.num_frames if frames else 0
        prompt = self.prompt_builder.build(
            self.prompt_version,
            segment,
            num_frames=num_frames,
        )

        # Check cache (variant_id included for sweep runs)
        prompt_hash = self.cache.hash_content(prompt)
        input_hash = self.cache.hash_content(frames.frame_paths if frames else [])
        cache_key = self.cache.make_key(model_cfg.model_id, prompt_hash, input_hash, variant_id)

        # Sweep fields to tag on every result
        sweep_fields = {}
        if variant_id:
            sweep_fields = dict(
                extraction_variant_id=variant_id,
                extraction_label=extraction_label,
                num_frames_used=num_frames_used,
                sampling_method_used=sampling_method_used,
                sweep_id=sweep_id,
            )

        cached = self.cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit for {segment.segment_id} x {model_name}")
            return parse_model_response(
                raw_text=cached,
                run_id=run_id,
                video_id=segment.video_id,
                segment_id=segment.segment_id,
                start_time_s=segment.start_time_s,
                end_time_s=segment.end_time_s,
                model_name=model_name,
                provider=model_cfg.provider,
                latency_ms=0.0,
                prompt_version=self.prompt_version,
                **sweep_fields,
            )

        # Prepare image paths
        image_paths = None
        if frames and frames.frame_paths and model_cfg.supports_images:
            image_paths = frames.frame_paths

        # Call provider
        response = provider.complete(
            model_id=model_cfg.model_id,
            prompt=prompt,
            image_paths=image_paths,
            max_tokens=model_cfg.max_tokens,
            temperature=model_cfg.temperature,
        )

        if response.success:
            self.cache.set(cache_key, response.text)

        # Use provider-reported cost, or leave None
        cost = response.estimated_cost

        return parse_model_response(
            raw_text=response.text if response.success else "",
            run_id=run_id,
            video_id=segment.video_id,
            segment_id=segment.segment_id,
            start_time_s=segment.start_time_s,
            end_time_s=segment.end_time_s,
            model_name=model_name,
            provider=model_cfg.provider,
            latency_ms=response.latency_ms,
            estimated_cost=cost,
            prompt_version=self.prompt_version,
            error=response.error if not response.success else None,
            **sweep_fields,
        )
