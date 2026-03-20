"""CLI interface for the video evaluation harness."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import sys

import typer
from rich.console import Console
from rich.table import Table

from . import __version__

if TYPE_CHECKING:
    from .caching import ResponseCache
    from .config import AppSettings, BenchmarkConfig
    from .storage import Storage
    from .sweep import SweepConfig, SweepOrchestrator

app = typer.Typer(
    name="vbench",
    help="Multi-model video segmentation and labeling benchmark harness.",
    no_args_is_help=True,
)
# Force UTF-8 on Windows to avoid cp1252 encoding errors with Rich
if sys.platform == "win32":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
console = Console(force_terminal=True)

# Default paths
DEFAULT_CONFIGS = Path("configs")
DEFAULT_ARTIFACTS = Path("artifacts")


def _setup(log_level: str = "INFO") -> None:
    from .log import setup_logging
    setup_logging(log_level)


@app.command()
def ingest(
    path: str = typer.Argument(..., help="Video file or directory path"),
    artifacts_dir: str = typer.Option(str(DEFAULT_ARTIFACTS), "--artifacts", "-a"),
    log_level: str = typer.Option("INFO", "--log-level", "-l"),
) -> None:
    """Ingest one or more videos and extract metadata."""
    _setup(log_level)
    from .adapters import DirectoryAdapter, LocalFileAdapter
    from .log import get_logger
    from .schemas import VideoMetadata
    from .storage import Storage
    from .utils.ffmpeg import probe_video
    from .utils.ids import generate_video_id

    logger = get_logger(__name__)
    storage = Storage(artifacts_dir)
    p = Path(path)

    if p.is_dir():
        adapter = DirectoryAdapter(p)
    else:
        adapter = LocalFileAdapter([p])

    entries = adapter.list_videos()
    if not entries:
        console.print("[yellow]No video files found.[/yellow]")
        raise typer.Exit(1)

    console.print(f"Found [cyan]{len(entries)}[/cyan] video(s)")

    for entry in entries:
        try:
            info = probe_video(entry.path)
            video_id = generate_video_id(entry.path)
            meta = VideoMetadata(
                video_id=video_id,
                source_path=str(entry.path.resolve()),
                filename=entry.path.name,
                duration_s=info.duration_s,
                width=info.width,
                height=info.height,
                fps=info.fps,
                codec=info.codec,
                file_size_bytes=info.file_size_bytes,
            )
            storage.save_video(meta)
            console.print(f"  [green]OK[/green] {entry.path.name} ->{video_id} ({info.duration_s:.1f}s)")
        except Exception as e:
            console.print(f"  [red]FAIL[/red] {entry.path.name}: {e}")
            logger.error(f"Ingest failed for {entry.path}: {e}")


@app.command()
def segment(
    video_id: Optional[str] = typer.Option(None, "--video-id", "-v", help="Segment a specific video"),
    window: float = typer.Option(10.0, "--window", "-w", help="Window size in seconds"),
    stride: Optional[float] = typer.Option(None, "--stride", "-s", help="Stride in seconds (default: window)"),
    artifacts_dir: str = typer.Option(str(DEFAULT_ARTIFACTS), "--artifacts", "-a"),
    log_level: str = typer.Option("INFO", "--log-level", "-l"),
) -> None:
    """Segment ingested videos into temporal windows."""
    _setup(log_level)
    from .config import SegmentationConfig
    from .segmentation import FixedWindowSegmenter
    from .storage import Storage

    storage = Storage(artifacts_dir)

    if video_id:
        videos = [storage.get_video(video_id)]
        if videos[0] is None:
            console.print(f"[red]Video not found: {video_id}[/red]")
            raise typer.Exit(1)
    else:
        videos = storage.list_videos()

    if not videos:
        console.print("[yellow]No videos to segment. Run 'ingest' first.[/yellow]")
        raise typer.Exit(1)

    cfg = SegmentationConfig(window_size_s=window, stride_s=stride)
    segmenter = FixedWindowSegmenter(cfg)

    total_segments = 0
    for v in videos:
        segments = segmenter.segment(v)
        storage.save_segments(segments)
        total_segments += len(segments)
        console.print(f"  [green]OK[/green] {v.video_id}: {len(segments)} segments")

    console.print(f"Total: [cyan]{total_segments}[/cyan] segments")


@app.command()
def extract_frames(
    video_id: Optional[str] = typer.Option(None, "--video-id", "-v"),
    num_frames: int = typer.Option(8, "--num-frames", "-n"),
    artifacts_dir: str = typer.Option(str(DEFAULT_ARTIFACTS), "--artifacts", "-a"),
    log_level: str = typer.Option("INFO", "--log-level", "-l"),
) -> None:
    """Extract representative frames from segments."""
    _setup(log_level)
    from .config import ExtractionConfig
    from .extraction import FrameExtractor
    from .storage import Storage

    storage = Storage(artifacts_dir)

    if video_id:
        videos = [storage.get_video(video_id)]
        if videos[0] is None:
            console.print(f"[red]Video not found: {video_id}[/red]")
            raise typer.Exit(1)
    else:
        videos = storage.list_videos()

    cfg = ExtractionConfig(num_frames=num_frames)
    extractor = FrameExtractor(cfg, storage)

    total = 0
    for v in videos:
        segments = storage.get_segments(v.video_id)
        if not segments:
            console.print(f"  [yellow]No segments for {v.video_id}. Run 'segment' first.[/yellow]")
            continue
        for seg in segments:
            try:
                frames = extractor.extract(seg, v.source_path)
                total += frames.num_frames
            except Exception as e:
                console.print(f"  [red]FAIL[/red] {seg.segment_id}: {e}")

    console.print(f"Extracted [cyan]{total}[/cyan] frames total")


@app.command()
def label(
    config_file: str = typer.Option(str(DEFAULT_CONFIGS / "benchmark.yaml"), "--config", "-c"),
    models_file: str = typer.Option(str(DEFAULT_CONFIGS / "models.yaml"), "--models", "-m"),
    video_id: Optional[str] = typer.Option(None, "--video-id", "-v"),
    prompt_version: str = typer.Option("concise", "--prompt", "-p"),
    artifacts_dir: str = typer.Option(str(DEFAULT_ARTIFACTS), "--artifacts", "-a"),
    log_level: str = typer.Option("INFO", "--log-level", "-l"),
) -> None:
    """Run labeling with configured models on segmented videos."""
    _setup(log_level)
    from .caching import ResponseCache
    from .config import get_settings, load_benchmark_config, load_models_config, setup_providers
    from .labeling import LabelingRunner
    from .prompting import PromptBuilder
    from .schemas import RunConfig
    from .storage import Storage
    from .utils.ids import generate_run_id

    settings = get_settings()
    storage = Storage(artifacts_dir)
    cache = ResponseCache()

    # Load configs
    models_cfg = load_models_config(models_file)
    bench_cfg = load_benchmark_config(config_file)

    # Determine which models to run
    model_names = bench_cfg.models if bench_cfg.models else list(models_cfg.keys())
    active_models = {k: v for k, v in models_cfg.items() if k in model_names}

    if not active_models:
        console.print("[red]No models configured. Check models.yaml and benchmark.yaml[/red]")
        raise typer.Exit(1)

    # Setup providers
    try:
        providers = setup_providers(active_models, settings)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    # Get videos and segments
    if video_id:
        videos = [storage.get_video(video_id)]
        if videos[0] is None:
            console.print(f"[red]Video not found: {video_id}[/red]")
            raise typer.Exit(1)
    else:
        videos = storage.list_videos()

    all_segments = []
    frames_map = {}
    for v in videos:
        segs = storage.get_segments(v.video_id)
        all_segments.extend(segs)
        for seg in segs:
            frames = storage.get_extracted_frames(seg.segment_id)
            if frames:
                frames_map[seg.segment_id] = frames

    if not all_segments:
        console.print("[yellow]No segments found. Run 'segment' and 'extract-frames' first.[/yellow]")
        raise typer.Exit(1)

    # Create run
    run_id = generate_run_id()
    run_config = RunConfig(
        run_id=run_id,
        models=model_names,
        prompt_version=prompt_version,
        video_ids=[v.video_id for v in videos],
    )
    storage.save_run(run_config)

    console.print(f"Run ID: [cyan]{run_id}[/cyan]")
    console.print(f"Models: {', '.join(model_names)}")
    console.print(f"Segments: {len(all_segments)}")

    # Run labeling
    prompt_builder = PromptBuilder()
    runner = LabelingRunner(
        providers=providers,
        models=active_models,
        prompt_builder=prompt_builder,
        storage=storage,
        cache=cache,
        prompt_version=prompt_version,
        max_concurrency=settings.vbench_max_concurrency,
    )

    results = runner.run(run_id, all_segments, frames_map, model_names)

    # Print summary
    from .evaluation.summaries import print_run_summary
    print_run_summary(results, run_id)

    cache.close()


@app.command()
def run_benchmark(
    path: str = typer.Argument(..., help="Video file or directory"),
    config_file: str = typer.Option(str(DEFAULT_CONFIGS / "benchmark.yaml"), "--config", "-c"),
    models_file: str = typer.Option(str(DEFAULT_CONFIGS / "models.yaml"), "--models", "-m"),
    prompt_version: Optional[str] = typer.Option(None, "--prompt", "-p", help="Override prompt from config"),
    window: Optional[float] = typer.Option(None, "--window", "-w", help="Override window size from config"),
    num_frames: Optional[int] = typer.Option(None, "--num-frames", "-n", help="Override num frames from config"),
    sweep: bool = typer.Option(False, "--sweep", help="Enable extraction sweep from config"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show sweep plan and cost estimate without running"),
    frames: Optional[str] = typer.Option(None, "--frames", help="Override sweep num_frames axis (e.g. 4,8,16)"),
    methods: Optional[str] = typer.Option(None, "--methods", help="Override sweep methods axis (e.g. uniform,keyframe)"),
    model_filter: Optional[str] = typer.Option(None, "--model-filter", help="Comma-separated model subset (e.g. gemini-3-flash,qwen3.5-27b)"),
    artifacts_dir: str = typer.Option(str(DEFAULT_ARTIFACTS), "--artifacts", "-a"),
    log_level: str = typer.Option("INFO", "--log-level", "-l"),
) -> None:
    """Full pipeline: ingest -> segment -> extract -> label -> summarize.

    Settings are read from benchmark.yaml, with CLI flags as overrides.
    Use --sweep to run a multi-config extraction sweep.
    Use --dry-run with --sweep to preview the matrix without API calls.
    Use --model-filter to run only a subset of models from the config.
    """
    _setup(log_level)
    from .caching import ResponseCache as _RC
    from .config import get_settings, load_benchmark_config, load_models_config, load_sweep_config
    from .storage import Storage as _ST
    from .sweep import SweepAxis
    from .sweep import SweepConfig as _SC

    settings = get_settings()
    storage = _ST(artifacts_dir)
    cache = _RC()

    # Load config — sweep-aware or standard
    bench_cfg = None
    sweep_cfg = None
    if sweep:
        sweep_cfg = load_sweep_config(config_file)
    else:
        bench_cfg = load_benchmark_config(config_file)

    models_cfg = load_models_config(models_file)

    # CLI overrides for sweep axes
    if frames is not None or methods is not None:
        frames_list = [int(x.strip()) for x in frames.split(",")] if frames else [8]
        methods_list = [x.strip() for x in methods.split(",")] if methods else ["uniform"]
        cli_axis = SweepAxis(num_frames=frames_list, methods=methods_list)
        if sweep_cfg is not None:
            sweep_cfg.axis = cli_axis
        else:
            # --frames/--methods without --sweep implies sweep mode
            bench_cfg_for_sweep = load_benchmark_config(config_file)
            sweep_cfg = _SC(benchmark=bench_cfg_for_sweep, axis=cli_axis)
        sweep = True

    # Parse model filter
    filter_models = None
    if model_filter:
        filter_models = [m.strip() for m in model_filter.split(",")]

    # Branch: sweep mode vs single-config mode
    if sweep and sweep_cfg is not None:
        _run_sweep(
            sweep_cfg, models_cfg, path, settings, storage, cache, prompt_version, window,
            dry_run, artifacts_dir, log_level, filter_models,
        )
    else:
        _run_single(
            bench_cfg, models_cfg, path, settings, storage, cache, prompt_version, window,
            num_frames, artifacts_dir, log_level, filter_models,
        )

    cache.close()


def _run_single(
    bench_cfg: "BenchmarkConfig",
    models_cfg: dict,
    path: str,
    settings: "AppSettings",
    storage: "Storage",
    cache: "ResponseCache",
    prompt_version: Optional[str],
    window: Optional[float],
    num_frames: Optional[int],
    artifacts_dir: str,
    log_level: str,
    filter_models: Optional[list[str]] = None,
) -> None:
    """Standard single-config benchmark run."""
    from .config import setup_providers, validate_run_config
    from .extraction import FrameExtractor
    from .labeling import LabelingRunner
    from .log import get_logger
    from .prompting import PromptBuilder
    from .schemas import RunConfig, VideoMetadata
    from .segmentation import FixedWindowSegmenter
    from .utils.ffmpeg import probe_video
    from .utils.ids import generate_run_id, generate_video_id

    logger = get_logger(__name__)

    # Pre-flight validation
    model_names_check = bench_cfg.models if bench_cfg.models else list(models_cfg.keys())
    if filter_models:
        model_names_check = [m for m in model_names_check if m in filter_models]
    errors = validate_run_config(model_names_check, models_cfg, settings)
    if errors:
        for err in errors:
            console.print(f"[red]  {err}[/red]")
        raise typer.Exit(1)

    # Segmentation: config-file values with CLI overrides
    seg_cfg = bench_cfg.segmentation
    if window is not None:
        seg_cfg.window_size_s = window

    # Extraction: config-file values with CLI overrides
    ext_cfg = bench_cfg.extraction
    if num_frames is not None:
        ext_cfg.num_frames = num_frames

    # Prompt: CLI override or config value
    effective_prompt = prompt_version or bench_cfg.prompt_version or "concise"

    # 1. Ingest
    console.rule("[bold]Step 1: Ingest[/bold]")
    from .adapters import DirectoryAdapter, LocalFileAdapter

    p = Path(path)
    if p.is_dir():
        adapter = DirectoryAdapter(p)
    else:
        adapter = LocalFileAdapter([p])

    entries = adapter.list_videos()
    if not entries:
        console.print("[red]No video files found.[/red]")
        raise typer.Exit(1)

    videos = []
    for entry in entries:
        try:
            info = probe_video(entry.path)
            vid = generate_video_id(entry.path)
            meta = VideoMetadata(
                video_id=vid,
                source_path=str(entry.path.resolve()),
                filename=entry.path.name,
                duration_s=info.duration_s,
                width=info.width,
                height=info.height,
                fps=info.fps,
                codec=info.codec,
                file_size_bytes=info.file_size_bytes,
            )
            storage.save_video(meta)
            videos.append(meta)
            console.print(f"  [green]OK[/green] {entry.path.name} ({info.duration_s:.1f}s)")
        except Exception as e:
            console.print(f"  [red]FAIL[/red] {entry.path.name}: {e}")

    if not videos:
        console.print("[red]No videos ingested successfully.[/red]")
        raise typer.Exit(1)

    # 2. Segment
    console.rule("[bold]Step 2: Segment[/bold]")
    segmenter = FixedWindowSegmenter(seg_cfg)

    all_segments = []
    for v in videos:
        segs = segmenter.segment(v)
        storage.save_segments(segs)
        all_segments.extend(segs)
        console.print(f"  {v.video_id}: {len(segs)} segments")

    # 3. Extract frames
    console.rule("[bold]Step 3: Extract Frames[/bold]")
    extractor = FrameExtractor(ext_cfg, storage)

    frames_map = {}
    for v in videos:
        segs = storage.get_segments(v.video_id)
        for seg in segs:
            try:
                frames = extractor.extract(seg, v.source_path)
                frames_map[seg.segment_id] = frames
            except Exception as e:
                logger.warning(f"Frame extraction failed for {seg.segment_id}: {e}")

    console.print(f"  Extracted frames for {len(frames_map)} segments")

    # 4. Label
    console.rule("[bold]Step 4: Label with Models[/bold]")
    model_names = bench_cfg.models if bench_cfg.models else list(models_cfg.keys())
    if filter_models:
        model_names = [m for m in model_names if m in filter_models]
    active_models = {k: v for k, v in models_cfg.items() if k in model_names}

    if not active_models:
        console.print("[red]No models configured.[/red]")
        raise typer.Exit(1)

    # Setup providers
    try:
        providers = setup_providers(active_models, settings)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    run_id = generate_run_id()
    run_config = RunConfig(
        run_id=run_id,
        models=model_names,
        prompt_version=effective_prompt,
        segmentation_config=seg_cfg.model_dump(),
        extraction_config=ext_cfg.model_dump(),
        video_ids=[v.video_id for v in videos],
    )
    storage.save_run(run_config)

    console.print(f"  Run ID: [cyan]{run_id}[/cyan]")
    console.print(f"  Models: {', '.join(model_names)}")
    console.print(f"  Window: {seg_cfg.window_size_s}s | Frames: {ext_cfg.num_frames} | Prompt: {effective_prompt}")

    prompt_builder = PromptBuilder()
    runner = LabelingRunner(
        providers=providers,
        models=active_models,
        prompt_builder=prompt_builder,
        storage=storage,
        cache=cache,
        prompt_version=effective_prompt,
        max_concurrency=settings.vbench_max_concurrency,
    )

    results = runner.run(run_id, all_segments, frames_map, model_names)

    # 5. Summarize
    console.rule("[bold]Step 5: Summary[/bold]")
    from .evaluation.summaries import export_results, print_run_summary

    print_run_summary(results, run_id)

    # Export
    run_dir = storage.run_dir(run_id)
    export_results(results, run_dir, run_id)
    console.print(f"\nResults saved to: [cyan]{run_dir}[/cyan]")


def _run_sweep(
    sweep_cfg: "SweepConfig",
    models_cfg: dict,
    path: str,
    settings: "AppSettings",
    storage: "Storage",
    cache: "ResponseCache",
    prompt_version: Optional[str],
    window: Optional[float],
    dry_run: bool,
    artifacts_dir: str,
    log_level: str,
    filter_models: Optional[list[str]] = None,
) -> None:
    """Sweep-mode benchmark run: iterate extraction variants x models."""
    from .config import setup_providers, validate_run_config
    from .labeling import LabelingRunner
    from .prompting import PromptBuilder
    from .schemas import RunConfig, VideoMetadata
    from .segmentation import FixedWindowSegmenter
    from .sweep import SweepOrchestrator
    from .utils.ffmpeg import probe_video
    from .utils.ids import generate_run_id, generate_video_id

    bench_cfg = sweep_cfg.benchmark

    # Pre-flight validation
    model_names_check = bench_cfg.models if bench_cfg.models else list(models_cfg.keys())
    if filter_models:
        model_names_check = [m for m in model_names_check if m in filter_models]
    errors = validate_run_config(
        model_names_check, models_cfg, settings,
        sweep_axis=sweep_cfg.axis if hasattr(sweep_cfg, "axis") else None,
    )
    if errors:
        for err in errors:
            console.print(f"[red]  {err}[/red]")
        raise typer.Exit(1)
    effective_prompt = prompt_version or bench_cfg.prompt_version or "concise"

    # Segmentation override
    seg_cfg = bench_cfg.segmentation
    if window is not None:
        seg_cfg.window_size_s = window

    # Resolve models
    model_names = bench_cfg.models if bench_cfg.models else list(models_cfg.keys())
    if filter_models:
        model_names = [m for m in model_names if m in filter_models]
    active_models = {k: v for k, v in models_cfg.items() if k in model_names}

    if not active_models:
        console.print("[red]No models configured (check --model-filter).[/red]")
        raise typer.Exit(1)

    # 1. Ingest
    console.rule("[bold]Step 1: Ingest[/bold]")
    from .adapters import DirectoryAdapter, LocalFileAdapter

    p = Path(path)
    if p.is_dir():
        adapter = DirectoryAdapter(p)
    else:
        adapter = LocalFileAdapter([p])

    entries = adapter.list_videos()
    if not entries:
        console.print("[red]No video files found.[/red]")
        raise typer.Exit(1)

    videos = []
    for entry in entries:
        try:
            info = probe_video(entry.path)
            vid = generate_video_id(entry.path)
            meta = VideoMetadata(
                video_id=vid,
                source_path=str(entry.path.resolve()),
                filename=entry.path.name,
                duration_s=info.duration_s,
                width=info.width,
                height=info.height,
                fps=info.fps,
                codec=info.codec,
                file_size_bytes=info.file_size_bytes,
            )
            storage.save_video(meta)
            videos.append(meta)
            console.print(f"  [green]OK[/green] {entry.path.name} ({info.duration_s:.1f}s)")
        except Exception as e:
            console.print(f"  [red]FAIL[/red] {entry.path.name}: {e}")

    if not videos:
        console.print("[red]No videos ingested successfully.[/red]")
        raise typer.Exit(1)

    # 2. Segment
    console.rule("[bold]Step 2: Segment[/bold]")
    segmenter = FixedWindowSegmenter(seg_cfg)

    all_segments = []
    for v in videos:
        segs = segmenter.segment(v)
        storage.save_segments(segs)
        all_segments.extend(segs)
        console.print(f"  {v.video_id}: {len(segs)} segments")

    # Sweep plan preview
    orchestrator = SweepOrchestrator(sweep_cfg)
    estimate = orchestrator.estimate_api_calls(len(all_segments))

    _print_sweep_plan(sweep_cfg, orchestrator, estimate)

    if dry_run:
        console.print("\n[yellow]Dry run — no API calls made.[/yellow]")
        return

    # 3. Setup providers
    try:
        providers = setup_providers(active_models, settings)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    # 4. Create run
    run_id = generate_run_id()
    run_config = RunConfig(
        run_id=run_id,
        models=model_names,
        prompt_version=effective_prompt,
        segmentation_config=seg_cfg.model_dump(),
        extraction_config=bench_cfg.extraction.model_dump(),
        video_ids=[v.video_id for v in videos],
        notes=f"sweep:{sweep_cfg.sweep_id}",
    )
    storage.save_run(run_config)

    console.rule("[bold]Step 3: Sweep — Extract & Label[/bold]")
    console.print(f"  Run ID: [cyan]{run_id}[/cyan]")
    console.print(f"  Sweep ID: [cyan]{sweep_cfg.sweep_id}[/cyan]")

    # Build video_paths map for runner.run_sweep
    video_paths = {v.video_id: Path(v.source_path) for v in videos}

    prompt_builder = PromptBuilder()
    runner = LabelingRunner(
        providers=providers,
        models=active_models,
        prompt_builder=prompt_builder,
        storage=storage,
        cache=cache,
        prompt_version=effective_prompt,
        max_concurrency=settings.vbench_max_concurrency,
    )

    results = runner.run_sweep(run_id, all_segments, video_paths, sweep_cfg, model_names)

    # 5. Summarize
    console.rule("[bold]Step 4: Sweep Summary[/bold]")
    from .evaluation.summaries import print_sweep_summary, export_results

    print_sweep_summary(results, run_id)

    run_dir = storage.run_dir(run_id)
    export_results(results, run_dir, run_id)
    console.print(f"\nResults saved to: [cyan]{run_dir}[/cyan]")


def _print_sweep_plan(
    sweep_cfg: "SweepConfig",
    orchestrator: "SweepOrchestrator",
    estimate: dict,
) -> None:
    """Print a Rich table showing the sweep matrix and cost estimate."""
    console.rule("[bold]Sweep Plan[/bold]")

    # Variant table
    vtable = Table(title="Extraction Variants", show_lines=True)
    vtable.add_column("Variant ID", style="cyan")
    vtable.add_column("Label")
    vtable.add_column("Frames", justify="right")
    vtable.add_column("Method")

    for vinfo in estimate["variants"]:
        vtable.add_row(
            vinfo["variant_id"],
            vinfo["label"],
            str(vinfo["num_frames"]),
            vinfo["method"],
        )
    console.print(vtable)

    # Matrix table: models x variants
    models = estimate["models"]
    variants = sweep_cfg.variants

    mtable = Table(title="Model x Variant Matrix", show_lines=True)
    mtable.add_column("Model", style="cyan")
    for v in variants:
        mtable.add_column(v.label, justify="center")

    for model in models:
        row = [model]
        for v in variants:
            row.append("[green]\u2713[/green]")
        mtable.add_row(*row)
    console.print(mtable)

    # Summary
    console.print(f"\n  Segments: [cyan]{estimate['num_segments']}[/cyan]")
    console.print(f"  Models: [cyan]{estimate['num_models']}[/cyan]")
    console.print(f"  Variants: [cyan]{estimate['num_variants']}[/cyan]")
    console.print(f"  Total API calls: [bold yellow]{estimate['total_calls']}[/bold yellow]")


@app.command()
def evaluate(
    run_id: str = typer.Argument(..., help="Run ID to evaluate"),
    artifacts_dir: str = typer.Option(str(DEFAULT_ARTIFACTS), "--artifacts", "-a"),
    export: bool = typer.Option(True, "--export/--no-export"),
    group_by: Optional[str] = typer.Option(None, "--group-by", "-g", help="Group results by 'variant' for sweep runs"),
    log_level: str = typer.Option("INFO", "--log-level", "-l"),
) -> None:
    """Evaluate and summarize results from a previous run."""
    _setup(log_level)
    from .evaluation.summaries import export_results, print_run_summary, print_sweep_summary
    from .storage import Storage

    storage = Storage(artifacts_dir)
    results = storage.get_run_results(run_id)

    if not results:
        console.print(f"[yellow]No results found for run {run_id}[/yellow]")
        raise typer.Exit(1)

    # Detect sweep results or use --group-by variant
    has_sweep_data = any(r.extraction_variant_id for r in results)
    if group_by == "variant" or has_sweep_data:
        print_sweep_summary(results, run_id)
    else:
        print_run_summary(results, run_id)

    if export:
        run_dir = storage.run_dir(run_id)
        paths = export_results(results, run_dir, run_id)
        for p in paths:
            console.print(f"  Exported: [cyan]{p}[/cyan]")


@app.command()
def sweep(
    path: str = typer.Argument(..., help="Video file or directory"),
    config_file: str = typer.Option(str(DEFAULT_CONFIGS / "benchmark.yaml"), "--config", "-c"),
    models_file: str = typer.Option(str(DEFAULT_CONFIGS / "models.yaml"), "--models", "-m"),
    frames: str = typer.Option("4,8,16", "--frames", help="Comma-separated frame counts"),
    methods: str = typer.Option("uniform,keyframe", "--methods", help="Comma-separated sampling methods"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan without running"),
    prompt_version: Optional[str] = typer.Option(None, "--prompt", "-p"),
    window: Optional[float] = typer.Option(None, "--window", "-w"),
    model_filter: Optional[str] = typer.Option(None, "--model-filter", help="Comma-separated model subset (e.g. gemini-3-flash,qwen3.5-27b)"),
    artifacts_dir: str = typer.Option(str(DEFAULT_ARTIFACTS), "--artifacts", "-a"),
    log_level: str = typer.Option("INFO", "--log-level", "-l"),
) -> None:
    """Run an extraction sweep: benchmark across frame counts and sampling methods.

    Equivalent to 'run-benchmark --sweep --frames 4,8,16 --methods uniform,keyframe'.
    Use --model-filter to run only a subset of models.
    """
    _setup(log_level)
    from .caching import ResponseCache
    from .config import load_benchmark_config, load_models_config, get_settings
    from .storage import Storage
    from .sweep import SweepAxis, SweepConfig

    settings = get_settings()
    storage = Storage(artifacts_dir)
    cache = ResponseCache()
    models_cfg = load_models_config(models_file)

    bench_cfg = load_benchmark_config(config_file)
    frames_list = [int(x.strip()) for x in frames.split(",")]
    methods_list = [x.strip() for x in methods.split(",")]
    axis = SweepAxis(num_frames=frames_list, methods=methods_list)
    sweep_cfg = SweepConfig(benchmark=bench_cfg, axis=axis)

    filter_list = [m.strip() for m in model_filter.split(",")] if model_filter else None

    _run_sweep(
        sweep_cfg, models_cfg, path, settings, storage, cache, prompt_version, window,
        dry_run, artifacts_dir, log_level, filter_list,
    )

    cache.close()


@app.command()
def export(
    run_id: str = typer.Argument(..., help="Run ID to export"),
    output_dir: str = typer.Option(".", "--output", "-o"),
    format: str = typer.Option("csv,parquet", "--format", "-f"),
    artifacts_dir: str = typer.Option(str(DEFAULT_ARTIFACTS), "--artifacts", "-a"),
    log_level: str = typer.Option("INFO", "--log-level", "-l"),
) -> None:
    """Export run results to CSV/Parquet."""
    _setup(log_level)
    from .evaluation.summaries import export_results
    from .storage import Storage

    storage = Storage(artifacts_dir)
    results = storage.get_run_results(run_id)

    if not results:
        console.print(f"[yellow]No results found for run {run_id}[/yellow]")
        raise typer.Exit(1)

    formats = [f.strip() for f in format.split(",")]
    paths = export_results(results, output_dir, run_id, formats)
    for p in paths:
        console.print(f"  [green]OK[/green] {p}")


@app.command()
def inspect_run(
    run_id: Optional[str] = typer.Argument(None, help="Run ID (omit to list all runs)"),
    artifacts_dir: str = typer.Option(str(DEFAULT_ARTIFACTS), "--artifacts", "-a"),
    log_level: str = typer.Option("INFO", "--log-level", "-l"),
) -> None:
    """Inspect run details or list all runs."""
    _setup(log_level)
    from .storage import Storage

    storage = Storage(artifacts_dir)

    if run_id is None:
        runs = storage.list_runs()
        if not runs:
            console.print("[yellow]No runs found.[/yellow]")
            return

        table = Table(title="All Runs", show_lines=True)
        table.add_column("Run ID", style="cyan")
        table.add_column("Created")
        table.add_column("Models")
        table.add_column("Videos")
        table.add_column("Prompt")

        for r in runs:
            table.add_row(
                r.run_id,
                r.created_at[:19],
                ", ".join(r.models[:3]),
                str(len(r.video_ids)),
                r.prompt_version,
            )
        console.print(table)
    else:
        run_config = storage.get_run(run_id)
        if run_config is None:
            console.print(f"[red]Run not found: {run_id}[/red]")
            raise typer.Exit(1)

        console.print(f"Run: [cyan]{run_config.run_id}[/cyan]")
        console.print(f"Created: {run_config.created_at}")
        console.print(f"Models: {', '.join(run_config.models)}")
        console.print(f"Videos: {len(run_config.video_ids)}")
        console.print(f"Prompt: {run_config.prompt_version}")

        results = storage.get_run_results(run_id)
        console.print(f"Results: {len(results)}")

        # Quick breakdown
        models = sorted({r.model_name for r in results})
        for m in models:
            mr = [r for r in results if r.model_name == m]
            ok = sum(1 for r in mr if r.parsed_success)
            console.print(f"  {m}: {ok}/{len(mr)} parsed")


@app.command()
def list_videos(
    artifacts_dir: str = typer.Option(str(DEFAULT_ARTIFACTS), "--artifacts", "-a"),
    log_level: str = typer.Option("INFO", "--log-level", "-l"),
) -> None:
    """List all ingested videos."""
    _setup(log_level)
    from .storage import Storage

    storage = Storage(artifacts_dir)
    videos = storage.list_videos()

    if not videos:
        console.print("[yellow]No videos ingested yet.[/yellow]")
        return

    table = Table(title="Ingested Videos", show_lines=True)
    table.add_column("Video ID", style="cyan")
    table.add_column("Filename")
    table.add_column("Duration")
    table.add_column("Resolution")
    table.add_column("FPS")

    for v in videos:
        table.add_row(
            v.video_id,
            v.filename,
            f"{v.duration_s:.1f}s",
            f"{v.width}x{v.height}",
            f"{v.fps:.1f}",
        )
    console.print(table)


@app.command()
def version() -> None:
    """Show version."""
    console.print(f"video-eval-harness v{__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
