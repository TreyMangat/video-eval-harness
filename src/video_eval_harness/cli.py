"""CLI interface for the video evaluation harness."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import sys

import typer
from rich.console import Console
from rich.table import Table

from . import __version__

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
    from .config import get_settings, load_benchmark_config, load_models_config
    from .labeling import LabelingRunner
    from .prompting import PromptBuilder
    from .providers import OpenRouterProvider
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
    providers = {}
    if any(m.provider == "openrouter" for m in active_models.values()):
        if not settings.openrouter_api_key:
            console.print("[red]OPENROUTER_API_KEY not set[/red]")
            raise typer.Exit(1)
        providers["openrouter"] = OpenRouterProvider(api_key=settings.openrouter_api_key)

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
    artifacts_dir: str = typer.Option(str(DEFAULT_ARTIFACTS), "--artifacts", "-a"),
    log_level: str = typer.Option("INFO", "--log-level", "-l"),
) -> None:
    """Full pipeline: ingest -> segment -> extract -> label -> summarize.

    Settings are read from benchmark.yaml, with CLI flags as overrides.
    """
    _setup(log_level)
    from .caching import ResponseCache
    from .config import (
        ExtractionConfig,
        SegmentationConfig,
        get_settings,
        load_benchmark_config,
        load_models_config,
    )
    from .extraction import FrameExtractor
    from .labeling import LabelingRunner
    from .log import get_logger
    from .prompting import PromptBuilder
    from .providers import OpenRouterProvider
    from .schemas import RunConfig, VideoMetadata
    from .segmentation import FixedWindowSegmenter
    from .storage import Storage
    from .utils.ffmpeg import probe_video
    from .utils.ids import generate_run_id, generate_video_id

    logger = get_logger(__name__)
    settings = get_settings()
    storage = Storage(artifacts_dir)
    cache = ResponseCache()

    # Load benchmark config first, then apply CLI overrides
    bench_cfg = load_benchmark_config(config_file)
    models_cfg = load_models_config(models_file)

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
    active_models = {k: v for k, v in models_cfg.items() if k in model_names}

    if not active_models:
        console.print("[red]No models configured.[/red]")
        raise typer.Exit(1)

    # Setup providers
    providers = {}
    if any(m.provider == "openrouter" for m in active_models.values()):
        if not settings.openrouter_api_key:
            console.print("[red]OPENROUTER_API_KEY not set. Set it in .env[/red]")
            raise typer.Exit(1)
        providers["openrouter"] = OpenRouterProvider(api_key=settings.openrouter_api_key)

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

    cache.close()


@app.command()
def evaluate(
    run_id: str = typer.Argument(..., help="Run ID to evaluate"),
    artifacts_dir: str = typer.Option(str(DEFAULT_ARTIFACTS), "--artifacts", "-a"),
    export: bool = typer.Option(True, "--export/--no-export"),
    log_level: str = typer.Option("INFO", "--log-level", "-l"),
) -> None:
    """Evaluate and summarize results from a previous run."""
    _setup(log_level)
    from .evaluation.summaries import export_results, print_run_summary
    from .storage import Storage

    storage = Storage(artifacts_dir)
    results = storage.get_run_results(run_id)

    if not results:
        console.print(f"[yellow]No results found for run {run_id}[/yellow]")
        raise typer.Exit(1)

    print_run_summary(results, run_id)

    if export:
        run_dir = storage.run_dir(run_id)
        paths = export_results(results, run_dir, run_id)
        for p in paths:
            console.print(f"  Exported: [cyan]{p}[/cyan]")


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
