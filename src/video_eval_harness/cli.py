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
    from .adapters import BuildAIAdapter, Ego4DAdapter, UCF101Adapter
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
    ground_truth: Optional[str] = typer.Option(None, "--ground-truth", help="Path to ground truth JSON file"),
    adapter: Optional[str] = typer.Option(None, "--adapter", help="Dataset adapter (ego4d, buildai, ucf101)"),
    manifest: Optional[str] = typer.Option(None, "--manifest", help="Manifest path for dataset adapters"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Data directory for dataset adapters"),
    categories: Optional[str] = typer.Option(None, "--categories", help="Comma-separated category filter (ucf101)"),
    limit_per_category: int = typer.Option(5, "--limit-per-category", help="Max clips per category (ucf101)"),
    action_vocabulary: Optional[str] = typer.Option(None, "--action-vocabulary", help="Path to text file with allowed actions (one per line)"),
    max_segments: Optional[int] = typer.Option(None, "--max-segments", help="Max segments (subsample if exceeded)"),
    name: Optional[str] = typer.Option(None, "--name", help="Custom run name for the ID (e.g. 'my-experiment')"),
    public: bool = typer.Option(False, "--public", help="Enforce public cost limits (for Modal API)"),
    llm_judge: bool = typer.Option(False, "--llm-judge", help="Use LLM to score agreement (~$0.001/pair)"),
    artifacts_dir: str = typer.Option(str(DEFAULT_ARTIFACTS), "--artifacts", "-a"),
    log_level: str = typer.Option("INFO", "--log-level", "-l"),
) -> None:
    """Full pipeline: ingest -> segment -> extract -> label -> summarize.

    Settings are read from benchmark.yaml, with CLI flags as overrides.
    Use --sweep to run a multi-config extraction sweep.
    Use --dry-run with --sweep to preview the matrix without API calls.
    Use --model-filter to run only a subset of models from the config.
    Use --ground-truth to evaluate against known labels.
    Use --adapter ego4d --manifest path/to/ego4d.json for Ego4D datasets.
    Use --adapter buildai --data-dir /path/to/data for Build.ai datasets.
    Use --action-vocabulary to constrain labels to a fixed taxonomy.
    Use --max-segments to cap segment count and prevent expensive runs.
    Use --name to set a custom name in the run ID.
    Use --public to enforce server-side cost limits for the public API.
    Use --llm-judge to add LLM-based semantic agreement scoring.
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

    # Load ground truth labels if provided (or auto-load from adapter)
    gt_labels = _load_ground_truth(ground_truth) if ground_truth else None
    dataset_adapter = None
    if adapter == "ego4d":
        dataset_adapter = _make_ego4d_adapter(manifest, path)
        if gt_labels is None:
            gt_labels = dataset_adapter.load_ground_truth()
            if gt_labels:
                console.print(f"Auto-loaded [cyan]{len(gt_labels)}[/cyan] ground truth labels from Ego4D manifest")
    elif adapter == "buildai":
        dataset_adapter = _make_buildai_adapter(data_dir, path)
    elif adapter == "ucf101":
        dataset_adapter = _make_ucf101_adapter(data_dir, path, categories, limit_per_category)
        if gt_labels is None:
            gt_labels = dataset_adapter.load_ground_truth()
            if gt_labels:
                console.print(f"Auto-loaded [cyan]{len(gt_labels)}[/cyan] ground truth labels from UCF101 categories")

    # Load action vocabulary if provided
    vocab_context = _load_action_vocabulary(action_vocabulary) if action_vocabulary else {}

    # Public mode: validate upload before running
    if public:
        from .limits import validate_public_request

        # Probe file size / duration for validation
        p = Path(path)
        if p.is_file():
            file_size = p.stat().st_size
            from .utils.ffmpeg import probe_video
            try:
                info = probe_video(p)
                duration = info.duration_s
            except Exception:
                duration = 0.0
        else:
            file_size = 0
            duration = 0.0

        req_models = []
        if filter_models:
            req_models = filter_models
        elif bench_cfg and bench_cfg.models:
            req_models = bench_cfg.models
        elif sweep_cfg:
            req_models = sweep_cfg.benchmark.models if sweep_cfg.benchmark.models else list(models_cfg.keys())

        is_valid, err_msg = validate_public_request(file_size, duration, req_models)
        if not is_valid:
            console.print(f"[red]Public mode rejected: {err_msg}[/red]")
            raise typer.Exit(1)

        console.print("[bold blue]Running in public mode with cost limits.[/bold blue]")

    # Branch: sweep mode vs single-config mode
    if sweep and sweep_cfg is not None:
        _run_sweep(
            sweep_cfg, models_cfg, path, settings, storage, cache, prompt_version, window,
            dry_run, artifacts_dir, log_level, filter_models, gt_labels, dataset_adapter,
            vocab_context, max_segments, name, public_mode=public, use_llm_judge=llm_judge,
        )
    else:
        _run_single(
            bench_cfg, models_cfg, path, settings, storage, cache, prompt_version, window,
            num_frames, artifacts_dir, log_level, filter_models, gt_labels, dataset_adapter,
            vocab_context, max_segments, name, public_mode=public, use_llm_judge=llm_judge,
        )

    cache.close()


def _ingest_videos(
    path: str,
    storage: "Storage",
    probe_video,
    generate_video_id,
    VideoMetadata,
    dataset_adapter: Optional[object] = None,
) -> tuple[list, list]:
    """Ingest videos using the appropriate adapter.

    Returns (videos, entries) where videos is a list of VideoMetadata and
    entries is the raw VideoEntry list from the adapter.
    """
    if dataset_adapter is not None:
        entries = dataset_adapter.list_videos()
    else:
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
            vid = entry.video_id or generate_video_id(entry.path)
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

    return videos, entries


def _auto_window(videos: list, seg_cfg, window_override: Optional[float]) -> None:
    """Auto-select window size based on max video duration.

    Only applies when --window was not explicitly set. Modifies seg_cfg
    in place.
    """
    if window_override is not None:
        return

    max_duration = max((v.duration_s for v in videos), default=0.0)
    if max_duration <= 0:
        return

    if max_duration < 60:
        auto_window = 10.0
    elif max_duration < 300:
        auto_window = 30.0
    elif max_duration < 1800:
        auto_window = 60.0
    else:
        auto_window = 120.0

    if auto_window != seg_cfg.window_size_s:
        seg_cfg.window_size_s = auto_window
        console.print(
            f"  Auto-selected window size: [cyan]{auto_window:.0f}s[/cyan] "
            f"(max video duration: {max_duration:.0f}s)"
        )


def _budget_guard(total_calls: int, dry_run: bool = False) -> None:
    """Warn and prompt if total API calls exceed the safety threshold."""
    if dry_run or total_calls <= 500:
        return

    est_cost = total_calls * 0.005  # rough average
    console.print(
        f"\n[bold yellow]Warning:[/bold yellow] This run will make "
        f"[bold]{total_calls:,}[/bold] API calls (~${est_cost:.2f} estimated).\n"
        f"  Use --max-segments to limit, or --dry-run to preview."
    )
    typer.confirm("Continue?", abort=True)


def _subsample_segments(segments: list, max_segments: Optional[int]) -> list:
    """Uniformly subsample segments if count exceeds max_segments."""
    if max_segments is None or len(segments) <= max_segments:
        return segments

    original = len(segments)
    step = len(segments) / max_segments
    indices = [int(i * step) for i in range(max_segments)]
    subsampled = [segments[i] for i in indices]
    console.print(
        f"  Subsampled from [yellow]{original}[/yellow] to "
        f"[cyan]{len(subsampled)}[/cyan] segments (--max-segments {max_segments})"
    )
    return subsampled


def _make_ego4d_adapter(manifest: Optional[str], path: str) -> "Ego4DAdapter":
    """Create an Ego4D adapter from CLI args."""
    from .adapters import Ego4DAdapter

    if not manifest:
        console.print("[red]--adapter ego4d requires --manifest path/to/ego4d.json[/red]")
        raise typer.Exit(1)
    return Ego4DAdapter(manifest_path=manifest, video_dir=path)


def _make_buildai_adapter(data_dir: Optional[str], path: str) -> "BuildAIAdapter":
    """Create a Build.ai adapter from CLI args."""
    from .adapters import BuildAIAdapter

    effective_dir = data_dir or path
    return BuildAIAdapter(data_dir=effective_dir)


def _make_ucf101_adapter(
    data_dir: Optional[str],
    path: str,
    categories: Optional[str] = None,
    limit_per_category: int = 5,
) -> "UCF101Adapter":
    """Create a UCF101 adapter from CLI args."""
    from .adapters import UCF101Adapter

    effective_dir = data_dir or path
    cat_list = [c.strip() for c in categories.split(",")] if categories else None
    return UCF101Adapter(
        data_dir=effective_dir,
        categories=cat_list,
        limit_per_category=limit_per_category,
    )


def _load_action_vocabulary(vocab_path: str) -> dict:
    """Load an action vocabulary file and return extra_context for templates."""
    p = Path(vocab_path)
    if not p.exists():
        console.print(f"[red]Action vocabulary file not found: {vocab_path}[/red]")
        raise typer.Exit(1)

    actions = [
        line.strip() for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not actions:
        console.print("[red]Action vocabulary file is empty[/red]")
        raise typer.Exit(1)

    console.print(f"Loaded [cyan]{len(actions)}[/cyan] allowed actions from {vocab_path}")
    return {"allowed_actions": "\n".join(f"  - {a}" for a in actions)}


def _load_ground_truth(gt_path: str) -> list:
    """Load ground truth labels from a JSON file."""
    import json

    from .schemas import GroundTruthLabel

    p = Path(gt_path)
    if not p.exists():
        console.print(f"[red]Ground truth file not found: {gt_path}[/red]")
        raise typer.Exit(1)

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    labels = []
    for entry in data:
        labels.append(GroundTruthLabel(
            video_id=entry.get("video_id", ""),
            segment_id=entry["segment_id"],
            start_time_s=entry.get("start_time_s", 0.0),
            end_time_s=entry.get("end_time_s", 0.0),
            primary_action=entry["primary_action"],
            secondary_actions=entry.get("secondary_actions", []),
            description=entry.get("description"),
            source=entry.get("source"),
        ))
    console.print(f"Loaded [cyan]{len(labels)}[/cyan] ground truth labels from {gt_path}")
    return labels


def _print_ground_truth_accuracy(
    results: list,
    gt_labels: list,
) -> None:
    """Compute and print ground truth accuracy metrics."""
    from .evaluation.metrics import compute_ground_truth_accuracy

    accuracy = compute_ground_truth_accuracy(results, gt_labels)
    if not accuracy:
        return

    gt_table = Table(title="Ground Truth Accuracy", show_lines=True)
    gt_table.add_column("Model", style="cyan", no_wrap=True)
    gt_table.add_column("Exact Match", justify="right")
    gt_table.add_column("Fuzzy Match", justify="right")
    gt_table.add_column("Mean Similarity", justify="right")
    gt_table.add_column("Evaluated Segs", justify="right")

    for model, metrics in sorted(accuracy.items()):
        gt_table.add_row(
            model,
            f"{metrics['exact_match_rate']:.0%}",
            f"{metrics['fuzzy_match_rate']:.0%}",
            f"{metrics['mean_similarity']:.3f}",
            str(int(metrics["evaluated_segments"])),
        )
    console.print(gt_table)


def _run_llm_judge(
    results: list,
    ground_truth_path: Optional[str] = None,
) -> dict:
    """Run LLM-as-judge evaluation and print results.

    Returns {"llm_agreement": ..., "llm_accuracy": ..., "judge_stats": ...}
    so callers can pass the data to export_results().
    """
    from .config import get_settings
    from .evaluation.llm_judge import LLMJudge
    from .evaluation.metrics import compute_agreement_matrix_llm, compute_ground_truth_accuracy_llm
    from .providers.openrouter import OpenRouterProvider

    settings = get_settings()
    if not settings.openrouter_api_key:
        console.print("[red]LLM judge requires OPENROUTER_API_KEY.[/red]")
        return {"llm_agreement": None, "llm_accuracy": None, "judge_stats": None}

    provider = OpenRouterProvider(api_key=settings.openrouter_api_key)
    judge = LLMJudge(provider)

    models = sorted({r.model_name for r in results})
    llm_agreement = None
    llm_accuracy = None

    # Pairwise agreement via LLM
    if len(models) > 1:
        console.print("\n[bold]LLM-Judge Agreement Matrix[/bold] (semantic scoring)")
        llm_agreement = compute_agreement_matrix_llm(results, judge)

        ag_table = Table(title="LLM-Judge Agreement", show_lines=True)
        ag_table.add_column("", style="cyan")
        for m in models:
            ag_table.add_column(m[:20], justify="right")

        for m1 in models:
            row = [m1[:20]]
            for m2 in models:
                val = llm_agreement.get(m1, {}).get(m2, 0)
                row.append(f"{val:.0%}")
            ag_table.add_row(*row)
        console.print(ag_table)

    # Ground truth accuracy via LLM
    if ground_truth_path:
        gt_labels = _load_ground_truth(ground_truth_path)
        if gt_labels:
            console.print("\n[bold]LLM-Judge Ground Truth Accuracy[/bold]")
            llm_accuracy = compute_ground_truth_accuracy_llm(results, gt_labels, judge)

            acc_table = Table(title="LLM-Judge Accuracy", show_lines=True)
            acc_table.add_column("Model", style="cyan", no_wrap=True)
            acc_table.add_column("LLM Accuracy", justify="right")
            acc_table.add_column("LLM Similarity", justify="right")
            acc_table.add_column("Evaluated Segs", justify="right")

            for model, metrics in sorted(llm_accuracy.items()):
                acc_table.add_row(
                    model,
                    f"{metrics['llm_accuracy']:.0%}",
                    f"{metrics['llm_mean_similarity']:.3f}",
                    str(int(metrics["evaluated_segments"])),
                )
            console.print(acc_table)

    # Print judge stats
    stats = judge.stats()
    console.print(
        f"\n  Judge stats: [cyan]{stats['calls']}[/cyan] calls, "
        f"[cyan]${stats['total_cost_usd']:.4f}[/cyan] total cost"
    )

    return {"llm_agreement": llm_agreement, "llm_accuracy": llm_accuracy, "judge_stats": stats}


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
    gt_labels: Optional[list] = None,
    dataset_adapter: Optional[object] = None,
    extra_context: Optional[dict] = None,
    max_segments: Optional[int] = None,
    run_name: Optional[str] = None,
    public_mode: bool = False,
    use_llm_judge: bool = False,
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
    videos, entries = _ingest_videos(path, storage, probe_video, generate_video_id, VideoMetadata, dataset_adapter)

    # Auto-select window size based on video duration
    _auto_window(videos, seg_cfg, window)

    # 2. Segment
    console.rule("[bold]Step 2: Segment[/bold]")
    segmenter = FixedWindowSegmenter(seg_cfg)

    all_segments = []
    for v in videos:
        segs = segmenter.segment(v)
        storage.save_segments(segs)
        all_segments.extend(segs)
        console.print(f"  {v.video_id}: {len(segs)} segments")

    all_segments = _subsample_segments(all_segments, max_segments)

    # Budget guard
    num_calls = len(all_segments) * len(
        (bench_cfg.models if bench_cfg.models else list(models_cfg.keys()))
    )
    _budget_guard(num_calls, dry_run=False)

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

    video_names = [v.filename for v in videos]
    run_id = generate_run_id(video_names=video_names, name=run_name)
    run_config = RunConfig(
        run_id=run_id,
        models=model_names,
        prompt_version=effective_prompt,
        segmentation_config=seg_cfg.model_dump(),
        extraction_config=ext_cfg.model_dump(),
        video_ids=[v.video_id for v in videos],
        display_name=run_name,
    )
    storage.save_run(run_config)

    display = f"[bold]{run_name}[/bold] ({run_id})" if run_name else f"[cyan]{run_id}[/cyan]"
    console.print(f"  Run: {display}")
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
        extra_context=extra_context,
    )

    results = runner.run(run_id, all_segments, frames_map, model_names, public_mode=public_mode)

    # 5. Summarize
    console.rule("[bold]Step 5: Summary[/bold]")
    from .evaluation.summaries import export_results, print_run_summary

    print_run_summary(results, run_id)

    if gt_labels:
        _print_ground_truth_accuracy(results, gt_labels)

    judge_data: dict = {}
    if use_llm_judge:
        judge_data = _run_llm_judge(results)

    # Export
    run_dir = storage.run_dir(run_id)
    export_results(results, run_dir, run_id, display_name=run_name, gt_labels=gt_labels, run_config=run_config, **judge_data)
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
    gt_labels: Optional[list] = None,
    dataset_adapter: Optional[object] = None,
    extra_context: Optional[dict] = None,
    max_segments: Optional[int] = None,
    run_name: Optional[str] = None,
    public_mode: bool = False,
    use_llm_judge: bool = False,
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
    videos, entries = _ingest_videos(path, storage, probe_video, generate_video_id, VideoMetadata, dataset_adapter)

    # Auto-select window size based on video duration
    _auto_window(videos, seg_cfg, window)

    # 2. Segment
    console.rule("[bold]Step 2: Segment[/bold]")
    segmenter = FixedWindowSegmenter(seg_cfg)

    all_segments = []
    for v in videos:
        segs = segmenter.segment(v)
        storage.save_segments(segs)
        all_segments.extend(segs)
        console.print(f"  {v.video_id}: {len(segs)} segments")

    all_segments = _subsample_segments(all_segments, max_segments)

    # Sweep plan preview
    orchestrator = SweepOrchestrator(sweep_cfg)
    estimate = orchestrator.estimate_api_calls(len(all_segments))

    _print_sweep_plan(sweep_cfg, orchestrator, estimate)

    # Budget guard
    _budget_guard(estimate["total_calls"], dry_run=dry_run)

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
    video_names = [v.filename for v in videos]
    run_id = generate_run_id(video_names=video_names, is_sweep=True, name=run_name)
    run_config = RunConfig(
        run_id=run_id,
        models=model_names,
        prompt_version=effective_prompt,
        segmentation_config=seg_cfg.model_dump(),
        extraction_config=bench_cfg.extraction.model_dump(),
        video_ids=[v.video_id for v in videos],
        notes=f"sweep:{sweep_cfg.sweep_id}",
        display_name=run_name,
    )
    storage.save_run(run_config)

    console.rule("[bold]Step 3: Sweep — Extract & Label[/bold]")
    display = f"[bold]{run_name}[/bold] ({run_id})" if run_name else f"[cyan]{run_id}[/cyan]"
    console.print(f"  Run: {display}")
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
        extra_context=extra_context,
    )

    results = runner.run_sweep(run_id, all_segments, video_paths, sweep_cfg, model_names)

    # 5. Summarize
    console.rule("[bold]Step 4: Sweep Summary[/bold]")
    from .evaluation.summaries import print_sweep_summary, export_results

    print_sweep_summary(results, run_id)

    if gt_labels:
        _print_ground_truth_accuracy(results, gt_labels)

    judge_data: dict = {}
    if use_llm_judge:
        judge_data = _run_llm_judge(results)

    run_dir = storage.run_dir(run_id)
    export_results(results, run_dir, run_id, display_name=run_name, gt_labels=gt_labels, run_config=run_config, **judge_data)
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
    llm_judge: bool = typer.Option(False, "--llm-judge", help="Use LLM to score agreement (~$0.001/pair)"),
    ground_truth: Optional[str] = typer.Option(None, "--ground-truth", help="Path to ground truth JSON file"),
    log_level: str = typer.Option("INFO", "--log-level", "-l"),
) -> None:
    """Evaluate and summarize results from a previous run.

    Use --llm-judge to add LLM-based semantic agreement scoring alongside
    the default string-matching metrics.
    """
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

    # LLM-as-judge evaluation
    judge_data: dict = {}
    if llm_judge:
        judge_data = _run_llm_judge(results, ground_truth)

    if export:
        run_cfg = storage.get_run(run_id)
        dn = run_cfg.display_name if run_cfg else None
        gt_labels = _load_ground_truth(ground_truth) if ground_truth else None
        run_dir = storage.run_dir(run_id)
        paths = export_results(results, run_dir, run_id, display_name=dn, gt_labels=gt_labels, run_config=run_cfg, **judge_data)
        for p in paths:
            console.print(f"  Exported: [cyan]{p}[/cyan]")
        if judge_data.get("llm_agreement"):
            console.print("  Updated JSON export with LLM judge scores.")


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
    ground_truth: Optional[str] = typer.Option(None, "--ground-truth", help="Path to ground truth JSON file"),
    adapter: Optional[str] = typer.Option(None, "--adapter", help="Dataset adapter (ego4d, buildai, ucf101)"),
    manifest: Optional[str] = typer.Option(None, "--manifest", help="Manifest path for dataset adapters"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Data directory for dataset adapters"),
    categories: Optional[str] = typer.Option(None, "--categories", help="Comma-separated category filter (ucf101)"),
    limit_per_category: int = typer.Option(5, "--limit-per-category", help="Max clips per category (ucf101)"),
    action_vocabulary: Optional[str] = typer.Option(None, "--action-vocabulary", help="Path to text file with allowed actions (one per line)"),
    max_segments: Optional[int] = typer.Option(None, "--max-segments", help="Max segments (subsample if exceeded)"),
    name: Optional[str] = typer.Option(None, "--name", help="Custom run name for the ID"),
    artifacts_dir: str = typer.Option(str(DEFAULT_ARTIFACTS), "--artifacts", "-a"),
    log_level: str = typer.Option("INFO", "--log-level", "-l"),
) -> None:
    """Run an extraction sweep: benchmark across frame counts and sampling methods.

    Equivalent to 'run-benchmark --sweep --frames 4,8,16 --methods uniform,keyframe'.
    Use --model-filter to run only a subset of models.
    Use --adapter ucf101 --data-dir data/UCF101/ for UCF101 with auto ground truth.
    Use --adapter ego4d --manifest path/to/ego4d.json for Ego4D datasets.
    Use --adapter buildai --data-dir /path/to/data for Build.ai datasets.
    Use --action-vocabulary to constrain labels to a fixed taxonomy.
    Use --max-segments to cap segment count and prevent expensive runs.
    Use --name to set a custom name in the run ID.
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
    gt_labels = _load_ground_truth(ground_truth) if ground_truth else None
    dataset_adapter = None
    if adapter == "ego4d":
        dataset_adapter = _make_ego4d_adapter(manifest, path)
        if gt_labels is None:
            gt_labels = dataset_adapter.load_ground_truth()
            if gt_labels:
                console.print(f"Auto-loaded [cyan]{len(gt_labels)}[/cyan] ground truth labels from Ego4D manifest")
    elif adapter == "buildai":
        dataset_adapter = _make_buildai_adapter(data_dir, path)
    elif adapter == "ucf101":
        dataset_adapter = _make_ucf101_adapter(data_dir, path, categories, limit_per_category)
        if gt_labels is None:
            gt_labels = dataset_adapter.load_ground_truth()
            if gt_labels:
                console.print(f"Auto-loaded [cyan]{len(gt_labels)}[/cyan] ground truth labels from UCF101 categories")

    vocab_context = _load_action_vocabulary(action_vocabulary) if action_vocabulary else {}

    _run_sweep(
        sweep_cfg, models_cfg, path, settings, storage, cache, prompt_version, window,
        dry_run, artifacts_dir, log_level, filter_list, gt_labels, dataset_adapter,
        vocab_context, max_segments, name,
    )

    cache.close()


@app.command()
def test_suite(
    videos: str = typer.Option("test_videos", "--videos", "-v", help="Video directory"),
    tier: str = typer.Option("fast", "--tier", "-t", help="Model tier: fast or frontier"),
    max_segments: int = typer.Option(25, "--max-segments", help="Max segments per run"),
    name: Optional[str] = typer.Option(None, "--name", help="Custom run name for the ID"),
    models_file: str = typer.Option(str(DEFAULT_CONFIGS / "models.yaml"), "--models", "-m"),
    artifacts_dir: str = typer.Option(str(DEFAULT_ARTIFACTS), "--artifacts", "-a"),
    log_level: str = typer.Option("INFO", "--log-level", "-l"),
) -> None:
    """Run the standard benchmark test suite.

    The recommended way to run benchmarks. Encapsulates best practices:
    fast models with 2 extraction variants, max 25 segments, auto-windowing.

    Shows a cost estimate and asks for confirmation before making API calls.
    """
    _setup(log_level)
    import math

    from .config import load_benchmark_config, load_models_config, get_settings
    from .caching import ResponseCache
    from .storage import Storage
    from .sweep import SweepAxis, SweepConfig
    from .utils.ffmpeg import probe_video

    # Select config by tier
    if tier == "frontier":
        config_file = str(DEFAULT_CONFIGS / "benchmark.yaml")
    else:
        config_file = str(DEFAULT_CONFIGS / "benchmark_fast.yaml")

    settings = get_settings()
    bench_cfg = load_benchmark_config(config_file)
    models_cfg = load_models_config(models_file)

    model_names = bench_cfg.models if bench_cfg.models else list(models_cfg.keys())

    # Standard sweep: 2 variants (4f + 8f, uniform only)
    frames_list = [4, 8]
    methods_list = ["uniform"]
    num_variants = len(frames_list) * len(methods_list)

    # Probe video durations for estimate
    videos_path = Path(videos)
    if not videos_path.is_dir():
        console.print(f"[red]Video directory not found: {videos}[/red]")
        raise typer.Exit(1)

    from .adapters import DirectoryAdapter
    entries = DirectoryAdapter(videos_path).list_videos()
    if not entries:
        console.print(f"[red]No video files found in {videos}[/red]")
        raise typer.Exit(1)

    durations = []
    for e in entries:
        try:
            info = probe_video(e.path)
            durations.append(info.duration_s)
        except Exception:
            pass

    if not durations:
        console.print("[red]Could not probe any video files.[/red]")
        raise typer.Exit(1)

    total_duration = sum(durations)
    max_duration = max(durations)

    # Auto-window
    if max_duration < 60:
        auto_window = 10.0
    elif max_duration < 300:
        auto_window = 30.0
    elif max_duration < 1800:
        auto_window = 60.0
    else:
        auto_window = 120.0

    total_segments = sum(math.ceil(d / auto_window) for d in durations)
    effective_segments = min(total_segments, max_segments)
    total_calls = effective_segments * len(model_names) * num_variants

    # Show estimate
    console.rule("[bold]Test Suite Plan[/bold]")
    table = Table(show_lines=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Videos", f"{len(durations)} clips ({total_duration:.0f}s total)")
    table.add_row("Tier", f"[bold]{tier}[/bold]")
    table.add_row("Models", f"{len(model_names)} ({', '.join(model_names[:3])}{'...' if len(model_names) > 3 else ''})")
    table.add_row("Prompt", bench_cfg.prompt_version)
    table.add_row("Window", f"{auto_window:.0f}s (auto)")
    table.add_row("Variants", f"{num_variants} ({', '.join(f'{f}f' for f in frames_list)} x {', '.join(methods_list)})")
    if total_segments > max_segments:
        table.add_row("Segments", f"{effective_segments} (capped from {total_segments})")
    else:
        table.add_row("Segments", str(effective_segments))
    table.add_row("API calls", f"[bold yellow]{total_calls}[/bold yellow]")
    table.add_row("Est. cost", f"${total_calls * 0.005:.2f} – ${total_calls * 0.015:.2f}")
    console.print(table)

    typer.confirm("\nProceed with test suite?", abort=True)

    # Run the sweep
    storage = Storage(artifacts_dir)
    cache = ResponseCache()
    axis = SweepAxis(num_frames=frames_list, methods=methods_list)
    sweep_cfg = SweepConfig(benchmark=bench_cfg, axis=axis)

    _run_sweep(
        sweep_cfg, models_cfg, videos, settings, storage, cache,
        None, None,  # prompt_version, window (use config defaults + auto)
        False, artifacts_dir, log_level,
        None, None, None, {},  # filter_models, gt_labels, dataset_adapter, extra_context
        max_segments, name,
    )

    # Export sweep summary
    runs = storage.list_runs()
    if runs:
        latest_run = runs[0]
        results = storage.get_run_results(latest_run.run_id)
        has_sweep = any(r.extraction_variant_id for r in results)
        if has_sweep:
            import json
            from dataclasses import asdict
            from .evaluation.metrics import compute_sweep_metrics

            sweep_data = compute_sweep_metrics(results)
            summary = {
                "run_id": latest_run.run_id,
                "cells": [asdict(c) for c in sweep_data["cells"]],
                "stability": [asdict(s) for s in sweep_data["stability"]],
                "agreement_by_variant": sweep_data["agreement_by_variant"],
            }
            run_dir = storage.run_dir(latest_run.run_id)
            out_path = Path(run_dir) / f"{latest_run.run_id}_sweep_summary.json"
            out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            console.print(f"\nSweep summary: [cyan]{out_path}[/cyan]")

        console.print("\nView results:    [bold]streamlit run src/video_eval_harness/viewer.py[/bold]")

    cache.close()


@app.command()
def estimate(
    path: str = typer.Argument(..., help="Video file or directory"),
    config_file: str = typer.Option(str(DEFAULT_CONFIGS / "benchmark.yaml"), "--config", "-c"),
    models_file: str = typer.Option(str(DEFAULT_CONFIGS / "models.yaml"), "--models", "-m"),
    window: Optional[float] = typer.Option(None, "--window", "-w", help="Override window size"),
    sweep: bool = typer.Option(False, "--sweep", help="Estimate as sweep run"),
    frames: Optional[str] = typer.Option(None, "--frames", help="Sweep frame counts (e.g. 4,8,16)"),
    methods: Optional[str] = typer.Option(None, "--methods", help="Sweep methods (e.g. uniform,keyframe)"),
    model_filter: Optional[str] = typer.Option(None, "--model-filter", help="Comma-separated model subset"),
    adapter: Optional[str] = typer.Option(None, "--adapter", help="Dataset adapter (ego4d, buildai)"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Data directory for dataset adapters"),
    manifest: Optional[str] = typer.Option(None, "--manifest", help="Manifest path for ego4d adapter"),
    log_level: str = typer.Option("INFO", "--log-level", "-l"),
) -> None:
    """Estimate API calls, cost, and time without running anything.

    Probes video duration(s), computes segment count, and multiplies by
    models and variants. No frames are extracted, no API calls are made.
    """
    _setup(log_level)
    from .config import load_benchmark_config, load_models_config
    from .utils.ffmpeg import probe_video

    bench_cfg = load_benchmark_config(config_file)
    models_cfg = load_models_config(models_file)

    # Resolve models
    model_names = bench_cfg.models if bench_cfg.models else list(models_cfg.keys())
    if model_filter:
        model_names = [m.strip() for m in model_filter.split(",") if m.strip() in model_names]

    # Probe video durations
    p = Path(path)
    if adapter == "buildai":
        from .adapters import BuildAIAdapter
        ba = BuildAIAdapter(data_dir=data_dir or path)
        entries = ba.list_videos()
        durations = []
        for e in entries:
            try:
                info = probe_video(e.path)
                durations.append(info.duration_s)
            except Exception:
                pass
    elif adapter == "ego4d":
        from .adapters import Ego4DAdapter
        ego = Ego4DAdapter(manifest_path=manifest or "", video_dir=data_dir or path)
        entries = ego.list_videos()
        durations = []
        for e in entries:
            try:
                info = probe_video(e.path)
                durations.append(info.duration_s)
            except Exception:
                pass
    elif p.is_dir():
        from .adapters import DirectoryAdapter
        entries = DirectoryAdapter(p).list_videos()
        durations = []
        for e in entries:
            try:
                info = probe_video(e.path)
                durations.append(info.duration_s)
            except Exception:
                pass
    else:
        try:
            info = probe_video(p)
            durations = [info.duration_s]
        except Exception as e:
            console.print(f"[red]Cannot probe {path}: {e}[/red]")
            raise typer.Exit(1)

    if not durations:
        console.print("[red]No videos found.[/red]")
        raise typer.Exit(1)

    total_duration = sum(durations)
    max_duration = max(durations)

    # Auto-window
    seg_window = window or bench_cfg.segmentation.window_size_s
    if window is None:
        if max_duration < 60:
            seg_window = 10.0
        elif max_duration < 300:
            seg_window = 30.0
        elif max_duration < 1800:
            seg_window = 60.0
        else:
            seg_window = 120.0

    # Compute segments
    import math
    total_segments = sum(math.ceil(d / seg_window) for d in durations)

    # Sweep variants
    num_variants = 1
    if sweep or frames or methods:
        frames_list = [int(x.strip()) for x in frames.split(",")] if frames else [4, 8, 16]
        methods_list = [x.strip() for x in methods.split(",")] if methods else ["uniform", "keyframe"]
        num_variants = len(frames_list) * len(methods_list)

    num_models = len(model_names)
    total_calls = total_segments * num_models * num_variants

    # Cost estimate: rough per-call cost by tier
    avg_cost_per_call = 0.005  # ~$0.005 per call for fast models, ~$0.015 for frontier
    tier_costs = {"frontier": 0.015, "fast": 0.003}
    cost_sum = 0.0
    for m in model_names:
        cfg = models_cfg.get(m)
        tier = getattr(cfg, "tier", "frontier") if cfg else "frontier"
        cost_sum += tier_costs.get(tier, 0.01)
    avg_cost_per_call = cost_sum / num_models if num_models > 0 else 0.005

    est_cost_low = total_calls * avg_cost_per_call * 0.5
    est_cost_high = total_calls * avg_cost_per_call * 1.5

    # Time estimate: 5s default per call
    avg_latency_s = 5.0
    est_time_s = total_calls * avg_latency_s / max(4, num_models)  # parallel across models

    # Print
    console.rule("[bold]Estimate[/bold]")
    table = Table(show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Videos", str(len(durations)))
    table.add_row("Total duration", f"{total_duration:.0f}s ({total_duration / 60:.1f}m)")
    table.add_row("Window size", f"{seg_window:.0f}s" + (" (auto)" if window is None else ""))
    table.add_row("Segments", str(total_segments))
    table.add_row("Models", f"{num_models} ({', '.join(model_names[:3])}{'...' if num_models > 3 else ''})")
    if num_variants > 1:
        table.add_row("Variants", str(num_variants))
    table.add_row("Total API calls", f"[bold yellow]{total_calls}[/bold yellow]")
    table.add_row("Est. cost", f"${est_cost_low:.2f} – ${est_cost_high:.2f}")
    table.add_row("Est. time", f"{est_time_s / 60:.1f} min")

    console.print(table)


@app.command()
def compare(
    run_a: str = typer.Argument(..., help="First run ID (baseline)"),
    run_b: str = typer.Argument(..., help="Second run ID (comparison)"),
    artifacts_dir: str = typer.Option(str(DEFAULT_ARTIFACTS), "--artifacts", "-a"),
    log_level: str = typer.Option("INFO", "--log-level", "-l"),
) -> None:
    """Compare two runs side-by-side: parse rate, latency, confidence, agreement, cost."""
    _setup(log_level)
    from .evaluation.metrics import compute_agreement_matrix, compute_model_summary
    from .storage import Storage

    storage = Storage(artifacts_dir)
    results_a = storage.get_run_results(run_a)
    results_b = storage.get_run_results(run_b)

    if not results_a:
        console.print(f"[red]No results for run {run_a}[/red]")
        raise typer.Exit(1)
    if not results_b:
        console.print(f"[red]No results for run {run_b}[/red]")
        raise typer.Exit(1)

    models_a = sorted({r.model_name for r in results_a})
    models_b = sorted({r.model_name for r in results_b})
    all_models = sorted(set(models_a) | set(models_b))

    summaries_a = {m: compute_model_summary(results_a, m) for m in models_a}
    summaries_b = {m: compute_model_summary(results_b, m) for m in models_b}

    def _delta_str(val_a: float | None, val_b: float | None, fmt: str, invert: bool = False) -> str:
        """Format a delta value with green/red coloring. invert=True means lower is better."""
        if val_a is None or val_b is None:
            return "-"
        diff = val_b - val_a
        if abs(diff) < 1e-9:
            return f"{diff:{fmt}}"
        better = diff < 0 if invert else diff > 0
        color = "green" if better else "red"
        sign = "+" if diff > 0 else ""
        return f"[{color}]{sign}{diff:{fmt}}[/{color}]"

    # Per-model comparison table
    table = Table(title=f"Compare: {run_a[:12]} vs {run_b[:12]}", show_lines=True)
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Parse %\n(delta)", justify="right")
    table.add_column("Avg Latency\n(delta ms)", justify="right")
    table.add_column("Avg Conf\n(delta)", justify="right")
    table.add_column("Cost\n(delta $)", justify="right")

    for m in all_models:
        sa = summaries_a.get(m)
        sb = summaries_b.get(m)
        if sa is None and sb is None:
            continue
        parse_d = _delta_str(
            sa.parse_success_rate if sa else None,
            sb.parse_success_rate if sb else None,
            ".1%",
        )
        lat_d = _delta_str(
            sa.avg_latency_ms if sa else None,
            sb.avg_latency_ms if sb else None,
            ".0f",
            invert=True,
        )
        conf_d = _delta_str(
            sa.avg_confidence if sa else None,
            sb.avg_confidence if sb else None,
            ".3f",
        )
        cost_d = _delta_str(
            sa.total_estimated_cost if sa else None,
            sb.total_estimated_cost if sb else None,
            ".4f",
            invert=True,
        )
        table.add_row(m, parse_d, lat_d, conf_d, cost_d)

    console.print(table)

    # Agreement comparison for shared models
    shared = sorted(set(models_a) & set(models_b))
    if len(shared) > 1:
        agree_a = compute_agreement_matrix(results_a)
        agree_b = compute_agreement_matrix(results_b)

        ag_table = Table(title="Agreement Delta (run_b - run_a)", show_lines=True)
        ag_table.add_column("", style="cyan")
        for m in shared:
            ag_table.add_column(m[:20], justify="right")

        for m1 in shared:
            row = [m1[:20]]
            for m2 in shared:
                va = agree_a.get(m1, {}).get(m2, 0.0)
                vb = agree_b.get(m1, {}).get(m2, 0.0)
                diff = vb - va
                if m1 == m2:
                    row.append("-")
                elif abs(diff) < 1e-9:
                    row.append(f"{diff:+.1%}")
                else:
                    color = "green" if diff > 0 else "red"
                    row.append(f"[{color}]{diff:+.1%}[/{color}]")
            ag_table.add_row(*row)

        console.print(ag_table)


@app.command()
def export(
    run_id: str = typer.Argument(..., help="Run ID to export"),
    output_dir: str = typer.Option(".", "--output", "-o"),
    format: str = typer.Option("csv,parquet,json", "--format", "-f"),
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

    run_cfg = storage.get_run(run_id)
    dn = run_cfg.display_name if run_cfg else None
    formats = [f.strip() for f in format.split(",")]
    paths = export_results(results, output_dir, run_id, formats, display_name=dn, run_config=run_cfg)
    for p in paths:
        console.print(f"  [green]OK[/green] {p}")


@app.command()
def export_sweep_summary(
    run_id: str = typer.Argument(..., help="Run ID of a sweep run"),
    artifacts_dir: str = typer.Option(str(DEFAULT_ARTIFACTS), "--artifacts", "-a"),
    log_level: str = typer.Option("INFO", "--log-level", "-l"),
) -> None:
    """Export pre-computed sweep metrics as JSON for the dashboard.

    Writes cells (model x variant metrics), stability scores, and
    per-variant agreement matrices to a single JSON file.
    """
    _setup(log_level)
    import json
    from dataclasses import asdict

    from .evaluation.metrics import compute_sweep_metrics
    from .storage import Storage

    storage = Storage(artifacts_dir)
    results = storage.get_run_results(run_id)

    if not results:
        console.print(f"[yellow]No results found for run {run_id}[/yellow]")
        raise typer.Exit(1)

    has_sweep_data = any(r.extraction_variant_id for r in results)
    if not has_sweep_data:
        console.print("[yellow]Run does not contain sweep data. Use a sweep run ID.[/yellow]")
        raise typer.Exit(1)

    sweep_data = compute_sweep_metrics(results)

    run_cfg = storage.get_run(run_id)
    dn = run_cfg.display_name if run_cfg else None

    # Serialize dataclass objects to dicts
    summary = {
        "run_id": run_id,
        "display_name": dn,
        "cells": [asdict(c) for c in sweep_data["cells"]],
        "stability": [asdict(s) for s in sweep_data["stability"]],
        "agreement_by_variant": sweep_data["agreement_by_variant"],
    }

    run_dir = storage.run_dir(run_id)
    out_path = Path(run_dir) / f"{run_id}_sweep_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    console.print(f"  [green]OK[/green] {out_path}")


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
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Date", justify="right")
        table.add_column("Models", justify="right")
        table.add_column("Videos", justify="right")
        table.add_column("Prompt")

        for r in runs:
            # Use display_name if set, otherwise derive from run_id
            if r.display_name:
                label = r.display_name
            elif r.run_id.startswith("run_") and len(r.run_id) == 16 and "_" not in r.run_id[4:]:
                label = "(legacy run)"
            else:
                label = r.run_id
            table.add_row(
                label,
                r.created_at[:16].replace("T", " "),
                f"{len(r.models)} model{'s' if len(r.models) != 1 else ''}",
                str(len(r.video_ids)),
                r.prompt_version,
            )
        console.print(table)
        console.print("\n[dim]Use 'vbench inspect-run <run_id>' for details.[/dim]")
    else:
        run_config = storage.get_run(run_id)
        if run_config is None:
            console.print(f"[red]Run not found: {run_id}[/red]")
            raise typer.Exit(1)

        if run_config.display_name:
            console.print(f"Name: [bold]{run_config.display_name}[/bold]")
        console.print(f"Run ID: [cyan]{run_config.run_id}[/cyan]")
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
    adapter: Optional[str] = typer.Option(None, "--adapter", help="Dataset adapter (ego4d, buildai, ucf101)"),
    data_dir: Optional[str] = typer.Option(None, "--data-dir", help="Data directory for dataset adapters"),
    manifest: Optional[str] = typer.Option(None, "--manifest", help="Manifest path for ego4d adapter"),
    categories: Optional[str] = typer.Option(None, "--categories", help="Comma-separated category filter (ucf101)"),
    limit_per_category: int = typer.Option(5, "--limit-per-category", help="Max clips per category (ucf101)"),
    artifacts_dir: str = typer.Option(str(DEFAULT_ARTIFACTS), "--artifacts", "-a"),
    log_level: str = typer.Option("INFO", "--log-level", "-l"),
) -> None:
    """List all ingested videos, or scan a dataset adapter's directory."""
    _setup(log_level)

    if adapter == "ucf101":
        from .adapters import UCF101Adapter

        effective_dir = data_dir or "."
        cat_list = [c.strip() for c in categories.split(",")] if categories else None
        ucf = UCF101Adapter(data_dir=effective_dir, categories=cat_list, limit_per_category=limit_per_category)
        entries = ucf.list_videos()
        if not entries:
            console.print("[yellow]No UCF101 videos found.[/yellow]")
            return

        # Group by category for display
        cats: dict[str, list] = {}
        for e in entries:
            cat = e.metadata["category"] if e.metadata else "unknown"
            cats.setdefault(cat, []).append(e)

        table = Table(title=f"UCF101 Videos ({len(entries)} clips, {len(cats)} categories)", show_lines=True)
        table.add_column("Category", style="cyan")
        table.add_column("Label")
        table.add_column("Clips", justify="right")

        for cat in sorted(cats):
            from .adapters.ucf101 import camel_to_label
            table.add_row(cat, camel_to_label(cat), str(len(cats[cat])))
        console.print(table)
        return

    if adapter == "buildai":
        from .adapters import BuildAIAdapter

        effective_dir = data_dir or "."
        ba = BuildAIAdapter(data_dir=effective_dir)
        entries = ba.list_videos()
        if not entries:
            console.print("[yellow]No videos found in Build.ai data directory.[/yellow]")
            return

        table = Table(title=f"Build.ai Videos ({len(entries)})", show_lines=True)
        table.add_column("Video ID", style="cyan")
        table.add_column("Path")
        table.add_column("Metadata")

        for e in entries:
            meta_str = ""
            if e.metadata:
                parts = []
                if "factory_id" in e.metadata:
                    parts.append(e.metadata["factory_id"])
                if "worker_id" in e.metadata:
                    parts.append(e.metadata["worker_id"])
                if "duration_sec" in e.metadata:
                    parts.append(f"{e.metadata['duration_sec']:.1f}s")
                meta_str = ", ".join(parts)
            table.add_row(e.video_id or "-", str(e.path.name), meta_str)
        console.print(table)
        return

    if adapter == "ego4d":
        from .adapters import Ego4DAdapter

        if not manifest:
            console.print("[red]--adapter ego4d requires --manifest[/red]")
            raise typer.Exit(1)
        effective_dir = data_dir or "."
        ego = Ego4DAdapter(manifest_path=manifest, video_dir=effective_dir)
        entries = ego.list_videos()
        if not entries:
            console.print("[yellow]No Ego4D videos found locally.[/yellow]")
            return

        table = Table(title=f"Ego4D Videos ({len(entries)})", show_lines=True)
        table.add_column("Video ID", style="cyan")
        table.add_column("Path")
        for e in entries:
            table.add_row(e.video_id or "-", str(e.path.name))
        console.print(table)
        return

    # Default: list from storage
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
def download_dataset(
    dataset: str = typer.Argument(..., help="Dataset to download (buildai-10k)"),
    output: str = typer.Option("./data", "--output", "-o", help="Output directory"),
    factory: str = typer.Option("factory_001", "--factory", help="Factory ID to download"),
    worker: str = typer.Option("worker_001", "--worker", help="Worker ID to download"),
    log_level: str = typer.Option("INFO", "--log-level", "-l"),
) -> None:
    """Download a dataset shard for benchmarking.

    Downloads a single worker's worth of clips (~50-100 videos) from the
    specified dataset. Use this to get started without downloading the
    full dataset.
    """
    _setup(log_level)

    if dataset != "buildai-10k":
        console.print(f"[red]Unknown dataset: {dataset}. Available: buildai-10k[/red]")
        raise typer.Exit(1)

    from .adapters.build_ai import download_buildai_shard

    console.print(f"Downloading {factory}/{worker} from builddotai/Egocentric-10K ...")
    try:
        tar_path = download_buildai_shard(
            output_dir=output,
            factory=factory,
            worker=worker,
        )
        console.print(f"  [green]OK[/green] Downloaded to {tar_path}")
        console.print(f"\nTo list videos: vbench list-videos --adapter buildai --data-dir {output}")
        console.print(f"To benchmark:   vbench run-benchmark {output} --adapter buildai --data-dir {output}")
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def accuracy_test(
    dataset: str = typer.Argument("ucf101", help="Dataset: ucf101"),
    data_dir: str = typer.Option("data/ucf101/UCF-101", "--data-dir", "-d",
        help="Path to dataset root (e.g. data/ucf101/UCF-101/)"),
    subset: int = typer.Option(50, "--subset", "-n",
        help="Number of clips to randomly sample"),
    categories: Optional[str] = typer.Option(None, "--categories",
        help="Comma-separated category filter (e.g. Hammering,Knitting)"),
    limit_per_category: int = typer.Option(5, "--limit-per-category",
        help="Max clips per category before sampling"),
    config_file: str = typer.Option(str(DEFAULT_CONFIGS / "benchmark_fast.yaml"), "--config", "-c"),
    models_file: str = typer.Option(str(DEFAULT_CONFIGS / "models.yaml"), "--models", "-m"),
    model_filter: Optional[str] = typer.Option(None, "--model-filter",
        help="Comma-separated model subset"),
    name: Optional[str] = typer.Option(None, "--name", help="Custom run name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show plan without running"),
    llm_judge: bool = typer.Option(False, "--llm-judge",
        help="Use LLM to score accuracy (~$0.001/pair)"),
    artifacts_dir: str = typer.Option(str(DEFAULT_ARTIFACTS), "--artifacts", "-a"),
    log_level: str = typer.Option("INFO", "--log-level", "-l"),
) -> None:
    """Run accuracy benchmark against a labeled dataset.

    Samples N clips, runs the benchmark pipeline, and scores each model's
    action labels against ground truth derived from the dataset structure.

    Requires the dataset to be downloaded first. For UCF101:
      Download from: https://www.crcv.ucf.edu/data/UCF101.php
      Extract to: data/ucf101/UCF-101/
    """
    _setup(log_level)
    import random

    from .caching import ResponseCache
    from .config import get_settings, load_benchmark_config, load_models_config

    if dataset != "ucf101":
        console.print(f"[red]Unknown dataset: {dataset}. Available: ucf101[/red]")
        raise typer.Exit(1)

    adapter = _make_ucf101_adapter(data_dir, data_dir, categories, limit_per_category)
    all_videos = adapter.list_videos()

    if not all_videos:
        console.print(f"[red]No videos found in {data_dir}[/red]")
        raise typer.Exit(1)

    # Sample subset
    sample_size = min(subset, len(all_videos))
    sample = random.sample(all_videos, sample_size)
    gt_labels = adapter.load_ground_truth()

    # Filter GT to sampled video IDs
    sampled_ids = {v.video_id for v in sample}
    gt_labels = [g for g in gt_labels if g.video_id in sampled_ids]

    cat_set = {v.metadata["category"] for v in sample if v.metadata}
    console.rule("[bold]Accuracy Test Plan[/bold]")
    console.print(f"  Dataset: [cyan]{dataset}[/cyan]")
    console.print(f"  Clips: [cyan]{sample_size}[/cyan] sampled from {len(all_videos)} total")
    console.print(f"  Categories: [cyan]{len(cat_set)}[/cyan] unique")
    console.print(f"  Ground truth labels: [cyan]{len(gt_labels)}[/cyan]")

    if dry_run:
        # Show sample of ground truth labels
        table = Table(title="Sample Ground Truth Labels", show_lines=True)
        table.add_column("Video ID", style="cyan")
        table.add_column("Action")
        table.add_column("Category")
        for v in sample[:10]:
            table.add_row(
                v.video_id or "-",
                v.metadata.get("ground_truth_action", "-") if v.metadata else "-",
                v.metadata.get("category", "-") if v.metadata else "-",
            )
        if sample_size > 10:
            table.add_row("...", f"({sample_size - 10} more)", "...")
        console.print(table)
        console.print("\n[yellow]Dry run — no API calls made.[/yellow]")
        raise typer.Exit(0)

    # Load configs and run the pipeline
    settings = get_settings()
    storage_mod = __import__("video_eval_harness.storage", fromlist=["Storage"])
    Storage = storage_mod.Storage
    storage = Storage(artifacts_dir)
    cache = ResponseCache()

    bench_cfg = load_benchmark_config(config_file)
    models_cfg = load_models_config(models_file)

    filter_list = [m.strip() for m in model_filter.split(",")] if model_filter else None

    # Build a temporary adapter that only returns the sampled clips
    class _SampledAdapter:
        def list_videos(self):
            return sample
        def load_ground_truth(self):
            return gt_labels
        def name(self):
            return "ucf101_sample"

    run_name = name or f"accuracy-{dataset}-{sample_size}clips"

    _run_single(
        bench_cfg, models_cfg, data_dir, settings, storage, cache,
        None, None, None, artifacts_dir, log_level,
        filter_models=filter_list,
        gt_labels=gt_labels,
        dataset_adapter=_SampledAdapter(),
        max_segments=None,
        run_name=run_name,
        use_llm_judge=llm_judge,
    )

    if llm_judge and gt_labels:
        results = storage.get_run_results(storage.list_runs()[0].run_id)
        _run_llm_judge(results)

    cache.close()


@app.command()
def version() -> None:
    """Show version."""
    console.print(f"video-eval-harness v{__version__}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
