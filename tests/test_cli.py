from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from video_eval_harness.cli import app
from video_eval_harness.config import AppSettings, BenchmarkConfig, ExtractionConfig, ModelConfig, SegmentationConfig
from video_eval_harness.schemas import RunConfig, VideoMetadata
from video_eval_harness.sweep import SweepAxis, SweepConfig
from video_eval_harness.storage import Storage

runner = CliRunner()


def _mock_models() -> dict[str, ModelConfig]:
    return {
        "model-a": ModelConfig(name="model-a", provider="openrouter", model_id="openai/model-a"),
        "model-b": ModelConfig(name="model-b", provider="openrouter", model_id="openai/model-b"),
    }


def _mock_probe() -> SimpleNamespace:
    return SimpleNamespace(
        duration_s=25.0,
        width=1280,
        height=720,
        fps=30.0,
        codec="h264",
        file_size_bytes=1024,
    )


def _configure_cli_dry_run(monkeypatch) -> None:
    benchmark = BenchmarkConfig(
        name="test-benchmark",
        models=["model-a", "model-b"],
        prompt_version="concise",
        segmentation=SegmentationConfig(window_size_s=10.0, min_segment_s=2.0),
        extraction=ExtractionConfig(num_frames=8, method="uniform"),
    )
    sweep = SweepConfig(
        benchmark=benchmark,
        axis=SweepAxis(num_frames=[4, 8], methods=["uniform", "keyframe"]),
    )

    monkeypatch.setattr("video_eval_harness.config.load_sweep_config", lambda *_: sweep)
    monkeypatch.setattr("video_eval_harness.config.load_benchmark_config", lambda *_: benchmark)
    monkeypatch.setattr("video_eval_harness.config.load_models_config", lambda *_: _mock_models())
    monkeypatch.setattr("video_eval_harness.config.validate_run_config", lambda *_, **__: [])
    monkeypatch.setattr(
        "video_eval_harness.config.get_settings",
        lambda: AppSettings(openrouter_api_key="test-key", vbench_artifacts_dir="./artifacts", ffmpeg_path="ffmpeg"),
    )
    monkeypatch.setattr("video_eval_harness.utils.ffmpeg.probe_video", lambda *_: _mock_probe())


def test_run_benchmark_sweep_dry_run(monkeypatch, tmp_path) -> None:
    _configure_cli_dry_run(monkeypatch)
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video")
    artifacts_dir = tmp_path / "artifacts"

    result = runner.invoke(
        app,
        ["run-benchmark", str(video_path), "--sweep", "--dry-run", "--artifacts", str(artifacts_dir)],
    )

    assert result.exit_code == 0
    assert "Dry run" in result.output
    assert "Model x Variant Matrix" in result.output


def test_sweep_command_dry_run(monkeypatch, tmp_path) -> None:
    _configure_cli_dry_run(monkeypatch)
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video")
    artifacts_dir = tmp_path / "artifacts"

    result = runner.invoke(
        app,
        ["sweep", str(video_path), "--dry-run", "--artifacts", str(artifacts_dir)],
    )

    assert result.exit_code == 0
    assert "Extraction Variants" in result.output
    assert "Dry run" in result.output


def test_inspect_run_lists_specific_run(tmp_path) -> None:
    storage = Storage(str(tmp_path / "artifacts"))
    run = RunConfig(run_id="run_cli_test", models=["model-a"], prompt_version="concise")
    storage.save_run(run)

    result = runner.invoke(app, ["inspect-run", "run_cli_test", "--artifacts", str(storage.artifacts_dir)])

    assert result.exit_code == 0
    assert "run_cli_test" in result.output
    assert "model-a" in result.output


def test_list_videos_shows_saved_video(tmp_path) -> None:
    storage = Storage(str(tmp_path / "artifacts"))
    storage.save_video(
        VideoMetadata(
            video_id="vid_cli_test",
            source_path=str(Path(tmp_path / "video.mp4")),
            filename="video.mp4",
            duration_s=25.0,
            width=1280,
            height=720,
            fps=30.0,
            codec="h264",
        )
    )

    result = runner.invoke(app, ["list-videos", "--artifacts", str(storage.artifacts_dir)])

    assert result.exit_code == 0
    assert "vid_cli_test" in result.output
    assert "video.mp4" in result.output
