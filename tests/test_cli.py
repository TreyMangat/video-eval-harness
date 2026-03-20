from __future__ import annotations

from pathlib import Path
import re
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from video_eval_harness import __version__
from video_eval_harness.cli import app
from video_eval_harness.config import (
    AppSettings,
    BenchmarkConfig,
    ExtractionConfig,
    ModelConfig,
    SegmentationConfig,
)
from video_eval_harness.schemas import RunConfig, VideoMetadata
from video_eval_harness.storage import Storage
from video_eval_harness.sweep import SweepAxis, SweepConfig

runner = CliRunner()


class FakeStorage:
    instances: list["FakeStorage"] = []

    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = Path(artifacts_dir)
        self.saved_videos: list[VideoMetadata] = []
        self.saved_segments: list[object] = []
        self.saved_runs: list[RunConfig] = []
        FakeStorage.instances.append(self)

    def save_video(self, video: VideoMetadata) -> None:
        self.saved_videos.append(video)

    def save_segments(self, segments: list[object]) -> None:
        self.saved_segments.extend(segments)

    def save_run(self, run: RunConfig) -> None:
        self.saved_runs.append(run)


class FakeCache:
    instances: list["FakeCache"] = []

    def __init__(self) -> None:
        self.closed = False
        FakeCache.instances.append(self)

    def close(self) -> None:
        self.closed = True


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


def _configure_cli_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_fake_storage: bool = False,
) -> dict[str, int]:
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
    provider_calls = {"count": 0}

    def track_setup_providers(*_args, **_kwargs) -> dict:
        provider_calls["count"] += 1
        return {}

    monkeypatch.setattr("video_eval_harness.config.load_sweep_config", lambda *_: sweep)
    monkeypatch.setattr("video_eval_harness.config.load_benchmark_config", lambda *_: benchmark)
    monkeypatch.setattr("video_eval_harness.config.load_models_config", lambda *_: _mock_models())
    monkeypatch.setattr("video_eval_harness.config.validate_run_config", lambda *_, **__: [])
    monkeypatch.setattr("video_eval_harness.config.setup_providers", track_setup_providers)
    monkeypatch.setattr(
        "video_eval_harness.config.get_settings",
        lambda: AppSettings(
            openrouter_api_key="test-key",
            vbench_artifacts_dir="./artifacts",
            ffmpeg_path="ffmpeg",
        ),
    )
    monkeypatch.setattr("video_eval_harness.utils.ffmpeg.probe_video", lambda *_: _mock_probe())

    if use_fake_storage:
        FakeStorage.instances.clear()
        FakeCache.instances.clear()
        monkeypatch.setattr("video_eval_harness.storage.Storage", FakeStorage)
        monkeypatch.setattr("video_eval_harness.caching.ResponseCache", FakeCache)

    return provider_calls


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert f"video-eval-harness v{__version__}" in clean_output


def test_list_videos_empty_db(tmp_path) -> None:
    result = runner.invoke(app, ["list-videos", "--artifacts", str(tmp_path / "artifacts")])

    assert result.exit_code == 0
    assert "No videos ingested yet." in result.output


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


def test_inspect_run_no_runs(tmp_path) -> None:
    result = runner.invoke(app, ["inspect-run", "--artifacts", str(tmp_path / "artifacts")])

    assert result.exit_code == 0
    assert "No runs found." in result.output


def test_inspect_run_missing_run(tmp_path) -> None:
    result = runner.invoke(app, ["inspect-run", "missing-run", "--artifacts", str(tmp_path / "artifacts")])

    assert result.exit_code == 1
    assert "Run not found: missing-run" in result.output


def test_inspect_run_lists_specific_run(tmp_path) -> None:
    storage = Storage(str(tmp_path / "artifacts"))
    run = RunConfig(run_id="run_cli_test", models=["model-a"], prompt_version="concise")
    storage.save_run(run)

    result = runner.invoke(app, ["inspect-run", "run_cli_test", "--artifacts", str(storage.artifacts_dir)])

    assert result.exit_code == 0
    assert "run_cli_test" in result.output
    assert "model-a" in result.output


def test_run_benchmark_sweep_dry_run(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    provider_calls = _configure_cli_dry_run(monkeypatch, use_fake_storage=True)
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video")

    result = runner.invoke(
        app,
        [
            "run-benchmark",
            str(video_path),
            "--sweep",
            "--dry-run",
            "--artifacts",
            str(tmp_path / "artifacts"),
        ],
    )

    assert result.exit_code == 0
    assert "Dry run" in result.output
    assert "Model x Variant Matrix" in result.output
    assert provider_calls["count"] == 0
    assert len(FakeStorage.instances) == 1
    assert len(FakeStorage.instances[0].saved_videos) == 1
    assert len(FakeStorage.instances[0].saved_segments) > 0
    assert FakeStorage.instances[0].saved_runs == []
    assert len(FakeCache.instances) == 1
    assert FakeCache.instances[0].closed is True


def test_run_benchmark_sweep_dry_run_respects_cli_axis_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _configure_cli_dry_run(monkeypatch, use_fake_storage=True)
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video")

    result = runner.invoke(
        app,
        [
            "run-benchmark",
            str(video_path),
            "--sweep",
            "--dry-run",
            "--frames",
            "4",
            "--methods",
            "keyframe",
            "--artifacts",
            str(tmp_path / "artifacts"),
        ],
    )

    assert result.exit_code == 0
    assert "keyframe_4f" in result.output
    assert "uniform_8f" not in result.output


def test_sweep_command_dry_run(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    provider_calls = _configure_cli_dry_run(monkeypatch, use_fake_storage=True)
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"fake video")

    result = runner.invoke(
        app,
        [
            "sweep",
            str(video_path),
            "--dry-run",
            "--artifacts",
            str(tmp_path / "artifacts"),
        ],
    )

    assert result.exit_code == 0
    assert "Extraction Variants" in result.output
    assert "Dry run" in result.output
    assert provider_calls["count"] == 0
