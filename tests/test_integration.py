"""Integration tests for the full pipeline (no API key required).

Uses mocked provider responses to test the end-to-end flow:
ingest -> segment -> extract frames -> label -> evaluate -> export.
"""

from __future__ import annotations

import json

import pytest

from video_eval_harness.caching import ResponseCache
from video_eval_harness.config import (
    ModelConfig,
    SegmentationConfig,
)
from video_eval_harness.evaluation.metrics import (
    compute_agreement_matrix,
    compute_model_summary,
)
from video_eval_harness.evaluation.summaries import export_results, results_to_dataframe
from video_eval_harness.labeling import LabelingRunner
from video_eval_harness.labeling.normalization import parse_model_response
from video_eval_harness.prompting import PromptBuilder
from video_eval_harness.providers.base import BaseProvider, ProviderResponse
from video_eval_harness.schemas import (
    ExtractedFrames,
    RunConfig,
    SegmentLabelResult,
    VideoMetadata,
)
from video_eval_harness.segmentation import FixedWindowSegmenter
from video_eval_harness.storage import Storage
from video_eval_harness.utils.ids import generate_run_id

pytestmark = pytest.mark.integration


class MockProvider(BaseProvider):
    """Mock provider that returns deterministic JSON responses."""

    provider_name = "mock"

    def __init__(self, responses: dict[str, str] | None = None):
        self._responses = responses or {}
        self._default_response = json.dumps({
            "primary_action": "testing",
            "secondary_actions": ["verifying"],
            "description": "A test pattern video showing color bars and timecode",
            "objects": ["color bars", "timecode"],
            "environment_context": "synthetic test environment",
            "confidence": 0.85,
            "reasoning_summary_or_notes": "This is a test video pattern",
            "uncertainty_flags": [],
        })

    def complete(self, model_id, prompt, image_paths=None, max_tokens=2048, temperature=0.1):
        text = self._responses.get(model_id, self._default_response)
        return ProviderResponse(
            text=text,
            model=model_id,
            provider=self.provider_name,
            latency_ms=150.0,
            prompt_tokens=500,
            completion_tokens=200,
            total_tokens=700,
            estimated_cost=0.001,
            success=True,
        )

    def list_models(self):
        return [{"id": "mock/model-a"}, {"id": "mock/model-b"}]


@pytest.fixture
def tmp_artifacts(tmp_path):
    """Create a temporary artifacts directory."""
    return tmp_path / "artifacts"


@pytest.fixture
def storage(tmp_artifacts):
    """Create a storage instance with temp directory."""
    return Storage(str(tmp_artifacts))


@pytest.fixture
def sample_video():
    """Create a mock VideoMetadata."""
    return VideoMetadata(
        video_id="vid_test_integration_abc123",
        source_path="/fake/test_video.mp4",
        filename="test_video.mp4",
        duration_s=30.0,
        width=1920,
        height=1080,
        fps=30.0,
        codec="h264",
        file_size_bytes=1_000_000,
    )


@pytest.fixture
def sample_segments(sample_video):
    """Create segments for the sample video."""
    cfg = SegmentationConfig(window_size_s=10.0)
    segmenter = FixedWindowSegmenter(cfg)
    return segmenter.segment(sample_video)


@pytest.fixture
def sample_frames(sample_segments, tmp_artifacts):
    """Create mock extracted frames (fake image files)."""
    frames_map = {}
    for seg in sample_segments:
        frame_dir = tmp_artifacts / "frames" / seg.video_id / seg.segment_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        timestamps = []
        for i in range(4):
            fp = frame_dir / f"frame_{i:03d}.jpg"
            fp.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # minimal JPEG header
            paths.append(str(fp))
            ts = seg.start_time_s + (i + 1) * seg.duration_s / 5
            timestamps.append(round(ts, 3))

        frames_map[seg.segment_id] = ExtractedFrames(
            segment_id=seg.segment_id,
            video_id=seg.video_id,
            frame_paths=paths,
            frame_timestamps_s=timestamps,
            num_frames=4,
        )
    return frames_map


@pytest.fixture
def mock_models():
    """Create mock model configs."""
    return {
        "model-a": ModelConfig(
            name="model-a",
            model_id="mock/model-a",
            provider="mock",
            max_tokens=1024,
            temperature=0.1,
            supports_images=True,
        ),
        "model-b": ModelConfig(
            name="model-b",
            model_id="mock/model-b",
            provider="mock",
            max_tokens=1024,
            temperature=0.1,
            supports_images=True,
        ),
    }


class TestSegmentation:
    def test_fixed_window_30s_video(self, sample_video):
        cfg = SegmentationConfig(window_size_s=10.0)
        segmenter = FixedWindowSegmenter(cfg)
        segments = segmenter.segment(sample_video)

        assert len(segments) == 3
        assert segments[0].start_time_s == 0.0
        assert segments[0].end_time_s == 10.0
        assert segments[1].start_time_s == 10.0
        assert segments[2].end_time_s == 30.0

    def test_overlap_segmentation(self, sample_video):
        cfg = SegmentationConfig(window_size_s=10.0, stride_s=5.0)
        segmenter = FixedWindowSegmenter(cfg)
        segments = segmenter.segment(sample_video)

        assert len(segments) >= 4  # more segments with overlap
        assert segments[0].start_time_s == 0.0
        assert segments[1].start_time_s == 5.0


class TestStorage:
    def test_save_and_retrieve_video(self, storage, sample_video):
        storage.save_video(sample_video)
        retrieved = storage.get_video(sample_video.video_id)
        assert retrieved is not None
        assert retrieved.video_id == sample_video.video_id
        assert retrieved.duration_s == 30.0

    def test_save_and_retrieve_segments(self, storage, sample_video, sample_segments):
        storage.save_video(sample_video)
        storage.save_segments(sample_segments)
        retrieved = storage.get_segments(sample_video.video_id)
        assert len(retrieved) == len(sample_segments)

    def test_save_and_retrieve_frames(self, storage, sample_segments, sample_frames):
        for seg_id, frames in sample_frames.items():
            storage.save_extracted_frames(frames)
        for seg in sample_segments:
            retrieved = storage.get_extracted_frames(seg.segment_id)
            assert retrieved is not None
            assert retrieved.num_frames == 4

    def test_save_and_retrieve_run(self, storage):
        run_id = generate_run_id()
        run_config = RunConfig(
            run_id=run_id,
            models=["model-a", "model-b"],
            prompt_version="concise",
        )
        storage.save_run(run_config)
        retrieved = storage.get_run(run_id)
        assert retrieved is not None
        assert retrieved.run_id == run_id

    def test_list_runs(self, storage):
        for _ in range(3):
            run_id = generate_run_id()
            storage.save_run(RunConfig(run_id=run_id, models=["a"]))
        runs = storage.list_runs()
        assert len(runs) == 3

    def test_resume_support(self, storage):
        result = SegmentLabelResult(
            run_id="run_test",
            video_id="vid_test",
            segment_id="seg_test",
            start_time_s=0.0,
            end_time_s=10.0,
            model_name="gpt4o",
            provider="mock",
            parsed_success=True,
            primary_action="testing",
        )
        storage.save_label_result(result)
        assert storage.has_result("run_test", "seg_test", "gpt4o") is True
        assert storage.has_result("run_test", "seg_test", "other") is False


class TestPromptBuilder:
    def test_concise_template(self, sample_segments):
        builder = PromptBuilder()
        prompt = builder.build("concise", sample_segments[0], num_frames=4)
        assert "4 frames" in prompt
        assert "0.0" in prompt  # start time

    def test_rich_template(self, sample_segments):
        builder = PromptBuilder()
        prompt = builder.build("rich", sample_segments[0], num_frames=8)
        assert "egocentric" in prompt.lower()

    def test_strict_json_template(self, sample_segments):
        builder = PromptBuilder()
        prompt = builder.build("strict_json", sample_segments[0], num_frames=4)
        assert "JSON" in prompt

    def test_unknown_template_raises(self, sample_segments):
        builder = PromptBuilder()
        with pytest.raises(ValueError, match="Unknown template"):
            builder.build("nonexistent", sample_segments[0], num_frames=4)

    def test_custom_template(self, sample_segments):
        builder = PromptBuilder(templates={
            "custom": "Analyze {{ num_frames }} frames from {{ start_time }}s"
        })
        prompt = builder.build("custom", sample_segments[0], num_frames=6)
        assert "6 frames" in prompt


class TestNormalization:
    def test_parse_valid_response(self):
        result = parse_model_response(
            raw_text='{"primary_action": "cooking", "confidence": 0.9}',
            run_id="run_test",
            video_id="vid_test",
            segment_id="seg_test",
            start_time_s=0.0,
            end_time_s=10.0,
            model_name="model-a",
            provider="mock",
            latency_ms=100.0,
        )
        assert result.parsed_success is True
        assert result.primary_action == "cooking"

    def test_parse_with_error(self):
        result = parse_model_response(
            raw_text="",
            run_id="run_test",
            video_id="vid_test",
            segment_id="seg_test",
            start_time_s=0.0,
            end_time_s=10.0,
            model_name="model-a",
            provider="mock",
            latency_ms=100.0,
            error="API rate limited",
        )
        assert result.parsed_success is False
        assert "Provider error" in result.parse_error


class TestLabelingRunner:
    def test_full_labeling_run(
        self, storage, sample_video, sample_segments, sample_frames, mock_models
    ):
        storage.save_video(sample_video)
        storage.save_segments(sample_segments)

        mock_provider = MockProvider()
        cache = ResponseCache(cache_dir=str(storage.artifacts_dir / "test_cache"))

        runner = LabelingRunner(
            providers={"mock": mock_provider},
            models=mock_models,
            prompt_builder=PromptBuilder(),
            storage=storage,
            cache=cache,
            prompt_version="concise",
            max_concurrency=2,
        )

        run_id = generate_run_id()
        storage.save_run(RunConfig(
            run_id=run_id,
            models=list(mock_models.keys()),
            prompt_version="concise",
        ))

        results = runner.run(
            run_id=run_id,
            segments=sample_segments,
            frames_map=sample_frames,
            model_names=list(mock_models.keys()),
        )

        # 3 segments x 2 models = 6 results
        assert len(results) == 6
        assert all(r.parsed_success for r in results)
        assert all(r.primary_action == "testing" for r in results)
        cache.close()

    def test_resume_skips_existing(
        self, storage, sample_video, sample_segments, sample_frames, mock_models
    ):
        storage.save_video(sample_video)
        storage.save_segments(sample_segments)

        run_id = generate_run_id()
        storage.save_run(RunConfig(run_id=run_id, models=["model-a"]))

        # Pre-save one result
        existing = SegmentLabelResult(
            run_id=run_id,
            video_id=sample_video.video_id,
            segment_id=sample_segments[0].segment_id,
            start_time_s=0.0,
            end_time_s=10.0,
            model_name="model-a",
            provider="mock",
            primary_action="pre-existing",
            parsed_success=True,
        )
        storage.save_label_result(existing)

        mock_provider = MockProvider()
        cache = ResponseCache(cache_dir=str(storage.artifacts_dir / "test_cache2"))

        runner = LabelingRunner(
            providers={"mock": mock_provider},
            models={"model-a": mock_models["model-a"]},
            prompt_builder=PromptBuilder(),
            storage=storage,
            cache=cache,
            prompt_version="concise",
        )

        results = runner.run(
            run_id=run_id,
            segments=sample_segments,
            frames_map=sample_frames,
            model_names=["model-a"],
        )

        # 3 results total (1 pre-existing + 2 new)
        assert len(results) == 3
        # The pre-existing result should still be there
        seg0_result = [r for r in results if r.segment_id == sample_segments[0].segment_id][0]
        assert seg0_result.primary_action == "pre-existing"
        cache.close()


class TestEvaluation:
    def _make_results(self):
        results = []
        for model in ["model-a", "model-b"]:
            for i in range(5):
                results.append(SegmentLabelResult(
                    run_id="run_eval_test",
                    video_id="vid_test",
                    segment_id=f"seg_{i:04d}",
                    start_time_s=i * 10.0,
                    end_time_s=(i + 1) * 10.0,
                    model_name=model,
                    provider="mock",
                    primary_action="cooking" if model == "model-a" else ("cooking" if i < 3 else "eating"),
                    description=f"Test description {i}",
                    confidence=0.9 if model == "model-a" else 0.75,
                    parsed_success=True,
                    latency_ms=100.0 + i * 10,
                    estimated_cost=0.001,
                ))
        return results

    def test_model_summary(self):
        results = self._make_results()
        summary = compute_model_summary(results, "model-a")
        assert summary.total_segments == 5
        assert summary.successful_parses == 5
        assert summary.parse_success_rate == 1.0
        assert summary.avg_confidence == 0.9
        assert summary.total_estimated_cost == 0.005

    def test_agreement_matrix(self):
        results = self._make_results()
        agreement = compute_agreement_matrix(results)
        assert agreement["model-a"]["model-a"] == 1.0
        assert agreement["model-b"]["model-b"] == 1.0
        # model-a always says "cooking", model-b says "cooking" 3/5 times
        assert agreement["model-a"]["model-b"] == pytest.approx(0.6)

    def test_results_to_dataframe(self):
        results = self._make_results()
        df = results_to_dataframe(results)
        assert len(df) == 10
        assert "model_name" in df.columns
        assert "primary_action" in df.columns

    def test_export_csv_parquet(self, tmp_path):
        results = self._make_results()
        paths = export_results(results, str(tmp_path), "run_eval_test", ["csv", "parquet"])
        assert len(paths) == 2
        assert any(str(p).endswith(".csv") for p in paths)
        assert any(str(p).endswith(".parquet") for p in paths)
        # Verify CSV is readable
        import pandas as pd
        df = pd.read_csv(paths[0])
        assert len(df) == 10


class TestCaching:
    def test_cache_set_get(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))
        key = cache.make_key("model-a", "prompt123", "input456")
        cache.set(key, '{"primary_action": "cached"}')
        result = cache.get(key)
        assert result == '{"primary_action": "cached"}'
        cache.close()

    def test_cache_miss(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))
        result = cache.get("nonexistent_key")
        assert result is None
        cache.close()

    def test_hash_content(self, tmp_path):
        cache = ResponseCache(cache_dir=str(tmp_path / "cache"))
        h1 = cache.hash_content("hello")
        h2 = cache.hash_content("hello")
        h3 = cache.hash_content("world")
        assert h1 == h2
        assert h1 != h3
        cache.close()


class TestMockProvider:
    def test_mock_provider_returns_valid_json(self):
        provider = MockProvider()
        response = provider.complete("mock/model-a", "test prompt")
        assert response.success is True
        assert response.latency_ms > 0
        data = json.loads(response.text)
        assert "primary_action" in data

    def test_mock_provider_list_models(self):
        provider = MockProvider()
        models = provider.list_models()
        assert len(models) == 2
