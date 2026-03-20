"""Modal backend for the public video benchmark dashboard.

Deploy with:
  modal deploy deploy/modal/app.py

Serve locally with:
  modal serve deploy/modal/app.py
"""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import modal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"

app = modal.App("vbench-video-eval")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "fastapi>=0.115",
        "httpx>=0.25",
        "pydantic>=2.0",
        "pydantic-settings>=2.0",
        "pyyaml>=6.0",
        "tenacity>=8.2",
        "pillow>=10.0",
        "opencv-python-headless>=4.8",
        "numpy>=1.24",
        "pandas>=2.0",
        "pyarrow>=14.0",
        "jinja2>=3.1",
        "diskcache>=5.6",
        "python-dotenv>=1.0",
    )
    .add_local_dir(str(SRC_DIR), remote_path="/root/src")
    .env({"PYTHONPATH": "/root/src"})
)

volume = modal.Volume.from_name("vbench-artifacts", create_if_missing=True)
ARTIFACTS_PATH = "/data/artifacts"

api = FastAPI(title="VBench Public API", version="0.2.0")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Every model gets identical input and is compared head-to-head.
# This is a benchmark, not a pipeline — no role assignments.
DEFAULT_MODELS = {
    "gemini-3.1-pro": {
        "model_id": "google/gemini-3.1-pro-preview",
        "provider": "openrouter",
        "supports_images": True,
        "notes": "Frontier reasoning model with multimodal support, 1M-token context",
    },
    "gpt-5.4": {
        "model_id": "openai/gpt-5.4",
        "provider": "openrouter",
        "supports_images": True,
        "notes": "OpenAI latest frontier model with 1M+ context, text and image inputs",
    },
    "qwen3.5-vl": {
        "model_id": "qwen/qwen3.5-397b-a17b",
        "provider": "openrouter",
        "supports_images": True,
        "notes": "Native vision-language model, hybrid MoE activating 17B params",
    },
    "claude-sonnet-4.6": {
        "model_id": "anthropic/claude-sonnet-4.6",
        "provider": "openrouter",
        "supports_images": True,
        "notes": "Strong multimodal reasoning from Anthropic",
    },
}


class BenchmarkRequest(BaseModel):
    """Request payload for launching a benchmark job."""

    video_url: str
    video_name: str = "video.mp4"
    models: list[str] = Field(default_factory=lambda: ["gemini-3.1-pro", "gpt-5.4", "qwen3.5-vl", "claude-sonnet-4.6"])
    segmentation_mode: str = "fixed_window"
    window_size: float = 10.0
    stride: Optional[float] = None
    num_frames: int = 8
    prompt_version: str = "concise"
    max_concurrency: int = 2


@app.function(
    image=image,
    volumes={ARTIFACTS_PATH: volume},
    secrets=[modal.Secret.from_name("vbench-api-keys")],
    timeout=60 * 20,
    memory=2048,
)
def process_video(request: dict[str, Any]) -> dict[str, Any]:
    """Download a video, segment it, label it, and persist the run."""
    import httpx

    from video_eval_harness.caching import ResponseCache
    from video_eval_harness.config import ExtractionConfig, ModelConfig, SegmentationConfig, setup_providers, AppSettings
    from video_eval_harness.extraction import FrameExtractor
    from video_eval_harness.labeling import LabelingRunner
    from video_eval_harness.prompting import PromptBuilder
    from video_eval_harness.schemas import RunConfig, VideoMetadata
    from video_eval_harness.segmentation import build_segmenter
    from video_eval_harness.storage import Storage
    from video_eval_harness.utils.ffmpeg import probe_video
    from video_eval_harness.utils.ids import generate_run_id, generate_video_id

    payload = BenchmarkRequest.model_validate(request)
    suffix = Path(payload.video_name).suffix or ".mp4"
    video_path: Optional[Path] = None

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        with httpx.stream("GET", payload.video_url, timeout=180) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                tmp.write(chunk)
        video_path = Path(tmp.name)

    try:
        storage = Storage(ARTIFACTS_PATH)
        cache = ResponseCache(cache_dir=f"{ARTIFACTS_PATH}/cache")

        info = probe_video(video_path)
        video_id = generate_video_id(video_path)
        meta = VideoMetadata(
            video_id=video_id,
            source_path=str(video_path),
            filename=payload.video_name,
            duration_s=info.duration_s,
            width=info.width,
            height=info.height,
            fps=info.fps,
            codec=info.codec,
            file_size_bytes=info.file_size_bytes,
        )
        storage.save_video(meta)

        seg_cfg = SegmentationConfig(
            mode=payload.segmentation_mode,
            window_size_s=payload.window_size,
            stride_s=payload.stride,
        )
        segments = build_segmenter(seg_cfg).segment(meta)
        storage.save_segments(segments)

        ext_cfg = ExtractionConfig(num_frames=payload.num_frames)
        extractor = FrameExtractor(ext_cfg, storage)
        frames_map = {}
        for seg in segments:
            frames_map[seg.segment_id] = extractor.extract(seg, video_path)

        model_configs: dict[str, ModelConfig] = {}
        for model_name in payload.models:
            model_meta = DEFAULT_MODELS.get(model_name)
            if model_meta is None:
                raise ValueError(f"Unknown model '{model_name}'. Available models: {sorted(DEFAULT_MODELS)}")
            model_configs[model_name] = ModelConfig(
                name=model_name,
                model_id=model_meta["model_id"],
                provider=model_meta["provider"],
                max_tokens=2048,
                temperature=0.1,
                supports_images=bool(model_meta.get("supports_images", True)),
                notes=model_meta.get("notes"),
            )

        settings = AppSettings(
            openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
            vbench_max_concurrency=payload.max_concurrency,
        )
        providers = setup_providers(model_configs, settings)

        run_id = generate_run_id()
        storage.save_run(
            RunConfig(
                run_id=run_id,
                models=payload.models,
                prompt_version=payload.prompt_version,
                segmentation_mode=seg_cfg.mode,
                segmentation_config=seg_cfg.model_dump(),
                extraction_config=ext_cfg.model_dump(),
                model_configs={name: model_configs[name].model_dump() for name in payload.models},
                video_ids=[video_id],
                notes=f"source_url={payload.video_url}",
            )
        )

        runner = LabelingRunner(
            providers=providers,
            models=model_configs,
            prompt_builder=PromptBuilder(),
            storage=storage,
            cache=cache,
            prompt_version=payload.prompt_version,
            max_concurrency=payload.max_concurrency,
        )
        runner.run(run_id, segments, frames_map, payload.models)
        volume.commit()
        return _build_run_payload(storage, run_id)
    finally:
        if "cache" in locals():
            cache.close()
        if video_path is not None and video_path.exists():
            video_path.unlink()


@app.function(
    image=image,
    volumes={ARTIFACTS_PATH: volume},
    timeout=60,
)
def get_run_payload(run_id: str) -> dict[str, Any]:
    """Fetch a persisted run payload."""
    from video_eval_harness.storage import Storage

    volume.reload()
    storage = Storage(ARTIFACTS_PATH)
    return _build_run_payload(storage, run_id)


@app.function(
    image=image,
    volumes={ARTIFACTS_PATH: volume},
    timeout=30,
)
def get_segment_media(run_id: str, segment_id: str) -> dict[str, Any]:
    """Return frontend-safe media previews for a segment."""
    from video_eval_harness.storage import Storage

    volume.reload()
    storage = Storage(ARTIFACTS_PATH)
    frames = storage.get_extracted_frames(segment_id)
    segment = storage.get_segment(segment_id)

    if segment is None or frames is None:
        raise ValueError(f"Segment '{segment_id}' not found in run '{run_id}'")

    return {
        "run_id": run_id,
        "segment_id": segment_id,
        "start_time_s": segment.start_time_s,
        "end_time_s": segment.end_time_s,
        "frame_timestamps_s": frames.frame_timestamps_s,
        "contact_sheet_data_url": _file_to_data_url(frames.contact_sheet_path),
        "frames": [
            {
                "timestamp_s": ts,
                "data_url": _file_to_data_url(path),
            }
            for path, ts in zip(frames.frame_paths, frames.frame_timestamps_s)
        ],
    }


@app.function(
    image=image,
    volumes={ARTIFACTS_PATH: volume},
    timeout=30,
)
def list_all_runs() -> list[dict[str, Any]]:
    """List run configs for the public dashboard."""
    from video_eval_harness.storage import Storage

    volume.reload()
    storage = Storage(ARTIFACTS_PATH)
    return [run.model_dump() for run in storage.list_runs()]


@api.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@api.get("/models")
def models() -> dict[str, list[dict[str, Any]]]:
    model_list = []
    for name, meta in DEFAULT_MODELS.items():
        model_list.append({"name": name, **meta})
    return {"models": model_list}


@api.get("/runs")
def runs() -> list[dict[str, Any]]:
    return list_all_runs.remote()


@api.get("/runs/{run_id}")
def run_details(run_id: str) -> dict[str, Any]:
    try:
        return get_run_payload.remote(run_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.get("/runs/{run_id}/segments/{segment_id}/media")
def segment_media(run_id: str, segment_id: str) -> dict[str, Any]:
    try:
        return get_segment_media.remote(run_id, segment_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.post("/benchmarks")
def submit_benchmark(request: BenchmarkRequest) -> dict[str, str]:
    call = process_video.spawn(request.model_dump())
    return {
        "call_id": call.object_id,
        "status": "queued",
    }


@api.get("/benchmarks/jobs/{call_id}")
def benchmark_job(call_id: str):
    try:
        function_call = modal.FunctionCall.from_id(call_id)
        result = function_call.get(timeout=0)
        return {
            "call_id": call_id,
            "status": "completed",
            "result": result,
        }
    except TimeoutError:
        return JSONResponse(
            status_code=202,
            content={
                "call_id": call_id,
                "status": "running",
            },
        )
    except Exception as exc:
        detail = str(exc)
        if "expired" in detail.lower():
            raise HTTPException(status_code=404, detail=detail) from exc
        return JSONResponse(
            status_code=500,
            content={
                "call_id": call_id,
                "status": "failed",
                "error": detail,
            },
        )


@app.function(image=image)
@modal.asgi_app()
def public_api():
    """Serve the FastAPI app."""
    return api


def _build_run_payload(storage, run_id: str) -> dict[str, Any]:
    from video_eval_harness.evaluation.metrics import compute_agreement_matrix, compute_model_summary

    run_config = storage.get_run(run_id)
    if run_config is None:
        raise ValueError(f"No run found for '{run_id}'")

    results = storage.get_run_results(run_id)
    segments = storage.get_run_segments(run_id)
    videos = {
        video_id: storage.get_video(video_id)
        for video_id in run_config.video_ids
    }

    models = sorted({r.model_name for r in results}) or sorted(run_config.models)
    summaries = {model: compute_model_summary(results, model).model_dump() for model in models}
    agreement = compute_agreement_matrix(results) if results else {}

    segment_items = []
    for segment in segments:
        frames = storage.get_extracted_frames(segment.segment_id)
        video_meta = videos.get(segment.video_id)
        segment_items.append(
            {
                "segment_id": segment.segment_id,
                "video_id": segment.video_id,
                "video_filename": video_meta.filename if video_meta else None,
                "segment_index": segment.segment_index,
                "start_time_s": segment.start_time_s,
                "end_time_s": segment.end_time_s,
                "duration_s": segment.duration_s,
                "segmentation_mode": segment.segmentation_mode.value,
                "frame_count": frames.num_frames if frames else 0,
                "frame_timestamps_s": frames.frame_timestamps_s if frames else [],
                "has_contact_sheet": bool(frames and frames.contact_sheet_path and Path(frames.contact_sheet_path).exists()),
            }
        )

    return {
        "run_id": run_id,
        "config": run_config.model_dump(),
        "models": models,
        "videos": [video.model_dump() for video in videos.values() if video is not None],
        "summaries": summaries,
        "agreement": agreement,
        "segments": segment_items,
        "results": [result.model_dump() for result in results],
    }


def _file_to_data_url(path_str: Optional[str]) -> Optional[str]:
    if not path_str:
        return None

    path = Path(path_str)
    if not path.exists():
        return None

    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"
