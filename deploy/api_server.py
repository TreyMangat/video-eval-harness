"""FastAPI backend for the public VBench dashboard.

Run locally with:
  uvicorn deploy.api_server:create_app --factory --reload --port 8000
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Optional, TypedDict

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from video_eval_harness.storage import Storage
from video_eval_harness.utils.ffmpeg import probe_video

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS_DIR = Path(os.environ.get("VBENCH_ARTIFACTS_DIR", str(ROOT / "artifacts")))
DEFAULT_RUNS_DIR = "runs"
UPLOADS_DIR = "uploads"
ALLOWED_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ESTIMATED_TIME_S = 90

try:
    from video_eval_harness.limits import PUBLIC_LIMITS, validate_public_request
except Exception:  # pragma: no cover - fallback only used if limits.py is unavailable
    PUBLIC_LIMITS = {
        "max_clip_duration_s": 60,
        "max_file_size_mb": 100,
        "max_segments": 6,
        "max_models": 3,
        "allowed_models": ["gemini-3-flash", "gpt-5.4-mini", "qwen3.5-27b"],
    }

    def validate_public_request(
        file_size_bytes: int,
        duration_s: float,
        requested_models: list[str],
        num_frames: int = 8,
    ) -> tuple[bool, Optional[str]]:
        max_file_size_mb = int(PUBLIC_LIMITS["max_file_size_mb"])
        max_clip_duration_s = int(PUBLIC_LIMITS["max_clip_duration_s"])
        max_models = int(PUBLIC_LIMITS["max_models"])
        allowed_models = {
            str(model_name) for model_name in PUBLIC_LIMITS["allowed_models"]
        }
        if file_size_bytes > max_file_size_mb * 1024 * 1024:
            return False, f"File too large. Max {max_file_size_mb}MB."
        if duration_s > max_clip_duration_s:
            return False, f"Clip too long. Max {max_clip_duration_s}s."
        if len(requested_models) > max_models:
            return False, f"Max {max_models} models per run."
        disallowed = [model_name for model_name in requested_models if model_name not in allowed_models]
        if disallowed:
            return False, f"Models not available for public use: {', '.join(disallowed)}"
        return True, None


class JobState(TypedDict):
    status: str
    run_id: Optional[str]
    error: Optional[str]


JobRunner = Callable[[Path, str, Path, list[str], Optional[str], set[str]], None]

jobs: dict[str, JobState] = {}

FAST_TIER_MODELS: list[dict[str, Any]] = [
    {
        "name": "gemini-3-flash",
        "display_name": "Gemini 3 Flash",
        "model_id": "google/gemini-3-flash-preview",
        "provider": "openrouter",
        "supports_images": True,
        "description": "Fast Gemini variant for quick multimodal benchmark runs.",
    },
    {
        "name": "gpt-5.4-mini",
        "display_name": "GPT-5.4 Mini",
        "model_id": "openai/gpt-5.4-mini",
        "provider": "openrouter",
        "supports_images": True,
        "description": "Lower-cost GPT-5.4 option for public interactive demos.",
    },
    {
        "name": "qwen3.5-27b",
        "display_name": "Qwen 3.5 27B",
        "model_id": "qwen/qwen3.5-27b",
        "provider": "openrouter",
        "supports_images": True,
        "description": "Fast Qwen variant that keeps demo costs low while staying multimodal.",
    },
]


def create_app(
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    *,
    job_runner: Optional[JobRunner] = None,
) -> FastAPI:
    """Create the dashboard backend application."""

    app = FastAPI(title="VBench API", version="0.3.0")
    app.state.artifacts_dir = Path(artifacts_dir)
    app.state.job_runner = job_runner or _run_benchmark_job
    app.state.artifacts_dir.mkdir(parents=True, exist_ok=True)

    allowed_origin = os.environ.get("ALLOWED_ORIGIN", "").strip()
    allow_origins = ["http://localhost:3000"]
    if allowed_origin:
        allow_origins.append(allowed_origin)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_origin_regex=r"https://.*\.vercel\.app",
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": "0.3.0",
            "limits": {
                "max_clip_s": int(PUBLIC_LIMITS["max_clip_duration_s"]),
                "max_file_size_mb": int(PUBLIC_LIMITS["max_file_size_mb"]),
                "max_models": int(PUBLIC_LIMITS["max_models"]),
                "allowed_models": list(PUBLIC_LIMITS["allowed_models"]),
            },
        }

    @app.get("/api/models")
    async def models() -> dict[str, list[dict[str, Any]]]:
        return {"models": FAST_TIER_MODELS}

    @app.get("/api/runs")
    async def list_runs() -> list[dict[str, Any]]:
        return _list_runs(Path(app.state.artifacts_dir))

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        storage = Storage(app.state.artifacts_dir)
        try:
            return _build_run_payload(storage, run_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/segments/{segment_id}/media")
    async def get_segment_media(run_id: str, segment_id: str) -> dict[str, Any]:
        storage = Storage(app.state.artifacts_dir)
        try:
            return _build_segment_media_payload(storage, run_id, segment_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/api/benchmark")
    async def start_benchmark(
        background_tasks: BackgroundTasks,
        video: UploadFile = File(...),
        models: Optional[str] = Form(None),
        name: Optional[str] = Form(None),
    ) -> dict[str, Any]:
        requested_models = _parse_requested_models(models)
        if not requested_models:
            requested_models = list(PUBLIC_LIMITS["allowed_models"])

        upload_name = Path(video.filename or "upload.mp4").name
        if Path(upload_name).suffix.lower() not in ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported file type for '{upload_name}'.",
            )

        uploads_root = Path(app.state.artifacts_dir) / UPLOADS_DIR
        uploads_root.mkdir(parents=True, exist_ok=True)
        upload_dir = Path(tempfile.mkdtemp(prefix="public_benchmark_", dir=uploads_root))
        upload_path = upload_dir / upload_name

        try:
            file_size_bytes = await _save_upload_file(
                video,
                upload_path,
                int(PUBLIC_LIMITS["max_file_size_mb"]) * 1024 * 1024,
            )
            video_info = probe_video(upload_path)
        except HTTPException:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise HTTPException(
                status_code=422,
                detail=f"Unable to read uploaded video: {exc}",
            ) from exc

        is_valid, error_message = validate_public_request(
            file_size_bytes=file_size_bytes,
            duration_s=video_info.duration_s,
            requested_models=requested_models,
        )
        if not is_valid:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise HTTPException(status_code=422, detail=error_message or "Upload rejected.")

        job_id = str(uuid.uuid4())
        jobs[job_id] = {"status": "queued", "run_id": None, "error": None}
        background_tasks.add_task(
            app.state.job_runner,
            Path(app.state.artifacts_dir),
            job_id,
            upload_path,
            requested_models,
            _normalize_name(name),
            _existing_run_ids(_runs_dir(Path(app.state.artifacts_dir))),
        )

        return {
            "job_id": job_id,
            "status": "queued",
            "estimated_time_s": ESTIMATED_TIME_S,
        }

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job ID.")
        return job

    return app


async def _save_upload_file(
    upload: UploadFile,
    destination: Path,
    max_bytes: int,
) -> int:
    size = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(
                    status_code=422,
                    detail=f"File too large. Max {int(PUBLIC_LIMITS['max_file_size_mb'])}MB.",
                )
            handle.write(chunk)
    await upload.close()
    return size


def _parse_requested_models(raw_models: Optional[str]) -> list[str]:
    if raw_models is None or not raw_models.strip():
        return []

    try:
        payload = json.loads(raw_models)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="models must be valid JSON.") from exc

    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise HTTPException(status_code=422, detail="models must be a JSON array of strings.")

    cleaned = [item.strip() for item in payload if item.strip()]
    if not cleaned:
        raise HTTPException(status_code=422, detail="At least one model must be selected.")
    return cleaned


def _normalize_name(raw_name: Optional[str]) -> Optional[str]:
    if raw_name is None:
        return None
    cleaned = raw_name.strip()
    return cleaned or None


def _runs_dir(artifacts_dir: Path) -> Path:
    runs_dir = artifacts_dir / DEFAULT_RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def _existing_run_ids(runs_dir: Path) -> set[str]:
    if not runs_dir.exists():
        return set()
    return {entry.name for entry in runs_dir.iterdir() if entry.is_dir()}


def _augment_pythonpath(extra_path: Path, current: Optional[str]) -> str:
    components = [str(extra_path), str(ROOT)]
    if current:
        components.append(current)
    return os.pathsep.join(components)


def _run_benchmark_job(
    artifacts_dir: Path,
    job_id: str,
    upload_path: Path,
    requested_models: list[str],
    name: Optional[str],
    before_runs: set[str],
) -> None:
    jobs[job_id] = {"status": "running", "run_id": None, "error": None}
    run_id: Optional[str] = None

    try:
        command = [
            sys.executable,
            "-m",
            "video_eval_harness.cli",
            "run-benchmark",
            str(upload_path),
            "--config",
            str(ROOT / "configs" / "benchmark_fast.yaml"),
            "--models",
            str(ROOT / "configs" / "models.yaml"),
            "--model-filter",
            ",".join(requested_models),
            "--max-segments",
            str(PUBLIC_LIMITS.get("max_segments", 6)),
            "--public",
            "--artifacts",
            str(artifacts_dir),
        ]
        if name:
            command.extend(["--name", name])

        env = os.environ.copy()
        env["PYTHONPATH"] = _augment_pythonpath(ROOT / "src", env.get("PYTHONPATH"))
        env["VBENCH_ARTIFACTS_DIR"] = str(artifacts_dir)
        env["VBENCH_RUNS_DIR"] = str(_runs_dir(artifacts_dir))

        result = subprocess.run(
            command,
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )

        (upload_path.parent / "benchmark.stdout.log").write_text(
            result.stdout or "",
            encoding="utf-8",
        )
        (upload_path.parent / "benchmark.stderr.log").write_text(
            result.stderr or "",
            encoding="utf-8",
        )

        if result.returncode != 0:
            jobs[job_id] = {
                "status": "failed",
                "run_id": None,
                "error": _command_error(result.stdout, result.stderr),
            }
            return

        run_id = _detect_run_id(result.stdout, before_runs, _runs_dir(artifacts_dir))
        if not run_id:
            jobs[job_id] = {
                "status": "failed",
                "run_id": None,
                "error": "Benchmark completed but the run ID could not be detected.",
            }
            return

        jobs[job_id] = {"status": "complete", "run_id": run_id, "error": None}
    except Exception as exc:  # pragma: no cover - exercised through API behavior
        jobs[job_id] = {"status": "failed", "run_id": run_id, "error": str(exc)}
    finally:
        shutil.rmtree(upload_path.parent, ignore_errors=True)


def _command_error(stdout: str, stderr: str) -> str:
    for content in (stderr, stdout):
        stripped = content.strip()
        if stripped:
            return stripped.splitlines()[-1]
    return "Benchmark process failed."


def _detect_run_id(stdout: str, before_runs: set[str], runs_dir: Path) -> Optional[str]:
    match = re.search(r"Run:\s*(?:\[bold\])?(?:.+?\()?(run_[A-Za-z0-9_\-]+)", stdout)
    if match:
        return match.group(1)

    current_runs = _existing_run_ids(runs_dir)
    new_runs = [run_id for run_id in current_runs if run_id not in before_runs]
    if not new_runs:
        return None

    return max(
        new_runs,
        key=lambda run_id: (runs_dir / run_id).stat().st_mtime,
    )


def _list_runs(artifacts_dir: Path) -> list[dict[str, Any]]:
    storage = Storage(artifacts_dir)
    run_rows: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    runs_dir = _runs_dir(artifacts_dir)

    for run in storage.list_runs():
        if not (runs_dir / run.run_id).exists():
            continue
        seen_run_ids.add(run.run_id)
        run_rows.append(
            {
                "run_id": run.run_id,
                "created_at": run.created_at,
                "models": run.models,
                "prompt_version": run.prompt_version,
                "video_ids": run.video_ids,
            }
        )

    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if not run_dir.is_dir() or run_dir.name in seen_run_ids:
            continue
        fallback = _load_run_meta_from_exports(run_dir)
        if fallback is not None:
            run_rows.append(fallback)

    return sorted(
        run_rows,
        key=lambda row: str(row.get("created_at", "")),
        reverse=True,
    )


def _load_run_meta_from_exports(run_dir: Path) -> Optional[dict[str, Any]]:
    run_id = run_dir.name
    json_path = run_dir / f"{run_id}_results.json"
    if not json_path.exists():
        return None

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    if isinstance(payload, dict):
        results = payload.get("results", [])
    elif isinstance(payload, list):
        results = payload
    else:
        return None

    if not isinstance(results, list):
        return None

    models = sorted(
        {
            str(result.get("model_name"))
            for result in results
            if isinstance(result, dict) and result.get("model_name")
        }
    )
    video_ids = sorted(
        {
            str(result.get("video_id"))
            for result in results
            if isinstance(result, dict) and result.get("video_id")
        }
    )
    timestamps = sorted(
        str(result.get("timestamp"))
        for result in results
        if isinstance(result, dict) and result.get("timestamp")
    )

    created_at = timestamps[0] if timestamps else ""
    return {
        "run_id": run_id,
        "created_at": created_at,
        "models": models,
        "prompt_version": "action_label",
        "video_ids": video_ids,
    }


def _collect_segments(storage: Storage, video_ids: list[str]) -> list[Any]:
    segments: list[Any] = []
    for video_id in video_ids:
        segments.extend(storage.get_segments(video_id))
    return sorted(segments, key=lambda segment: (segment.video_id, segment.segment_index))


def _find_segment(storage: Storage, run_id: str, segment_id: str) -> Any:
    run_config = storage.get_run(run_id)
    if run_config is None:
        raise ValueError(f"No run found for '{run_id}'")

    for segment in _collect_segments(storage, run_config.video_ids):
        if segment.segment_id == segment_id:
            return segment
    raise ValueError(f"Segment '{segment_id}' not found in run '{run_id}'")


def _build_run_payload(storage: Storage, run_id: str) -> dict[str, Any]:
    from video_eval_harness.evaluation.metrics import (
        compute_agreement_matrix,
        compute_model_summary,
    )

    run_config = storage.get_run(run_id)
    if run_config is None:
        raise ValueError(f"No run found for '{run_id}'")

    results = storage.get_run_results(run_id)
    segments = _collect_segments(storage, run_config.video_ids)
    video_lookup = {
        video_id: storage.get_video(video_id)
        for video_id in run_config.video_ids
    }

    models = sorted({result.model_name for result in results}) or sorted(run_config.models)
    summaries = {
        model_name: compute_model_summary(results, model_name).model_dump()
        for model_name in models
    }
    agreement = compute_agreement_matrix(results) if results else {}

    segment_items = []
    for segment in segments:
        frames = storage.get_extracted_frames(segment.segment_id)
        video_meta = video_lookup.get(segment.video_id)
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
                "has_contact_sheet": bool(
                    frames
                    and frames.contact_sheet_path
                    and Path(frames.contact_sheet_path).exists()
                ),
            }
        )

    return {
        "run_id": run_id,
        "config": run_config.model_dump(),
        "models": models,
        "videos": [
            video.model_dump()
            for video in video_lookup.values()
            if video is not None
        ],
        "summaries": summaries,
        "agreement": agreement,
        "segments": segment_items,
        "results": [result.model_dump() for result in results],
    }


def _build_segment_media_payload(
    storage: Storage,
    run_id: str,
    segment_id: str,
) -> dict[str, Any]:
    frames = storage.get_extracted_frames(segment_id)
    segment = _find_segment(storage, run_id, segment_id)

    if frames is None:
        raise ValueError(f"No extracted frames found for '{segment_id}'")

    return {
        "run_id": run_id,
        "segment_id": segment_id,
        "start_time_s": segment.start_time_s,
        "end_time_s": segment.end_time_s,
        "frame_timestamps_s": frames.frame_timestamps_s,
        "contact_sheet_data_url": _file_to_data_url(frames.contact_sheet_path),
        "frames": [
            {
                "timestamp_s": timestamp_s,
                "data_url": _file_to_data_url(frame_path),
            }
            for frame_path, timestamp_s in zip(
                frames.frame_paths,
                frames.frame_timestamps_s,
            )
        ],
    }


def _file_to_data_url(path_str: Optional[str]) -> Optional[str]:
    if not path_str:
        return None

    path = Path(path_str)
    if not path.exists():
        return None

    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


app = create_app()
