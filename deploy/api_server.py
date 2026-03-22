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
DEFAULT_JOBS_DIR = "jobs"
UPLOADS_DIR = "uploads"
PUBLIC_BENCHMARK_CONFIG = ROOT / "configs" / "benchmark_all_models_optimized.yaml"
ALLOWED_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ESTIMATED_TIME_S = 90

try:
    from video_eval_harness.limits import PUBLIC_LIMITS
except Exception:  # pragma: no cover - fallback only used if limits.py is unavailable
    PUBLIC_LIMITS = {
        "max_clip_duration_s": 60,
        "max_file_size_mb": 100,
        "max_segments": 6,
        "max_models": 3,
        "allowed_models": ["gemini-3-flash", "gpt-5.4-mini", "qwen3.5-27b"],
    }


class JobState(TypedDict):
    job_id: str
    status: str
    run_id: Optional[str]
    error: Optional[str]
    stage: Optional[str]
    progress: Optional[str]


ArtifactSync = Callable[[], None]
JobStateLoader = Callable[[str], Optional[JobState]]
JobStateSaver = Callable[[JobState], None]
JobRunner = Callable[
    [
        Path,
        str,
        Path,
        list[str],
        Optional[str],
        set[str],
        Optional[ArtifactSync],
        Optional[JobStateSaver],
    ],
    None,
]
JobSubmitter = Callable[[Path, str, Path, list[str], Optional[str], set[str]], None]

PUBLIC_MODEL_CATALOG: list[dict[str, Any]] = [
    {
        "name": "gemini-3-flash",
        "display_name": "Gemini 3 Flash",
        "model_id": "google/gemini-3-flash-preview",
        "provider": "openrouter",
        "supports_images": True,
        "description": "Fast Gemini variant for quick multimodal benchmark runs.",
        "tier": "fast",
        "estimated_cost_per_segment": 0.01,
    },
    {
        "name": "gpt-5.4-mini",
        "display_name": "GPT-5.4 Mini",
        "model_id": "openai/gpt-5.4-mini",
        "provider": "openrouter",
        "supports_images": True,
        "description": "Lower-cost GPT-5.4 option for public interactive demos.",
        "tier": "fast",
        "estimated_cost_per_segment": 0.01,
    },
    {
        "name": "qwen3.5-27b",
        "display_name": "Qwen 3.5 27B",
        "model_id": "qwen/qwen3.5-27b",
        "provider": "openrouter",
        "supports_images": True,
        "description": "Fast Qwen variant that keeps demo costs low while staying multimodal.",
        "tier": "fast",
        "estimated_cost_per_segment": 0.01,
    },
    {
        "name": "gemini-3.1-pro",
        "display_name": "Gemini 3.1 Pro",
        "model_id": "google/gemini-3.1-pro-preview",
        "provider": "openrouter",
        "supports_images": True,
        "description": "Higher-accuracy Gemini tier for stronger label quality.",
        "tier": "frontier",
        "estimated_cost_per_segment": 0.08,
    },
    {
        "name": "gpt-5.4",
        "display_name": "GPT-5.4",
        "model_id": "openai/gpt-5.4",
        "provider": "openrouter",
        "supports_images": True,
        "description": "Frontier GPT tier with stronger reasoning and structured outputs.",
        "tier": "frontier",
        "estimated_cost_per_segment": 0.08,
    },
    {
        "name": "qwen3.5-vl",
        "display_name": "Qwen 3.5 VL",
        "model_id": "qwen/qwen3.5-397b-a17b",
        "provider": "openrouter",
        "supports_images": True,
        "description": "Frontier multimodal Qwen tier for broader accuracy comparisons.",
        "tier": "frontier",
        "estimated_cost_per_segment": 0.08,
    },
    {
        "name": "llama-4-maverick",
        "display_name": "Llama 4 Maverick",
        "model_id": "meta-llama/llama-4-maverick",
        "provider": "openrouter",
        "supports_images": True,
        "description": "Meta's frontier multimodal model for higher-cost benchmark runs.",
        "tier": "frontier",
        "estimated_cost_per_segment": 0.08,
    },
]

API_PUBLIC_LIMITS = {
    **PUBLIC_LIMITS,
    "max_models": len(PUBLIC_MODEL_CATALOG),
    "allowed_models": [str(model["name"]) for model in PUBLIC_MODEL_CATALOG],
}
DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:3000",
    "https://video-eval-harness-qu4m.vercel.app",
)


def validate_api_public_request(
    file_size_bytes: int,
    duration_s: float,
    requested_models: list[str],
    num_frames: int = 8,
) -> tuple[bool, str | None]:
    """Validate uploads against the public API limits used by the Modal deployment."""

    max_file_size_mb = int(API_PUBLIC_LIMITS["max_file_size_mb"])
    max_clip_duration_s = int(API_PUBLIC_LIMITS["max_clip_duration_s"])
    max_models = int(API_PUBLIC_LIMITS["max_models"])
    max_frames = int(API_PUBLIC_LIMITS.get("max_frames_per_segment", 8))
    allowed_models = {str(model_name) for model_name in API_PUBLIC_LIMITS["allowed_models"]}

    if file_size_bytes > max_file_size_mb * 1024 * 1024:
        return False, f"File too large. Max {max_file_size_mb}MB."
    if duration_s > max_clip_duration_s:
        return False, f"Clip too long. Max {max_clip_duration_s}s."
    if len(requested_models) > max_models:
        return False, f"Max {max_models} models per run."
    disallowed = [model_name for model_name in requested_models if model_name not in allowed_models]
    if disallowed:
        return False, f"Models not available for public use: {', '.join(disallowed)}"
    if num_frames > max_frames:
        return False, f"Max {max_frames} frames per segment."
    return True, None


def _configured_allowed_origins() -> list[str]:
    origins = list(DEFAULT_ALLOWED_ORIGINS)
    for env_name in ("VBENCH_FRONTEND_URL", "ALLOWED_ORIGIN"):
        raw_value = os.environ.get(env_name, "")
        for candidate in raw_value.split(","):
            origin = candidate.strip()
            if origin and origin not in origins:
                origins.append(origin)
    return origins


def create_app(
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    *,
    job_runner: Optional[JobRunner] = None,
    job_submitter: Optional[JobSubmitter] = None,
    job_state_loader: Optional[JobStateLoader] = None,
    job_state_saver: Optional[JobStateSaver] = None,
    sync_artifacts: Optional[ArtifactSync] = None,
    refresh_artifacts: Optional[ArtifactSync] = None,
) -> FastAPI:
    """Create the dashboard backend application."""

    app = FastAPI(title="VBench API", version="0.3.0")
    app.state.artifacts_dir = Path(artifacts_dir)
    app.state.job_runner = job_runner or _run_benchmark_job
    app.state.job_submitter = job_submitter
    app.state.job_state_loader = job_state_loader
    app.state.job_state_saver = job_state_saver
    app.state.sync_artifacts = sync_artifacts or _noop_sync
    app.state.refresh_artifacts = refresh_artifacts or _noop_sync
    app.state.artifacts_dir.mkdir(parents=True, exist_ok=True)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_configured_allowed_origins(),
        allow_origin_regex=r"https://.*\.vercel\.app",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": "0.3.0",
            "limits": {
                "max_clip_s": int(API_PUBLIC_LIMITS["max_clip_duration_s"]),
                "max_file_size_mb": int(API_PUBLIC_LIMITS["max_file_size_mb"]),
                "max_models": int(API_PUBLIC_LIMITS["max_models"]),
                "allowed_models": list(API_PUBLIC_LIMITS["allowed_models"]),
            },
        }

    @app.get("/api/models")
    async def models() -> dict[str, list[dict[str, Any]]]:
        return {"models": PUBLIC_MODEL_CATALOG}

    @app.get("/api/runs")
    async def list_runs() -> list[dict[str, Any]]:
        app.state.refresh_artifacts()
        return _list_runs(Path(app.state.artifacts_dir))

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        app.state.refresh_artifacts()
        storage = Storage(app.state.artifacts_dir)
        try:
            return _build_run_payload(storage, run_id)
        except Exception as exc:
            exported_payload = _load_exported_run_payload(Path(app.state.artifacts_dir), run_id)
            if exported_payload is not None:
                return exported_payload
            if isinstance(exc, ValueError):
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            raise HTTPException(status_code=500, detail="Unable to load run payload.") from exc

    @app.get("/api/runs/{run_id}/segments/{segment_id}/media")
    async def get_segment_media(
        run_id: str,
        segment_id: str,
        variantId: Optional[str] = None,
    ) -> dict[str, Any]:
        app.state.refresh_artifacts()
        storage = Storage(app.state.artifacts_dir)
        try:
            if variantId:
                exported_media = _build_exported_segment_media_payload(
                    Path(app.state.artifacts_dir),
                    run_id,
                    segment_id,
                    variantId,
                )
                if exported_media is not None:
                    return exported_media
            return _build_segment_media_payload(storage, run_id, segment_id)
        except Exception as exc:
            exported_media = _build_exported_segment_media_payload(
                Path(app.state.artifacts_dir),
                run_id,
                segment_id,
                variantId,
            )
            if exported_media is not None:
                return exported_media
            if isinstance(exc, ValueError):
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            raise HTTPException(status_code=500, detail="Unable to load segment media.") from exc

    @app.post("/api/benchmark")
    async def start_benchmark(
        background_tasks: BackgroundTasks,
        video: Optional[UploadFile] = File(None),
        uploaded_file: Optional[UploadFile] = File(None, alias="file"),
        models: Optional[str] = Form(None),
        name: Optional[str] = Form(None),
    ) -> dict[str, Any]:
        upload = video or uploaded_file
        if upload is None:
            raise HTTPException(status_code=422, detail="A video file is required.")

        requested_models = _parse_requested_models(models)
        if not requested_models:
            requested_models = list(API_PUBLIC_LIMITS["allowed_models"])

        upload_name = Path(upload.filename or "upload.mp4").name
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
                upload,
                upload_path,
                int(API_PUBLIC_LIMITS["max_file_size_mb"]) * 1024 * 1024,
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

        is_valid, error_message = validate_api_public_request(
            file_size_bytes=file_size_bytes,
            duration_s=video_info.duration_s,
            requested_models=requested_models,
        )
        if not is_valid:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise HTTPException(status_code=422, detail=error_message or "Upload rejected.")

        job_id = str(uuid.uuid4())
        _save_job(
            Path(app.state.artifacts_dir),
            job_id,
            "queued",
            stage="queued",
            progress="Upload received. Waiting for a worker...",
            sync_artifacts=app.state.sync_artifacts,
            persist_job_state=app.state.job_state_saver,
        )
        try:
            if app.state.job_submitter is not None:
                app.state.job_submitter(
                    Path(app.state.artifacts_dir),
                    job_id,
                    upload_path,
                    requested_models,
                    _normalize_name(name),
                    _existing_run_ids(_runs_dir(Path(app.state.artifacts_dir))),
                )
            else:
                background_tasks.add_task(
                    app.state.job_runner,
                    Path(app.state.artifacts_dir),
                    job_id,
                    upload_path,
                    requested_models,
                    _normalize_name(name),
                    _existing_run_ids(_runs_dir(Path(app.state.artifacts_dir))),
                    app.state.sync_artifacts,
                    app.state.job_state_saver,
                )
        except Exception as exc:
            _save_job(
                Path(app.state.artifacts_dir),
                job_id,
                "failed",
                error=f"Unable to start benchmark job: {exc}",
                stage="failed",
                progress="Unable to start benchmark job.",
                sync_artifacts=app.state.sync_artifacts,
                persist_job_state=app.state.job_state_saver,
            )
            raise HTTPException(status_code=500, detail="Failed to start benchmark job.") from exc

        return {
            "job_id": job_id,
            "status": "queued",
            "estimated_time_s": ESTIMATED_TIME_S,
        }

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        app.state.refresh_artifacts()
        job = _load_job(Path(app.state.artifacts_dir), job_id)
        if job is None and app.state.job_state_loader is not None:
            job = _normalize_loaded_job(app.state.job_state_loader(job_id), job_id)
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
                    detail=f"File too large. Max {int(API_PUBLIC_LIMITS['max_file_size_mb'])}MB.",
                )
            handle.write(chunk)
    await upload.close()
    return size


def _parse_requested_models(raw_models: Optional[str]) -> list[str]:
    if raw_models is None or not raw_models.strip():
        return []

    try:
        payload = json.loads(raw_models)
    except json.JSONDecodeError:
        payload = [item.strip() for item in raw_models.split(",") if item.strip()]

    if isinstance(payload, str):
        payload = [payload]

    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise HTTPException(
            status_code=422,
            detail="models must be a JSON array of strings or a comma-separated string.",
        )

    cleaned = [item.strip() for item in payload if item.strip()]
    if not cleaned:
        raise HTTPException(status_code=422, detail="At least one model must be selected.")
    return cleaned


def _normalize_name(raw_name: Optional[str]) -> Optional[str]:
    if raw_name is None:
        return None
    cleaned = raw_name.strip()
    return cleaned or None


def _noop_sync() -> None:
    return None


def _runs_dir(artifacts_dir: Path) -> Path:
    runs_dir = artifacts_dir / DEFAULT_RUNS_DIR
    runs_dir.mkdir(parents=True, exist_ok=True)
    return runs_dir


def _existing_run_ids(runs_dir: Path) -> set[str]:
    if not runs_dir.exists():
        return set()
    return {entry.name for entry in runs_dir.iterdir() if entry.is_dir()}


def _jobs_dir(artifacts_dir: Path) -> Path:
    jobs_dir = artifacts_dir / DEFAULT_JOBS_DIR
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return jobs_dir


def _save_job(
    artifacts_dir: Path,
    job_id: str,
    status: str,
    *,
    run_id: Optional[str] = None,
    error: Optional[str] = None,
    stage: Optional[str] = None,
    progress: Optional[str] = None,
    sync_artifacts: Optional[ArtifactSync] = None,
    persist_job_state: Optional[JobStateSaver] = None,
) -> JobState:
    job_data: JobState = {
        "job_id": job_id,
        "status": status,
        "run_id": run_id,
        "error": error,
        "stage": stage,
        "progress": progress,
    }
    (_jobs_dir(artifacts_dir) / f"{job_id}.json").write_text(
        json.dumps(job_data),
        encoding="utf-8",
    )
    if persist_job_state is not None:
        persist_job_state(job_data)
    if sync_artifacts is not None:
        sync_artifacts()
    return job_data


def _load_job(artifacts_dir: Path, job_id: str) -> Optional[JobState]:
    path = _jobs_dir(artifacts_dir) / f"{job_id}.json"
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None

    status = payload.get("status")
    if not isinstance(status, str):
        return None

    run_id = payload.get("run_id")
    error = payload.get("error")
    stored_job_id = payload.get("job_id")
    stage = payload.get("stage")
    progress_msg = payload.get("progress")

    return {
        "job_id": stored_job_id if isinstance(stored_job_id, str) else job_id,
        "status": status,
        "run_id": run_id if isinstance(run_id, str) else None,
        "error": error if isinstance(error, str) else None,
        "stage": stage if isinstance(stage, str) else None,
        "progress": progress_msg if isinstance(progress_msg, str) else None,
    }


def _normalize_loaded_job(raw_job: Any, fallback_job_id: str) -> Optional[JobState]:
    if not isinstance(raw_job, dict):
        return None

    status = raw_job.get("status")
    if not isinstance(status, str):
        return None

    run_id = raw_job.get("run_id")
    error = raw_job.get("error")
    stage = raw_job.get("stage")
    progress_msg = raw_job.get("progress")
    stored_job_id = raw_job.get("job_id")

    return {
        "job_id": stored_job_id if isinstance(stored_job_id, str) else fallback_job_id,
        "status": status,
        "run_id": run_id if isinstance(run_id, str) else None,
        "error": error if isinstance(error, str) else None,
        "stage": stage if isinstance(stage, str) else None,
        "progress": progress_msg if isinstance(progress_msg, str) else None,
    }


def make_progress_callback(
    artifacts_dir: Path,
    job_id: str,
    sync_artifacts: Optional[ArtifactSync] = None,
    persist_job_state: Optional[JobStateSaver] = None,
) -> Callable[[str, str], None]:
    """Create a progress callback that writes stage updates to the job file.

    The returned callable has signature ``callback(stage, message)`` matching
    the ``progress_callback`` parameter on :class:`LabelingRunner.run`.
    """
    def callback(stage: str, message: str) -> None:
        _save_job(
            artifacts_dir,
            job_id,
            "running",
            stage=stage,
            progress=message,
            sync_artifacts=sync_artifacts,
            persist_job_state=persist_job_state,
        )
    return callback


def _update_job_stage(
    artifacts_dir: Path,
    job_id: str,
    stage: str,
    progress: str,
    sync_artifacts: Optional[ArtifactSync] = None,
    persist_job_state: Optional[JobStateSaver] = None,
) -> None:
    _save_job(
        artifacts_dir,
        job_id,
        "running",
        stage=stage,
        progress=progress,
        sync_artifacts=sync_artifacts,
        persist_job_state=persist_job_state,
    )


def _parse_progress_stage(line: str, requested_models: list[str]) -> Optional[tuple[str, str]]:
    normalized = line.strip()
    if not normalized:
        return None

    if "Step 1:" in normalized and "Ingest" in normalized:
        return "preparing", "Validating and ingesting the uploaded clip..."
    if "Step 2:" in normalized and "Segment" in normalized:
        return "segmenting", "Splitting the clip into benchmark segments..."
    if "Step 3:" in normalized and ("Extract" in normalized or "Label" in normalized):
        return "extracting", "Extracting frames from each segment..."
    if re.search(r"Variant\s+\d+/\d+", normalized):
        variant_match = re.search(r"Variant\s+(\d+)/(\d+)", normalized)
        if variant_match:
            return (
                "labeling",
                f"Labeling sweep variant {variant_match.group(1)} of {variant_match.group(2)}...",
            )
    if normalized.startswith("Run:"):
        return "labeling", f"Labeling with {len(requested_models)} models..."
    if "Step 4:" in normalized and ("Summary" in normalized or "Evaluate" in normalized):
        return "evaluating", "Computing agreement, accuracy, and summary metrics..."
    if "Exported JSON" in normalized or "Results saved to" in normalized:
        return "finalizing", "Saving benchmark artifacts and preparing the report..."

    return None


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
    sync_artifacts: Optional[ArtifactSync] = None,
    persist_job_state: Optional[JobStateSaver] = None,
) -> None:
    _update_job_stage(
        artifacts_dir,
        job_id,
        "preparing",
        "Preparing benchmark...",
        sync_artifacts=sync_artifacts,
        persist_job_state=persist_job_state,
    )
    run_id: Optional[str] = None

    try:
        _update_job_stage(
            artifacts_dir,
            job_id,
            "labeling",
            f"Running benchmark with {len(requested_models)} models...",
            sync_artifacts=sync_artifacts,
            persist_job_state=persist_job_state,
        )

        command = [
            sys.executable,
            "-u",
            "-m",
            "video_eval_harness.cli",
            "run-benchmark",
            str(upload_path),
            "--config",
            str(PUBLIC_BENCHMARK_CONFIG),
            "--models",
            str(ROOT / "configs" / "models.yaml"),
            "--model-filter",
            ",".join(requested_models),
            "--max-segments",
            str(API_PUBLIC_LIMITS.get("max_segments", 6)),
            "--public",
            "--llm-judge",
            "--artifacts",
            str(artifacts_dir),
        ]
        if name:
            command.extend(["--name", name])

        env = os.environ.copy()
        env["PYTHONPATH"] = _augment_pythonpath(ROOT / "src", env.get("PYTHONPATH"))
        env["VBENCH_ARTIFACTS_DIR"] = str(artifacts_dir)
        env["VBENCH_RUNS_DIR"] = str(_runs_dir(artifacts_dir))

        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        combined_output: list[str] = []
        assert process.stdout is not None
        for line in process.stdout:
            combined_output.append(line)
            stage_update = _parse_progress_stage(line, requested_models)
            if stage_update is not None:
                stage, progress = stage_update
                _update_job_stage(
                    artifacts_dir,
                    job_id,
                    stage,
                    progress,
                    sync_artifacts=sync_artifacts,
                    persist_job_state=persist_job_state,
                )

        return_code = process.wait()
        stdout_text = "".join(combined_output)
        stderr_text = ""

        (upload_path.parent / "benchmark.stdout.log").write_text(
            stdout_text,
            encoding="utf-8",
        )
        (upload_path.parent / "benchmark.stderr.log").write_text(
            stderr_text,
            encoding="utf-8",
        )

        if return_code != 0:
            _save_job(
                artifacts_dir,
                job_id,
                "failed",
                error=_command_error(stdout_text, stderr_text),
                stage="failed",
                progress="Benchmark failed before results were saved.",
                sync_artifacts=sync_artifacts,
                persist_job_state=persist_job_state,
            )
            return

        _update_job_stage(
            artifacts_dir,
            job_id,
            "finalizing",
            "Benchmark complete, saving results...",
            sync_artifacts=sync_artifacts,
            persist_job_state=persist_job_state,
        )

        if sync_artifacts is not None:
            sync_artifacts()

        run_id = _detect_run_id(stdout_text, before_runs, _runs_dir(artifacts_dir))
        if not run_id:
            _save_job(
                artifacts_dir,
                job_id,
                "failed",
                error="Benchmark completed but the run ID could not be detected.",
                stage="failed",
                progress="Benchmark finished, but the run ID could not be detected.",
                sync_artifacts=sync_artifacts,
                persist_job_state=persist_job_state,
            )
            return

        _save_job(
            artifacts_dir,
            job_id,
            "complete",
            run_id=run_id,
            stage="complete",
            progress="Benchmark complete.",
            sync_artifacts=sync_artifacts,
            persist_job_state=persist_job_state,
        )
    except Exception as exc:  # pragma: no cover - exercised through API behavior
        _save_job(
            artifacts_dir,
            job_id,
            "failed",
            run_id=run_id,
            error=str(exc),
            stage="failed",
            progress="Benchmark failed unexpectedly.",
            sync_artifacts=sync_artifacts,
            persist_job_state=persist_job_state,
        )
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


def _run_export_candidates(artifacts_dir: Path, run_id: str) -> list[Path]:
    runs_dir = _runs_dir(artifacts_dir)
    return [
        runs_dir / run_id / f"{run_id}_results.json",
        runs_dir / run_id / "results.json",
        runs_dir / f"{run_id}_results.json",
    ]


def _load_export_json(artifacts_dir: Path, run_id: str) -> Optional[Any]:
    for candidate in _run_export_candidates(artifacts_dir, run_id):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, (dict, list)):
            return payload
    return None


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _coerce_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def _export_video_ids(payload: dict[str, Any]) -> list[str]:
    config = payload.get("config")
    if isinstance(config, dict):
        video_ids = config.get("video_ids")
        if isinstance(video_ids, list):
            cleaned = [str(video_id) for video_id in video_ids if str(video_id).strip()]
            if cleaned:
                return cleaned

    segments = payload.get("segments")
    if isinstance(segments, list):
        video_ids = sorted(
            {
                str(segment.get("video_id"))
                for segment in segments
                if isinstance(segment, dict) and segment.get("video_id")
            }
        )
        if video_ids:
            return video_ids

    results = payload.get("results")
    if isinstance(results, list):
        return sorted(
            {
                str(result.get("video_id"))
                for result in results
                if isinstance(result, dict) and result.get("video_id")
            }
        )

    return []


def _normalize_export_summaries(payload: dict[str, Any], models: list[str]) -> dict[str, dict[str, Any]]:
    raw_summaries = payload.get("summaries")
    summaries: dict[str, dict[str, Any]] = {}
    if isinstance(raw_summaries, dict):
        for model_name, raw_summary in raw_summaries.items():
            if isinstance(raw_summary, dict):
                summaries[str(model_name)] = {
                    **raw_summary,
                    "model_name": str(raw_summary.get("model_name") or model_name),
                }

    accuracy_by_model = payload.get("accuracy_by_model")
    if isinstance(accuracy_by_model, dict):
        for model_name, raw_accuracy in accuracy_by_model.items():
            if not isinstance(raw_accuracy, dict):
                continue
            summary = summaries.setdefault(
                str(model_name),
                {
                    "model_name": str(model_name),
                },
            )
            if isinstance(raw_accuracy.get("accuracy"), (int, float)):
                summary.setdefault("accuracy", raw_accuracy["accuracy"])
            if isinstance(raw_accuracy.get("exact_match_rate"), (int, float)):
                summary["exact_match_rate"] = raw_accuracy["exact_match_rate"]
            if isinstance(raw_accuracy.get("fuzzy_match_rate"), (int, float)):
                summary["fuzzy_match_rate"] = raw_accuracy["fuzzy_match_rate"]

    llm_accuracy = payload.get("llm_accuracy")
    if isinstance(llm_accuracy, dict):
        for model_name, raw_accuracy in llm_accuracy.items():
            if not isinstance(raw_accuracy, dict):
                continue
            summary = summaries.setdefault(
                str(model_name),
                {
                    "model_name": str(model_name),
                },
            )
            if isinstance(raw_accuracy.get("llm_accuracy"), (int, float)):
                summary["llm_accuracy"] = raw_accuracy["llm_accuracy"]

    for model_name in models:
        summaries.setdefault(model_name, {"model_name": model_name})

    return summaries


def _normalize_export_segments(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_segments = payload.get("segments")
    results = payload.get("results")
    if isinstance(raw_segments, list) and raw_segments:
        normalized_segments: list[dict[str, Any]] = []
        for index, raw_segment in enumerate(raw_segments):
            if not isinstance(raw_segment, dict):
                continue
            start_time_s = _coerce_float(raw_segment.get("start_time_s"))
            end_time_s = _coerce_float(raw_segment.get("end_time_s"), start_time_s)
            normalized_segments.append(
                {
                    "segment_id": str(raw_segment.get("segment_id") or f"segment_{index:04d}"),
                    "video_id": str(raw_segment.get("video_id") or ""),
                    "video_filename": raw_segment.get("video_filename"),
                    "segment_index": _coerce_int(raw_segment.get("segment_index"), index),
                    "start_time_s": start_time_s,
                    "end_time_s": end_time_s,
                    "duration_s": _coerce_float(
                        raw_segment.get("duration_s"),
                        max(0.0, end_time_s - start_time_s),
                    ),
                    "segmentation_mode": str(
                        raw_segment.get("segmentation_mode") or "exported_run"
                    ),
                    "frame_count": _coerce_int(raw_segment.get("frame_count")),
                    "frame_timestamps_s": (
                        raw_segment.get("frame_timestamps_s")
                        if isinstance(raw_segment.get("frame_timestamps_s"), list)
                        else []
                    ),
                    "has_contact_sheet": bool(raw_segment.get("has_contact_sheet", False)),
                }
            )
        if normalized_segments:
            return normalized_segments

    if not isinstance(results, list):
        return []

    by_segment: dict[str, dict[str, Any]] = {}
    for raw_result in results:
        if not isinstance(raw_result, dict):
            continue
        segment_id = str(raw_result.get("segment_id") or "").strip()
        if not segment_id:
            continue
        existing = by_segment.get(segment_id)
        if existing is None:
            start_time_s = _coerce_float(raw_result.get("start_time_s"))
            end_time_s = _coerce_float(raw_result.get("end_time_s"), start_time_s)
            by_segment[segment_id] = {
                "segment_id": segment_id,
                "video_id": str(raw_result.get("video_id") or ""),
                "video_filename": str(raw_result.get("video_id") or "") or None,
                "segment_index": len(by_segment),
                "start_time_s": start_time_s,
                "end_time_s": end_time_s,
                "duration_s": max(0.0, end_time_s - start_time_s),
                "segmentation_mode": "exported_run",
                "frame_count": _coerce_int(raw_result.get("num_frames_used")),
                "frame_timestamps_s": [],
                "has_contact_sheet": False,
            }
        else:
            existing["frame_count"] = max(
                _coerce_int(existing.get("frame_count", 0)),
                _coerce_int(raw_result.get("num_frames_used")),
            )

    return sorted(
        by_segment.values(),
        key=lambda segment: (
            str(segment.get("video_id", "")),
            float(segment.get("start_time_s", 0.0)),
            str(segment.get("segment_id", "")),
        ),
    )


def _normalize_export_payload(payload: dict[str, Any], run_id: str) -> dict[str, Any]:
    config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    models = [
        str(model_name)
        for model_name in (
            payload.get("models")
            if isinstance(payload.get("models"), list)
            else config.get("models", [])
        )
        if str(model_name).strip()
    ]
    if not models:
        results = payload.get("results")
        if isinstance(results, list):
            models = sorted(
                {
                    str(result.get("model_name"))
                    for result in results
                    if isinstance(result, dict) and result.get("model_name")
                }
            )

    video_ids = _export_video_ids(payload)
    segments = _normalize_export_segments(payload)
    created_at = str(config.get("created_at") or payload.get("created_at") or "")
    normalized_config = {
        "models": models,
        "prompt_version": str(config.get("prompt_version") or "unknown"),
        "segmentation_mode": str(config.get("segmentation_mode") or "unknown"),
        "segmentation_config": (
            config.get("segmentation_config")
            if isinstance(config.get("segmentation_config"), dict)
            else {}
        ),
        "extraction_config": (
            config.get("extraction_config")
            if isinstance(config.get("extraction_config"), dict)
            else {}
        ),
        "model_configs": (
            config.get("model_configs")
            if isinstance(config.get("model_configs"), dict)
            else {}
        ),
        "video_ids": video_ids,
        "created_at": created_at,
    }

    raw_videos = payload.get("videos")
    if isinstance(raw_videos, list):
        videos = [video for video in raw_videos if isinstance(video, dict)]
    else:
        videos = [{"video_id": video_id, "filename": video_id} for video_id in video_ids]

    normalized_payload = {
        **payload,
        "run_id": str(payload.get("run_id") or run_id),
        "config": normalized_config,
        "models": models,
        "videos": videos,
        "summaries": _normalize_export_summaries(payload, models),
        "agreement": payload.get("agreement") if isinstance(payload.get("agreement"), dict) else {},
        "segments": segments,
        "results": payload.get("results") if isinstance(payload.get("results"), list) else [],
    }

    if isinstance(payload.get("sweep"), dict):
        normalized_payload["sweep"] = payload.get("sweep")
    elif isinstance(payload.get("sweep_summary"), dict):
        normalized_payload["sweep"] = payload.get("sweep_summary")

    return normalized_payload


def _load_exported_run_payload(artifacts_dir: Path, run_id: str) -> Optional[dict[str, Any]]:
    payload = _load_export_json(artifacts_dir, run_id)
    if payload is None:
        return None
    if isinstance(payload, list):
        payload = {"run_id": run_id, "results": payload}
    if not isinstance(payload, dict):
        return None
    try:
        return _normalize_export_payload(payload, run_id)
    except Exception:
        results = payload.get("results")
        if isinstance(results, list):
            try:
                return _normalize_export_payload({"run_id": run_id, "results": results}, run_id)
            except Exception:
                return None
        return None


def _find_export_segment(payload: dict[str, Any], segment_id: str) -> Optional[dict[str, Any]]:
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return None
    for segment in segments:
        if isinstance(segment, dict) and str(segment.get("segment_id")) == segment_id:
            return segment
    return None


def _resolve_export_frame_path(raw_path: str, artifacts_dir: Path) -> Path:
    normalized = raw_path.replace("\\", os.sep).replace("/", os.sep)
    path = Path(normalized)
    if path.is_absolute():
        return path
    if normalized.startswith(f"artifacts{os.sep}"):
        return artifacts_dir.parent / normalized
    if normalized.startswith(f"frames{os.sep}"):
        return artifacts_dir / normalized
    return (artifacts_dir / normalized).resolve()


def _find_export_segment_manifest(
    artifacts_dir: Path,
    video_id: str,
    segment_id: str,
    variant_id: Optional[str] = None,
) -> tuple[Optional[Path], Optional[str]]:
    frames_root = artifacts_dir / "frames" / video_id
    if not frames_root.exists():
        return None, None

    if variant_id and variant_id != "default":
        explicit_variant_manifest = frames_root / variant_id / segment_id / "metadata.json"
        if explicit_variant_manifest.exists():
            return explicit_variant_manifest, variant_id

    legacy_manifest = frames_root / segment_id / "metadata.json"
    if legacy_manifest.exists():
        return legacy_manifest, None

    for child in frames_root.iterdir():
        if not child.is_dir():
            continue
        nested_manifest = child / segment_id / "metadata.json"
        if nested_manifest.exists():
            return nested_manifest, child.name

    return None, None


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

    # Read LLM judge data from the exported JSON (computed during benchmark run)
    llm_agreement, llm_accuracy, judge_stats, accuracy_by_model = _load_judge_data_from_export(
        storage, run_id
    )

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
        "llm_agreement": llm_agreement,
        "llm_accuracy": llm_accuracy,
        "judge_stats": judge_stats,
        "accuracy_by_model": accuracy_by_model,
        "segments": segment_items,
        "results": [result.model_dump() for result in results],
    }


def _load_judge_data_from_export(
    storage: Storage, run_id: str,
) -> tuple[Optional[dict], Optional[dict], Optional[dict], Optional[dict]]:
    """Load LLM judge and accuracy data from the exported JSON results file.

    Returns (llm_agreement, llm_accuracy, judge_stats, accuracy_by_model),
    all None if the file doesn't exist or lacks judge data.
    """
    run_dir = storage.run_dir(run_id)
    json_path = Path(run_dir) / f"{run_id}_results.json"
    if not json_path.exists():
        return None, None, None, None

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, None, None, None

    if not isinstance(payload, dict):
        return None, None, None, None

    return (
        payload.get("llm_agreement"),
        payload.get("llm_accuracy"),
        payload.get("judge_stats"),
        payload.get("accuracy_by_model"),
    )


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


def _build_exported_segment_media_payload(
    artifacts_dir: Path,
    run_id: str,
    segment_id: str,
    variant_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    payload = _load_exported_run_payload(artifacts_dir, run_id)
    if payload is None:
        return None

    segment = _find_export_segment(payload, segment_id)
    if segment is None:
        return None

    manifest_path, resolved_variant_id = _find_export_segment_manifest(
        artifacts_dir,
        str(segment.get("video_id") or ""),
        segment_id,
        variant_id,
    )

    if manifest_path is None:
        return {
            "run_id": run_id,
            "segment_id": segment_id,
            "start_time_s": _coerce_float(segment.get("start_time_s")),
            "end_time_s": _coerce_float(segment.get("end_time_s")),
            "frame_timestamps_s": [],
            "contact_sheet_data_url": None,
            "frames": [],
            "variant_id": variant_id or None,
            "variant_label": variant_id or None,
        }

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, dict):
        return None

    frame_paths = manifest_payload.get("frame_paths")
    frame_timestamps_s = manifest_payload.get("frame_timestamps_s")
    resolved_timestamps = frame_timestamps_s if isinstance(frame_timestamps_s, list) else []
    frames: list[dict[str, Any]] = []
    if isinstance(frame_paths, list):
        for index, frame_path in enumerate(frame_paths):
            if not isinstance(frame_path, str):
                continue
            frames.append(
                {
                    "timestamp_s": (
                        _coerce_float(resolved_timestamps[index])
                        if index < len(resolved_timestamps)
                        else _coerce_float(segment.get("start_time_s"))
                    ),
                    "data_url": _file_to_data_url(
                        str(_resolve_export_frame_path(frame_path, artifacts_dir))
                    ),
                }
            )

    variant_label: Optional[str] = None
    sweep_payload = payload.get("sweep")
    if isinstance(sweep_payload, dict):
        variant_map = sweep_payload.get("variant_id_by_label")
        if isinstance(variant_map, dict) and resolved_variant_id:
            for label, mapped_variant_id in variant_map.items():
                if mapped_variant_id == resolved_variant_id:
                    variant_label = str(label)
                    break

    contact_sheet_path = manifest_payload.get("contact_sheet_path")
    return {
        "run_id": run_id,
        "segment_id": segment_id,
        "start_time_s": _coerce_float(segment.get("start_time_s")),
        "end_time_s": _coerce_float(segment.get("end_time_s")),
        "frame_timestamps_s": resolved_timestamps,
        "contact_sheet_data_url": (
            _file_to_data_url(str(_resolve_export_frame_path(contact_sheet_path, artifacts_dir)))
            if isinstance(contact_sheet_path, str) and contact_sheet_path
            else None
        ),
        "frames": frames,
        "variant_id": resolved_variant_id,
        "variant_label": variant_label or resolved_variant_id,
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
