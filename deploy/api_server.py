"""Lightweight API server for the VBench dashboard.

Run with: uvicorn deploy.api_server:app --reload --port 8000
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_RUNS_DIR = ROOT / "artifacts" / "runs"
UPLOADS_DIR = ROOT / "artifacts" / "uploads"
ALLOWED_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
TIER_CONFIGS = {
    "fast": ROOT / "configs" / "benchmark_fast.yaml",
    "frontier": ROOT / "configs" / "benchmark.yaml",
}

app = FastAPI(title="VBench API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@dataclass
class BenchmarkJob:
    job_id: str
    process: subprocess.Popen
    tmp_dir: Path
    stdout_path: Path
    stderr_path: Path
    before_runs: set[str] = field(default_factory=set)
    name: Optional[str] = None


JOBS: dict[str, BenchmarkJob] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _detect_run_id(job: BenchmarkJob) -> Optional[str]:
    """Parse stdout for the run_id, or detect new run dirs."""
    output = _read_text(job.stdout_path)
    match = re.search(r"Run(?:\s+ID)?:\s*(?:\[cyan\])?\s*(\S+?)(?:\[/cyan\])?\s*$", output, re.MULTILINE)
    if match:
        rid = match.group(1).strip()
        if rid.startswith("run_"):
            return rid

    if not ARTIFACT_RUNS_DIR.exists():
        return None
    current_runs = {p.name for p in ARTIFACT_RUNS_DIR.iterdir() if p.is_dir()}
    new_runs = sorted(current_runs - job.before_runs)
    return new_runs[-1] if new_runs else None


def _tail_error(job: BenchmarkJob) -> str:
    for path in [job.stderr_path, job.stdout_path]:
        text = _read_text(path).strip()
        if text:
            return text.splitlines()[-1]
    return "Benchmark process failed."


def _load_run_meta(run_dir: Path) -> Optional[dict]:
    """Load metadata from a run's exported JSON."""
    run_id = run_dir.name
    json_path = run_dir / f"{run_id}_results.json"
    if not json_path.exists():
        return None

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))

        # Handle both envelope format and legacy flat-list format
        if isinstance(data, dict):
            display_name = data.get("display_name")
            results = data.get("results", [])
        elif isinstance(data, list):
            display_name = None
            results = data
        else:
            return None

        models = sorted({r.get("model_name", "") for r in results if r.get("model_name")})
        video_ids = sorted({r.get("video_id", "") for r in results if r.get("video_id")})
        dates = [r.get("timestamp", "") for r in results if r.get("timestamp")]
        date = min(dates)[:16].replace("T", " ") if dates else ""

        return {
            "run_id": run_id,
            "display_name": display_name,
            "date": date,
            "models": models,
            "video_count": len(video_ids),
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "0.3.0"}


@app.post("/api/benchmark")
async def start_benchmark(
    files: list[UploadFile] = File(...),
    tier: str = Form("fast"),
    name: str = Form(""),
):
    """Upload videos and start a benchmark sweep."""
    if tier not in TIER_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Tier must be one of: {', '.join(TIER_CONFIGS)}")
    if not files:
        raise HTTPException(status_code=400, detail="At least one video file is required.")

    # Save uploads
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(prefix="vbench_", dir=UPLOADS_DIR))

    for upload in files:
        filename = Path(upload.filename or "upload.mp4").name
        if Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")
        destination = tmp_dir / filename
        with destination.open("wb") as f:
            shutil.copyfileobj(upload.file, f)

    # Build command
    cmd = [
        "py", "-3.12", "-m", "video_eval_harness.cli",
        "sweep", str(tmp_dir),
        "--config", str(TIER_CONFIGS[tier]),
        "--frames", "4,8",
        "--methods", "uniform",
        "--max-segments", "20",
    ]
    if name:
        cmd.extend(["--name", name])

    # Snapshot existing runs for new-run detection
    before_runs = set()
    if ARTIFACT_RUNS_DIR.exists():
        before_runs = {p.name for p in ARTIFACT_RUNS_DIR.iterdir() if p.is_dir()}

    stdout_path = tmp_dir / "benchmark.stdout.log"
    stderr_path = tmp_dir / "benchmark.stderr.log"

    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=stdout_path.open("w", encoding="utf-8"),
        stderr=stderr_path.open("w", encoding="utf-8"),
        text=True,
    )

    job_id = str(uuid.uuid4())
    JOBS[job_id] = BenchmarkJob(
        job_id=job_id,
        process=process,
        tmp_dir=tmp_dir,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        before_runs=before_runs,
        name=name or None,
    )

    return {"run_id": "pending", "job_id": job_id, "status": "running"}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Check benchmark job status."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job ID.")

    return_code = job.process.poll()

    if return_code is None:
        return {"job_id": job_id, "status": "running", "run_id": None, "error": None}

    if return_code != 0:
        return {
            "job_id": job_id,
            "status": "failed",
            "run_id": None,
            "error": _tail_error(job),
        }

    run_id = _detect_run_id(job)
    return {
        "job_id": job_id,
        "status": "complete",
        "run_id": run_id,
        "error": None,
    }


@app.get("/api/runs")
async def list_runs():
    """List all completed runs with metadata."""
    if not ARTIFACT_RUNS_DIR.exists():
        return []

    runs = []
    for run_dir in sorted(ARTIFACT_RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        meta = _load_run_meta(run_dir)
        if meta:
            runs.append(meta)
    return runs


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    """Get full results for a specific run."""
    run_dir = ARTIFACT_RUNS_DIR / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    result: dict = {"run_id": run_id}

    # Results JSON
    results_path = run_dir / f"{run_id}_results.json"
    if results_path.exists():
        try:
            result["results"] = json.loads(results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result["results"] = None

    # Sweep summary JSON
    sweep_path = run_dir / f"{run_id}_sweep_summary.json"
    if sweep_path.exists():
        try:
            result["sweep_summary"] = json.loads(sweep_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            result["sweep_summary"] = None

    if "results" not in result and "sweep_summary" not in result:
        raise HTTPException(status_code=404, detail=f"No exported data for run: {run_id}")

    return result
