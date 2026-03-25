"""Modal deployment entrypoint for the public VBench API."""

from __future__ import annotations

from pathlib import Path

import modal

FILE_PATH = Path(__file__).resolve()
ROOT = FILE_PATH.parents[2] if len(FILE_PATH.parents) > 2 else Path("/app")

app = modal.App("vbench")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install_from_pyproject(str(ROOT / "pyproject.toml"))
    .pip_install("fastapi>=0.115", "python-multipart>=0.0.9")
    .env(
        {
            "PYTHONPATH": "/app/src:/app",
            "VBENCH_ARTIFACTS_DIR": "/data/artifacts",
            "VBENCH_RUNS_DIR": "/data/artifacts/runs",
        }
    )
    .add_local_dir(str(ROOT / "src"), remote_path="/app/src")
    .add_local_file(str(ROOT / "deploy" / "__init__.py"), remote_path="/app/deploy/__init__.py")
    .add_local_file(str(ROOT / "deploy" / "api_server.py"), remote_path="/app/deploy/api_server.py")
    .add_local_dir(str(ROOT / "configs"), remote_path="/app/configs")
)

volume = modal.Volume.from_name("vbench-data", create_if_missing=True)
job_state_store = modal.Dict.from_name("vbench-job-states", create_if_missing=True)


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("openrouter-key")],
    timeout=1800,
    memory=1024,
    cpu=1,
)
def run_benchmark_job(
    job_id: str,
    upload_path: str,
    requested_models: list[str],
    name: str | None,
    before_runs: list[str],
    run_type: str,
    ground_truth_path: str | None = None,
    mode: str = "coarse",
) -> None:
    from deploy.api_server import _run_benchmark_job

    def persist_job_state(job_data: dict[str, object]) -> None:
        job_state_store.put(job_id, job_data)

    volume.reload()
    _run_benchmark_job(
        Path("/data/artifacts"),
        job_id,
        Path(upload_path),
        requested_models,
        name,
        set(before_runs),
        run_type,
        Path(ground_truth_path) if ground_truth_path else None,
        mode,
        sync_artifacts=volume.commit,
        persist_job_state=persist_job_state,
    )


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("openrouter-key")],
    timeout=300,
    min_containers=1,
    memory=512,
    cpu=1,
)
@modal.asgi_app()
def api():
    from deploy.api_server import create_app

    def persist_job_state(job_data: dict[str, object]) -> None:
        job_state_store.put(str(job_data["job_id"]), job_data)

    def load_job_state(job_id: str) -> dict[str, object] | None:
        payload = job_state_store.get(job_id, None)
        return payload if isinstance(payload, dict) else None

    def submit_job(
        artifacts_dir: Path,
        job_id: str,
        upload_path: Path,
        requested_models: list[str],
        name: str | None,
        before_runs: set[str],
        run_type: str,
        ground_truth_path: Path | None,
        mode: str,
    ) -> None:
        volume.commit()
        run_benchmark_job.spawn(
            job_id,
            str(upload_path),
            requested_models,
            name,
            sorted(before_runs),
            run_type,
            str(ground_truth_path) if ground_truth_path else None,
            mode,
        )

    return create_app(
        artifacts_dir="/data/artifacts",
        job_submitter=submit_job,
        job_state_loader=load_job_state,
        job_state_saver=persist_job_state,
        sync_artifacts=volume.commit,
        refresh_artifacts=volume.reload,
    )
