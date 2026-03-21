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


@app.function(
    image=image,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("openrouter-key")],
    timeout=300,
    memory=512,
    cpu=1,
)
@modal.asgi_app()
def api():
    from deploy.api_server import create_app

    return create_app(
        artifacts_dir="/data/artifacts",
        sync_artifacts=volume.commit,
        refresh_artifacts=volume.reload,
    )
