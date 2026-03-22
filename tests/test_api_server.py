from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from deploy.api_server import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path, job_runner=lambda *args, **kwargs: None)
    return TestClient(app)


def test_post_benchmark_rejects_files_over_100mb(client, monkeypatch) -> None:
    async def oversize_upload(*args, **kwargs):
        raise HTTPException(status_code=422, detail="File too large. Max 100MB.")

    monkeypatch.setattr("deploy.api_server._save_upload_file", oversize_upload)

    response = client.post(
        "/api/benchmark",
        files={"video": ("clip.mp4", b"tiny", "video/mp4")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "File too large. Max 100MB."


def test_post_benchmark_rejects_video_longer_than_60s(client, monkeypatch) -> None:
    monkeypatch.setattr(
        "deploy.api_server.probe_video",
        lambda path: SimpleNamespace(duration_s=61.0),
    )

    response = client.post(
        "/api/benchmark",
        files={"video": ("clip.mp4", b"tiny", "video/mp4")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Clip too long. Max 60s."


def test_get_health_returns_limits_structure(client) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"] == "0.3.0"
    assert payload["limits"]["max_clip_s"] == 60
    assert payload["limits"]["max_file_size_mb"] == 100
    assert payload["limits"]["max_models"] == 7
    assert payload["limits"]["allowed_models"] == [
        "gemini-3-flash",
        "gpt-5.4-mini",
        "qwen3.5-27b",
        "gemini-3.1-pro",
        "gpt-5.4",
        "qwen3.5-vl",
        "llama-4-maverick",
    ]


def test_get_unknown_job_returns_404(client) -> None:
    response = client.get("/api/jobs/missing-job")

    assert response.status_code == 404
    assert response.json()["detail"] == "Unknown job ID."


def test_post_benchmark_persists_job_state(client, monkeypatch) -> None:
    monkeypatch.setattr(
        "deploy.api_server.probe_video",
        lambda path: SimpleNamespace(duration_s=5.0),
    )

    response = client.post(
        "/api/benchmark",
        data={"models": '["gemini-3-flash"]'},
        files={"video": ("clip.mp4", b"tiny", "video/mp4")},
    )

    assert response.status_code == 200
    job_id = response.json()["job_id"]

    job_response = client.get(f"/api/jobs/{job_id}")
    assert job_response.status_code == 200
    assert job_response.json() == {
        "job_id": job_id,
        "status": "queued",
        "run_id": None,
        "error": None,
    }


def test_cors_headers_are_present(client) -> None:
    response = client.get(
        "/api/health",
        headers={"Origin": "http://localhost:3000"},
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
