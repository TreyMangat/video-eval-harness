from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from deploy.api_server import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path, job_runner=lambda *args, **kwargs: None)
    return TestClient(app)


def test_post_benchmark_rejects_files_over_limit(client, monkeypatch) -> None:
    async def oversize_upload(*args, **kwargs):
        raise HTTPException(status_code=422, detail="File too large. Max 500MB.")

    monkeypatch.setattr("deploy.api_server._save_upload_file", oversize_upload)

    response = client.post(
        "/api/benchmark",
        files={"video": ("clip.mp4", b"tiny", "video/mp4")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "File too large. Max 500MB."


def test_post_benchmark_rejects_video_longer_than_600s(client, monkeypatch) -> None:
    monkeypatch.setattr(
        "deploy.api_server.probe_video",
        lambda path: SimpleNamespace(duration_s=601.0),
    )

    response = client.post(
        "/api/benchmark",
        files={"video": ("clip.mp4", b"tiny", "video/mp4")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Clip too long. Max 600s."


def test_get_health_returns_limits_structure(client) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["version"] == "0.3.0"
    assert payload["limits"]["max_clip_s"] == 600
    assert payload["limits"]["max_file_size_mb"] == 500
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
        "stage": "queued",
        "progress": "Upload received. Waiting for a worker...",
    }


def test_get_job_falls_back_to_external_state_loader(tmp_path) -> None:
    app = create_app(
        tmp_path,
        job_runner=lambda *args, **kwargs: None,
        job_state_loader=lambda job_id: {
            "job_id": job_id,
            "status": "queued",
            "run_id": None,
            "error": None,
            "stage": None,
            "progress": None,
        },
    )
    client = TestClient(app)

    response = client.get("/api/jobs/external-job")

    assert response.status_code == 200
    assert response.json() == {
        "job_id": "external-job",
        "status": "queued",
        "run_id": None,
        "error": None,
        "stage": None,
        "progress": None,
    }


def test_get_models_returns_all_public_model_tiers(client) -> None:
    response = client.get("/api/models")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["models"]) == 7
    assert {model["tier"] for model in payload["models"]} == {"fast", "frontier"}
    assert {model["name"] for model in payload["models"]} == {
        "gemini-3-flash",
        "gpt-5.4-mini",
        "qwen3.5-27b",
        "gemini-3.1-pro",
        "gpt-5.4",
        "qwen3.5-vl",
        "llama-4-maverick",
    }


def test_cors_headers_are_present(client) -> None:
    response = client.get(
        "/api/health",
        headers={"Origin": "http://localhost:3000"},
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_get_run_falls_back_to_exported_json_payload(tmp_path) -> None:
    run_id = "run_api_upload_1234"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / f"{run_id}_results.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config": {
                    "created_at": "2026-03-22T10:00:00+00:00",
                    "models": ["gemini-3-flash"],
                    "prompt_version": "action_label_optimized",
                    "segmentation_mode": "fixed_window",
                    "segmentation_config": {"mode": "fixed_window"},
                    "extraction_config": {"num_frames": 8, "method": "uniform"},
                    "video_ids": ["clip001"],
                },
                "models": ["gemini-3-flash"],
                "segments": [
                    {
                        "segment_id": "clip001_seg0000",
                        "video_id": "clip001",
                        "start_time_s": 0.0,
                        "end_time_s": 5.0,
                    }
                ],
                "agreement": {"gemini-3-flash": {"gemini-3-flash": 1.0}},
                "summaries": {
                    "gemini-3-flash": {
                        "model_name": "gemini-3-flash",
                        "parse_success_rate": 1.0,
                        "avg_latency_ms": 1200.0,
                        "total_estimated_cost": 0.01,
                    }
                },
                "results": [
                    {
                        "run_id": run_id,
                        "video_id": "clip001",
                        "segment_id": "clip001_seg0000",
                        "start_time_s": 0.0,
                        "end_time_s": 5.0,
                        "model_name": "gemini-3-flash",
                        "provider": "openrouter",
                        "primary_action": "opening drawer",
                        "secondary_actions": [],
                        "description": "A person opens a drawer.",
                        "objects": ["drawer"],
                        "environment_context": None,
                        "confidence": 0.9,
                        "reasoning_summary_or_notes": None,
                        "uncertainty_flags": [],
                        "parsed_success": True,
                        "parse_error": None,
                        "latency_ms": 1200.0,
                        "estimated_cost": 0.01,
                        "prompt_version": "action_label_optimized",
                        "timestamp": "2026-03-22T10:00:00+00:00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    client = TestClient(create_app(tmp_path, job_runner=lambda *args, **kwargs: None))
    response = client.get(f"/api/runs/{run_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == run_id
    assert payload["config"]["video_ids"] == ["clip001"]
    assert payload["videos"] == [{"video_id": "clip001", "filename": "clip001"}]
    assert payload["segments"][0]["duration_s"] == 5.0


def test_get_segment_media_falls_back_to_exported_artifacts(tmp_path) -> None:
    run_id = "run_api_upload_1234"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    frame_dir = tmp_path / "frames" / "clip001" / "clip001_seg0000"
    frame_dir.mkdir(parents=True)
    frame_path = frame_dir / "frame_0001.jpg"
    frame_path.write_bytes(b"\xff\xd8\xff\xd9")
    (frame_dir / "metadata.json").write_text(
        json.dumps(
            {
                "frame_paths": ["frames/clip001/clip001_seg0000/frame_0001.jpg"],
                "frame_timestamps_s": [1.25],
                "contact_sheet_path": None,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / f"{run_id}_results.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "config": {
                    "created_at": "2026-03-22T10:00:00+00:00",
                    "models": ["gemini-3-flash"],
                    "prompt_version": "action_label_optimized",
                    "segmentation_mode": "fixed_window",
                    "segmentation_config": {"mode": "fixed_window"},
                    "extraction_config": {"num_frames": 8, "method": "uniform"},
                    "video_ids": ["clip001"],
                },
                "models": ["gemini-3-flash"],
                "segments": [
                    {
                        "segment_id": "clip001_seg0000",
                        "video_id": "clip001",
                        "start_time_s": 0.0,
                        "end_time_s": 5.0,
                    }
                ],
                "agreement": {"gemini-3-flash": {"gemini-3-flash": 1.0}},
                "summaries": {
                    "gemini-3-flash": {
                        "model_name": "gemini-3-flash",
                        "parse_success_rate": 1.0,
                    }
                },
                "results": [
                    {
                        "run_id": run_id,
                        "video_id": "clip001",
                        "segment_id": "clip001_seg0000",
                        "start_time_s": 0.0,
                        "end_time_s": 5.0,
                        "model_name": "gemini-3-flash",
                        "provider": "openrouter",
                        "primary_action": "opening drawer",
                        "secondary_actions": [],
                        "description": "A person opens a drawer.",
                        "objects": ["drawer"],
                        "environment_context": None,
                        "confidence": 0.9,
                        "reasoning_summary_or_notes": None,
                        "uncertainty_flags": [],
                        "parsed_success": True,
                        "parse_error": None,
                        "latency_ms": 1200.0,
                        "estimated_cost": 0.01,
                        "prompt_version": "action_label_optimized",
                        "timestamp": "2026-03-22T10:00:00+00:00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    client = TestClient(create_app(tmp_path, job_runner=lambda *args, **kwargs: None))
    response = client.get(f"/api/runs/{run_id}/segments/clip001_seg0000/media")

    assert response.status_code == 200
    payload = response.json()
    assert payload["segment_id"] == "clip001_seg0000"
    assert payload["frame_timestamps_s"] == [1.25]
    assert payload["frames"][0]["data_url"].startswith("data:image/jpeg;base64,")


def test_get_run_falls_back_to_legacy_list_export(tmp_path) -> None:
    run_id = "run_legacy_export_1234"
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / f"{run_id}_results.json").write_text(
        json.dumps(
            [
                {
                    "run_id": run_id,
                    "video_id": "clip001",
                    "segment_id": "clip001_seg0000",
                    "start_time_s": 0.0,
                    "end_time_s": 5.0,
                    "model_name": "gemini-3-flash",
                    "provider": "openrouter",
                    "primary_action": "opening drawer",
                    "secondary_actions": [],
                    "description": "A person opens a drawer.",
                    "objects": ["drawer"],
                    "environment_context": None,
                    "confidence": 0.9,
                    "reasoning_summary_or_notes": None,
                    "uncertainty_flags": [],
                    "parsed_success": True,
                    "parse_error": None,
                    "latency_ms": 1200.0,
                    "estimated_cost": 0.01,
                    "prompt_version": "action_label_optimized",
                    "timestamp": "2026-03-22T10:00:00+00:00",
                }
            ]
        ),
        encoding="utf-8",
    )

    client = TestClient(create_app(tmp_path, job_runner=lambda *args, **kwargs: None))
    response = client.get(f"/api/runs/{run_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == run_id
    assert payload["models"] == ["gemini-3-flash"]
    assert payload["segments"][0]["segment_id"] == "clip001_seg0000"
