# Deployment

VBench can be run three ways: local development, a single Docker CLI container, or the full Docker Compose stack with the Streamlit viewer.

## Prerequisites

- Python 3.10 or newer. This repo is tested with Python 3.12.
- FFmpeg installed and available on `PATH` for local runs.
- Docker Desktop running for any Docker or Compose workflow.
- An `OPENROUTER_API_KEY` for model calls through OpenRouter.

## Environment Variables

- `OPENROUTER_API_KEY`: required for `run-benchmark` and `sweep`.
- `VBENCH_ARTIFACTS_DIR`: optional override for the artifacts root when running the React dashboard locally.
- `VBENCH_RUNS_DIR`: optional override pointing directly at `artifacts/runs` for the React dashboard.

## Local Development

```bash
make install
make run-test          # smoke test with fast models
make sweep-fast        # full sweep
make viewer            # Streamlit on :8501
make dashboard         # React dashboard
```

What these do:

- `make install` installs the editable package plus development dependencies with `py -3.12`.
- `make run-test` runs the fast benchmark config against `test_videos/test_25s.mp4`.
- `make sweep-fast` runs the fast sweep config.
- `make viewer` starts the Streamlit viewer on `http://localhost:8501`.
- `make dashboard` starts the React dashboard on `http://localhost:3000`.

## Docker (Single Run)

```bash
docker build -t vbench .
docker run -v ./configs:/app/configs -v ./artifacts:/app/artifacts -v ./test_videos:/app/test_videos -e OPENROUTER_API_KEY vbench run-benchmark /app/test_videos/test_25s.mp4 --config /app/configs/benchmark_fast.yaml
```

Volume mounts:

- `./configs:/app/configs`: makes local benchmark and model configs available inside the container.
- `./artifacts:/app/artifacts`: persists run outputs, exports, and extracted frames.
- `./test_videos:/app/test_videos`: provides local input videos to the CLI.

## Docker Compose (Full Stack)

```bash
docker compose -f deploy/docker-compose.prod.yml up
# CLI: docker compose exec vbench-cli vbench sweep /app/data/video.mp4
# Viewer: http://localhost:8501
```

What Compose gives you:

- `vbench-cli`: a CLI container with the project installed.
- `vbench-viewer`: the Streamlit viewer bound to `localhost:8501`.

Mounted data:

- `../configs:/app/configs`: shared benchmark configuration.
- `../artifacts:/app/artifacts`: shared run exports and viewer input data.
- `../data:/app/data`: optional video input directory for Compose CLI usage.

## Dashboard Data Loading

The React dashboard reads run exports from `artifacts/runs`.

- By default it auto-detects `artifacts/runs` from the repo layout.
- Set `VBENCH_RUNS_DIR` if you want to point it at a different runs directory.
- Set `VBENCH_ARTIFACTS_DIR` if you want to point it at a different artifacts root instead.

The dashboard prefers:

- `<run_id>_results.json` for raw run data
- `<run_id>_sweep_summary.json` for precomputed sweep metrics

If a run only has CSV exports, the dashboard falls back to `<run_id>_results.csv`.

## Troubleshooting

- Python 3.9 on `python`:
  This machine often resolves bare `python` to 3.9. Use `make` targets or `py -3.12 ...`. The Makefile now defaults to `py -3.12`, and `scripts/dev-setup.sh` exits early if Python is older than 3.10.
- FFmpeg not found:
  Install FFmpeg locally and confirm `ffmpeg -version` works in your shell before running local extraction or benchmark commands.
- Docker Desktop not running:
  Start Docker Desktop before `docker build` or `docker compose ... up`. If Compose hangs or cannot connect to the daemon, this is the first thing to check.
- Viewer shows no runs:
  Confirm `artifacts/runs` exists and contains exported run directories with `*_results.json`, `*_results.csv`, or both.
- Dashboard loads but API routes are empty:
  Confirm `artifacts/runs` exists, or set `VBENCH_RUNS_DIR` explicitly before starting `make dashboard`.
- Port already in use:
  Streamlit defaults to `8501` and the React dashboard defaults to `3000`. Stop the existing process or choose a different port before retrying.
