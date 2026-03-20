# Public Deployment Guide

This repo now supports the cheapest practical public path:

1. Vercel hosts the dashboard.
2. The dashboard submits a benchmark job to Modal using a video URL.
3. Modal downloads the clip, segments it, extracts frames, calls the selected models, and stores the run in a persistent volume.
4. The dashboard polls until the run is complete and then opens the results automatically.

## Architecture

- `deploy/frontend/`: Next.js app for Vercel
- `deploy/modal/app.py`: Modal FastAPI backend
- `deploy/modal/smoke_test.py`: CLI smoke test for the public API

## Cheapest ingestion strategy

Use URL-based ingestion first.

- Best option: paste a public `.mp4` URL into the dashboard.
- Also works: a pre-signed S3, R2, GCS, or similar object URL.
- Avoid building browser uploads first unless you need them. URL submission is much cheaper and simpler because Vercel stays stateless.

## Local visual test

1. Backend:

```bash
py -3.12 -m pip install -e ".[deploy]"
modal serve deploy/modal/app.py
```

2. Frontend:

```bash
cd deploy/frontend
npm install
copy .env.example .env.local
```

Set `MODAL_API_BASE_URL` in `deploy/frontend/.env.local` to the local Modal URL from `modal serve`.

3. Start the frontend:

```bash
npm run dev
```

4. Open `http://localhost:3000`, switch to `Live Backend`, paste a clip URL, and start a benchmark.

## Production deploy

### 1. Configure Modal

Install deps:

```bash
py -3.12 -m pip install -e ".[deploy]"
```

Create a secret called `vbench-api-keys` containing:

- `OPENROUTER_API_KEY`
- `OPENAI_API_KEY` if needed later
- `GOOGLE_API_KEY` if needed later

Deploy:

```bash
modal deploy deploy/modal/app.py
```

Save the public Modal API URL.

### 2. Smoke test Modal before Vercel

Run:

```bash
py -3.12 deploy/modal/smoke_test.py ^
  --api-base https://your-modal-app-url/ ^
  --video-url https://your-public-video-url.mp4
```

You should see:

- a `call_id`
- polling updates
- a final `run_id`
- model summaries

### 3. Deploy Vercel

From `deploy/frontend`:

```bash
npm install
npm run build
```

Set:

```bash
MODAL_API_BASE_URL=https://your-modal-app-url/
```

Then deploy the frontend to Vercel.

## What the public flow does with a clip

1. The dashboard sends `video_url`, segmentation settings, prompt version, and selected models to `POST /benchmarks`.
2. Modal downloads the clip to temporary storage.
3. The backend probes the video with FFmpeg.
4. The backend segments the video using:
   - `fixed_window`: evenly sized windows
   - `scene_heuristic`: histogram-based scene boundary detection with fallback to fixed windows
5. Each segment is sampled into still frames.
6. The backend calls each model with the same frames.
7. Results, summaries, and extracted media previews are stored in the Modal volume.
8. The frontend polls `GET /benchmarks/jobs/{call_id}` until the run is complete.

## Build.ai clips

For Build.ai or other dataset clips, the easiest public path is still:

1. materialize or export the clip somewhere you can address by URL
2. paste that URL into the dashboard

If you later want true browser uploads, add Vercel Blob or another object store and feed the resulting URL into the same benchmark form.
