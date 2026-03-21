## Deploying VBench

### Prerequisites
- Python 3.12
- GitHub repo pushed
- Modal account (modal.com) with $30 free credits
- Vercel account (vercel.com) — free tier
- OpenRouter API key

### Step 1: Deploy Modal Backend
```powershell
py -3.12 -m pip install modal
py -3.12 -m modal setup
py -3.12 -m modal secret create openrouter-key OPENROUTER_API_KEY=sk-or-XXXX
py -3.12 -m modal deploy deploy/modal/app.py
```

Copy the URL printed (`https://YOUR_USER--vbench-api.modal.run`)

### Step 2: Deploy Vercel Frontend
1. Push to GitHub
2. `vercel.com` → New Project → Import repo
3. Set root directory: `deploy/frontend`
4. Add env var: `NEXT_PUBLIC_API_URL = (Modal URL from step 1)`
5. Deploy

### Step 3: Verify
- Visit the Vercel URL
- Upload a short clip
- Wait ~90 seconds
- View results

### Cost
- Each visitor benchmark: ~$0.04-0.08 (OpenRouter API)
- Modal compute: ~$0.003 per run
- Monthly at light demo usage: < $1

### Static Mode (No Backend)
To deploy as viewer only (no uploads):
1. Export runs locally: `vbench export <run_id> --format json --output data/`
2. Commit `data/` to the repo
3. Deploy to Vercel WITHOUT setting `NEXT_PUBLIC_API_URL`
4. Dashboard loads from committed JSON files
