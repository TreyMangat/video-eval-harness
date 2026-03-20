# Frontend Quick Start

## Demo mode only

```bash
cd deploy/frontend
npm install
npm run dev
```

Open `http://localhost:3000` and keep the dashboard in `Demo Data` mode.

## Live backend mode

1. Copy the env file:

```bash
copy .env.example .env.local
```

2. Set:

```bash
MODAL_API_BASE_URL=https://your-modal-app-url/
```

3. Start the app:

```bash
npm run dev
```

4. Switch to `Live Backend`, paste a public or pre-signed video URL, choose models, and launch the benchmark.

The dashboard will poll the Modal backend and open the new run automatically when it finishes.
