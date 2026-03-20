#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is not set. Put it in .env or export it before running the smoke test." >&2
  exit 1
fi

if [[ ! -f "configs/benchmark_single.yaml" && -f "configs/templates/benchmark_single.yaml" ]]; then
  cp "configs/templates/benchmark_single.yaml" "configs/benchmark_single.yaml"
fi

mkdir -p artifacts test_videos

SOURCE_URL="${TEST_VIDEO_URL:-https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4}"
SOURCE_VIDEO="test_videos/smoke_source.mp4"
SMOKE_VIDEO="test_videos/smoke_30s.mp4"

echo "Downloading Creative Commons smoke-test clip..."
"$PYTHON_BIN" - "$SOURCE_URL" "$SOURCE_VIDEO" <<'PY'
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

url = sys.argv[1]
target = Path(sys.argv[2])
target.parent.mkdir(parents=True, exist_ok=True)

with urllib.request.urlopen(url) as response:
    target.write_bytes(response.read())
PY

echo "Trimming clip to ~30 seconds..."
ffmpeg -y -i "$SOURCE_VIDEO" -t 30 -vf "scale=640:-2" "$SMOKE_VIDEO" >/dev/null 2>&1

echo "Running single-model smoke test..."
vbench run-benchmark "$SMOKE_VIDEO" --config configs/benchmark_single.yaml

echo "Validating SQLite results..."
"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import sqlite3
from pathlib import Path

db_path = Path("artifacts/vbench.db")
if not db_path.exists():
    raise SystemExit("Smoke test failed: artifacts/vbench.db was not created.")

with sqlite3.connect(db_path) as conn:
    result_count = conn.execute("SELECT COUNT(*) FROM label_results").fetchone()[0]

if result_count <= 0:
    raise SystemExit("Smoke test failed: no label_results rows were written.")

print(f"Smoke test passed with {result_count} label_results rows.")
PY
