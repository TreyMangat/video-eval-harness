#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

echo "Installing development dependencies..."
"$PYTHON_BIN" -m pip install -e ".[dev]"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "FFmpeg is not available on PATH. Install FFmpeg before continuing." >&2
  exit 1
fi

mkdir -p artifacts configs configs/templates test_videos

for config_name in benchmark.yaml benchmark_single.yaml models.yaml prompts.yaml; do
  template_path="configs/templates/${config_name}"
  target_path="configs/${config_name}"
  if [[ -f "$template_path" && ! -f "$target_path" ]]; then
    cp "$template_path" "$target_path"
    echo "Created ${target_path} from template."
  fi
done

echo "Validating CLI install..."
vbench version

echo "Local development environment is ready."
