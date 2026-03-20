FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./

RUN python - <<'PY'
from __future__ import annotations

import subprocess
import sys
import tomllib

with open("pyproject.toml", "rb") as fh:
    project = tomllib.load(fh)["project"]

deps = list(project.get("dependencies", []))
deps.extend(project.get("optional-dependencies", {}).get("ui", []))

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "--prefix=/install", *deps]
)
PY


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . /app

RUN python -m pip install -e ".[ui]" --no-deps

ENTRYPOINT ["vbench"]
