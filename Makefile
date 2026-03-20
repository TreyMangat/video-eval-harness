PYTHON ?= py -3.12
PIP_INSTALL = $(PYTHON) -m pip install -e ".[dev,ui]"
PYTEST = $(PYTHON) -m pytest
RUFF = $(PYTHON) -m ruff

.PHONY: install test lint run-test sweep-dry-run docker-build docker-run

install:
	$(PIP_INSTALL)

test:
	$(PYTEST) -q

lint:
	$(RUFF) check src tests

run-test:
	vbench run-benchmark test_videos/test_25s.mp4 --config configs/benchmark_fast.yaml

sweep-dry-run:
	vbench run-benchmark test_videos/test_25s.mp4 --config configs/benchmark_fast.yaml --sweep --dry-run

docker-build:
	docker build -t vbench .

docker-run:
	docker run --rm -v "$(CURDIR)/configs:/app/configs" -v "$(CURDIR)/artifacts:/app/artifacts" -v "$(CURDIR)/test_videos:/app/test_videos" --env-file .env vbench run-benchmark /app/test_videos/test_25s.mp4 --config /app/configs/benchmark_fast.yaml
