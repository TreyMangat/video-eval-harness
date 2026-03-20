PYTHON ?= py -3.12
PIP = $(PYTHON) -m pip
PYTEST = $(PYTHON) -m pytest
RUFF = $(PYTHON) -m ruff
STREAMLIT = $(PYTHON) -m streamlit
VBENCH = $(PYTHON) -m video_eval_harness.cli
NPM ?= npm

.PHONY: install test test-all lint run-test sweep-dry-run sweep-fast docker-build viewer docker-run dashboard dashboard-build

install:
	$(PIP) install -e ".[dev]"

test:
	$(PYTEST) -q -m "not integration"

test-all:
	$(PYTEST) -q

lint:
	$(RUFF) check src/ tests/

run-test:
	$(VBENCH) run-benchmark test_videos/test_25s.mp4 --config configs/benchmark_fast.yaml

sweep-dry-run:
	$(VBENCH) run-benchmark test_videos/test_25s.mp4 --sweep --dry-run

sweep-fast:
	$(VBENCH) sweep test_videos/test_25s.mp4 --config configs/benchmark_fast.yaml

docker-build:
	docker build -t vbench .

viewer:
	$(STREAMLIT) run src/video_eval_harness/viewer.py --server.headless true --browser.gatherUsageStats false

dashboard:
	$(NPM) --prefix deploy/frontend run dev

dashboard-build:
	$(NPM) --prefix deploy/frontend run build

docker-run:
	docker run --rm -v "$(CURDIR)/configs:/app/configs" -v "$(CURDIR)/artifacts:/app/artifacts" -v "$(CURDIR)/test_videos:/app/test_videos" --env-file .env vbench run-benchmark /app/test_videos/test_25s.mp4 --config /app/configs/benchmark_fast.yaml
