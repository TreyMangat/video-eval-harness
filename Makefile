PYTHON ?= py -3.12
PIP = $(PYTHON) -m pip
PYTEST = $(PYTHON) -m pytest
RUFF = $(PYTHON) -m ruff
STREAMLIT = $(PYTHON) -m streamlit
VBENCH = $(PYTHON) -m video_eval_harness.cli
NPM ?= npm

.PHONY: install test test-all lint run-test sweep-dry-run sweep-fast sweep-frontier docker-build viewer docker-run dashboard dashboard-build report compare push status api

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

sweep-frontier:
	$(VBENCH) sweep test_videos/test_25s.mp4 --config configs/benchmark.yaml

docker-build:
	docker build -t vbench .

viewer:
	$(STREAMLIT) run src/video_eval_harness/viewer.py --server.headless true --browser.gatherUsageStats false

dashboard:
	$(NPM) --prefix deploy/frontend run dev

dashboard-build:
	$(NPM) --prefix deploy/frontend run build

api:
	$(PYTHON) -m uvicorn deploy.api_server:app --reload --port 8000

report:
	$(VBENCH) export $(RUN) --format json
	@echo Results at artifacts/runs/$(RUN)/

# Usage: make compare RUN_A=run_abc RUN_B=run_def
compare:
	$(VBENCH) compare $(RUN_A) $(RUN_B)

push:
	git push origin main --tags

status:
	@echo Branch:
	@git branch --show-current
	@echo Unpushed commits:
	@$(PYTHON) -c "import subprocess; result = subprocess.run(['git', 'rev-list', '--count', '@{u}..HEAD'], capture_output=True, text=True); print(result.stdout.strip() or '0')"
	@echo Test count:
	@$(PYTHON) -c "import subprocess, sys; result = subprocess.run([sys.executable, '-m', 'pytest', '--collect-only', '-q'], capture_output=True, text=True); lines = [line for line in (result.stdout + result.stderr).splitlines() if '::' in line]; print(len(lines))"
	@echo Ruff status:
	@$(PYTHON) -c "import subprocess, sys; result = subprocess.run([sys.executable, '-m', 'ruff', 'check', 'src/', 'tests/'], capture_output=True, text=True); print('clean' if result.returncode == 0 else 'issues found'); sys.exit(result.returncode)"

docker-run:
	docker run --rm -v "$(CURDIR)/configs:/app/configs" -v "$(CURDIR)/artifacts:/app/artifacts" -v "$(CURDIR)/test_videos:/app/test_videos" --env-file .env vbench run-benchmark /app/test_videos/test_25s.mp4 --config /app/configs/benchmark_fast.yaml
