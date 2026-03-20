.PHONY: install test test-all lint run-test sweep-dry-run sweep-fast docker-build viewer docker-run

install:
	pip install -e ".[dev]"

test:
	python -m pytest -q -m "not integration"

test-all:
	python -m pytest -q

lint:
	python -m ruff check src/ tests/

run-test:
	vbench run-benchmark test_videos/test_25s.mp4 --config configs/benchmark_fast.yaml

sweep-dry-run:
	vbench run-benchmark test_videos/test_25s.mp4 --sweep --dry-run

sweep-fast:
	vbench sweep test_videos/test_25s.mp4 --config configs/benchmark_fast.yaml

docker-build:
	docker build -t vbench .

viewer:
	streamlit run src/video_eval_harness/viewer.py

docker-run:
	docker run --rm -v "$(CURDIR)/configs:/app/configs" -v "$(CURDIR)/artifacts:/app/artifacts" -v "$(CURDIR)/test_videos:/app/test_videos" --env-file .env vbench run-benchmark /app/test_videos/test_25s.mp4 --config /app/configs/benchmark_fast.yaml
