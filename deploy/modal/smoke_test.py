"""Smoke test the public Modal benchmark API.

Usage:
  py -3.12 deploy/modal/smoke_test.py ^
    --api-base https://your-modal-app-url/ ^
    --video-url https://example.com/clip.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def request_json(url: str, method: str = "GET", payload: dict | None = None) -> tuple[int, dict]:
    body = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request) as response:
            raw = response.read().decode("utf-8")
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        data = json.loads(raw) if raw else {}
        return exc.code, data


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the public Modal benchmark API.")
    parser.add_argument("--api-base", required=True, help="Public Modal API base URL")
    parser.add_argument("--video-url", required=True, help="Public or pre-signed video URL")
    parser.add_argument("--video-name", default="smoke-test.mp4", help="Display name for the run")
    parser.add_argument("--window-size", type=float, default=10.0, help="Segment window size")
    parser.add_argument("--num-frames", type=int, default=8, help="Frames per segment")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gemini-3.1-pro", "gpt-5.4"],
        help="Models to compare",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0, help="Polling interval")
    parser.add_argument("--max-polls", type=int, default=60, help="Maximum polls before giving up")
    args = parser.parse_args()

    api_base = args.api_base.rstrip("/") + "/"
    submit_url = api_base + "benchmarks"
    payload = {
        "video_url": args.video_url,
        "video_name": args.video_name,
        "models": args.models,
        "segmentation_mode": "fixed_window",
        "window_size": args.window_size,
        "stride": None,
        "num_frames": args.num_frames,
        "prompt_version": "concise",
        "max_concurrency": 2,
    }

    status_code, submit_data = request_json(submit_url, method="POST", payload=payload)
    if status_code >= 400:
        print(f"Submit failed ({status_code}): {submit_data}", file=sys.stderr)
        return 1

    call_id = submit_data.get("call_id")
    if not call_id:
        print(f"Unexpected submit response: {submit_data}", file=sys.stderr)
        return 1

    print(f"Submitted benchmark job: {call_id}")

    poll_url = api_base + f"benchmarks/jobs/{call_id}"
    for attempt in range(1, args.max_polls + 1):
        status_code, poll_data = request_json(poll_url)
        status = poll_data.get("status", "unknown")
        print(f"[{attempt}/{args.max_polls}] status={status}")

        if status == "completed":
            result = poll_data.get("result", {})
            print(f"Run ready: {result.get('run_id')}")
            print(f"Models: {', '.join(result.get('models', []))}")
            print(f"Segments: {len(result.get('segments', []))}")
            print("Summaries:")
            for model_name, summary in result.get("summaries", {}).items():
                parse_rate = summary.get("parse_success_rate")
                latency = summary.get("avg_latency_ms")
                cost = summary.get("total_estimated_cost")
                print(
                    f"  - {model_name}: parse_rate={parse_rate}, "
                    f"avg_latency_ms={latency}, total_cost={cost}"
                )
            return 0

        if status == "failed":
            print(f"Benchmark failed: {poll_data}", file=sys.stderr)
            return 1

        time.sleep(args.poll_seconds)

    print("Timed out waiting for benchmark completion.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
