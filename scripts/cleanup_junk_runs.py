"""Standalone script to delete junk/debug/test runs from MongoDB.

Usage:
    python scripts/cleanup_junk_runs.py --dry-run
    python scripts/cleanup_junk_runs.py --mongodb-uri "mongodb+srv://..."
    python scripts/cleanup_junk_runs.py --force
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from pymongo import MongoClient

# Allow running from repo root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from video_eval_harness.cleanup import cleanup_junk_runs, find_junk_runs


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Delete junk runs from MongoDB")
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--mongodb-uri", default=None, help="MongoDB URI (default: MONGODB_URI env)")
    args = parser.parse_args()

    uri = args.mongodb_uri or os.environ.get("MONGODB_URI", "")
    if not uri:
        print("Error: MONGODB_URI not set and --mongodb-uri not provided.")
        sys.exit(1)

    client: MongoClient = MongoClient(uri)
    db = client["vbench"]

    junk = find_junk_runs(db)
    if not junk and not args.dry_run:
        print("No junk runs found. Nothing to do.")
        client.close()
        return

    # In dry-run mode, just run cleanup_junk_runs which will print and exit
    if args.dry_run:
        cleanup_junk_runs(db, dry_run=True)
        client.close()
        return

    # Preview
    print(f"\nFound {len(junk)} junk run(s) to delete:")
    for run in junk:
        run_id = run.get("_id") or run.get("run_id", "?")
        display = run.get("display_name") or run.get("name") or run_id
        label_count = db["label_results"].count_documents({"run_id": run_id})
        print(f"  {run_id}  ({display})  — {label_count} label_results")

    if not args.force:
        answer = input("\nProceed with deletion? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            client.close()
            return

    result = cleanup_junk_runs(db, dry_run=False)

    print("\nFinal summary:")
    print(f"  Runs deleted:          {result.runs_deleted}")
    print(f"  Label results deleted: {result.label_results_deleted}")
    print(f"  Orphans cleaned:       {result.orphans_cleaned}")

    client.close()


if __name__ == "__main__":
    main()
