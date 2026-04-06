"""One-time migration: SQLite + exported JSON -> MongoDB.

Usage:
    py -3.12 scripts/migrate_to_mongodb.py --artifacts ./artifacts --mongodb-uri "mongodb+srv://..."

    # Or with exported JSON files (e.g. from data/ directory):
    py -3.12 scripts/migrate_to_mongodb.py --json-dir ./data --mongodb-uri "mongodb+srv://..."

    # Both:
    py -3.12 scripts/migrate_to_mongodb.py --artifacts ./artifacts --json-dir ./data --mongodb-uri "mongodb+srv://..."
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from pymongo import MongoClient
from pymongo.errors import BulkWriteError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate VBench data from SQLite/JSON to MongoDB"
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="Path to artifacts directory containing vbench.db",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=None,
        help="Path to directory containing exported *_results.json files",
    )
    parser.add_argument(
        "--mongodb-uri",
        required=True,
        help="MongoDB connection string",
    )
    parser.add_argument(
        "--database",
        default="vbench",
        help="MongoDB database name (default: vbench)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be migrated without writing to MongoDB",
    )
    return parser.parse_args()


def _json_or_default(value: str | None, default: list | dict):
    """Parse a JSON string, returning default if None or empty."""
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def migrate_sqlite(db_path: Path, db, dry_run: bool) -> dict[str, int]:
    """Migrate all tables from SQLite to MongoDB collections."""
    counts: dict[str, int] = {}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # --- Videos ---
    rows = conn.execute("SELECT * FROM videos").fetchall()
    video_docs = []
    for r in rows:
        doc = dict(r)
        doc["_id"] = doc.pop("video_id")
        video_docs.append(doc)
    counts["videos"] = len(video_docs)
    if not dry_run and video_docs:
        _bulk_insert(db["videos"], video_docs, "videos")

    # --- Segments ---
    rows = conn.execute("SELECT * FROM segments").fetchall()
    segment_docs = []
    for r in rows:
        doc = dict(r)
        doc["_id"] = doc.pop("segment_id")
        doc["segmentation_config"] = _json_or_default(
            doc.get("segmentation_config"), {}
        )
        segment_docs.append(doc)
    counts["segments"] = len(segment_docs)
    if not dry_run and segment_docs:
        _bulk_insert(db["segments"], segment_docs, "segments")

    # --- Extracted Frames ---
    rows = conn.execute("SELECT * FROM extracted_frames").fetchall()
    frame_docs = []
    for r in rows:
        doc = dict(r)
        doc["_id"] = doc.pop("segment_id")
        doc["frame_paths"] = _json_or_default(doc.get("frame_paths"), [])
        doc["frame_timestamps_s"] = _json_or_default(
            doc.get("frame_timestamps_s"), []
        )
        frame_docs.append(doc)
    counts["extracted_frames"] = len(frame_docs)
    if not dry_run and frame_docs:
        _bulk_insert(db["extracted_frames"], frame_docs, "extracted_frames")

    # --- Label Results ---
    rows = conn.execute("SELECT * FROM label_results").fetchall()
    label_docs = []
    for r in rows:
        doc = dict(r)
        doc.pop("id", None)
        doc["secondary_actions"] = _json_or_default(
            doc.get("secondary_actions"), []
        )
        doc["objects"] = _json_or_default(doc.get("objects"), [])
        doc["uncertainty_flags"] = _json_or_default(
            doc.get("uncertainty_flags"), []
        )
        doc["parsed_success"] = bool(doc.get("parsed_success"))
        label_docs.append(doc)
    counts["label_results"] = len(label_docs)
    if not dry_run and label_docs:
        _bulk_insert(db["label_results"], label_docs, "label_results")

    # --- Runs ---
    rows = conn.execute("SELECT * FROM runs").fetchall()
    run_docs = []
    for r in rows:
        doc = dict(r)
        doc["_id"] = doc.pop("run_id")
        config_json = doc.pop("config_json", None)
        if config_json:
            try:
                config_data = json.loads(config_json)
                doc.update(config_data)
            except (json.JSONDecodeError, TypeError):
                pass
        run_docs.append(doc)
    counts["runs"] = len(run_docs)
    if not dry_run and run_docs:
        _bulk_insert(db["runs"], run_docs, "runs")

    conn.close()
    return counts


def migrate_json_files(json_dir: Path, db, dry_run: bool) -> dict[str, int]:
    """Migrate exported JSON result files to MongoDB."""
    counts: dict[str, int] = {"runs": 0, "label_results": 0}

    json_files = sorted(json_dir.glob("*_results.json"))
    if not json_files:
        print(f"  No *_results.json files found in {json_dir}")
        return counts

    for json_path in json_files:
        print(f"  Processing {json_path.name}...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        run_id = data.get("run_id", "")
        if not run_id:
            print(f"    Skipping {json_path.name}: no run_id found")
            continue

        # Upsert the run — preserve all top-level metadata for the frontend
        config = data.get("config", {})
        run_doc = {
            "_id": run_id,
            "created_at": config.get("created_at", data.get("created_at", "")),
            "models": data.get("models", []),
            "display_name": data.get("display_name"),
            "config": config,
            "summaries": data.get("summaries"),
            "agreement": data.get("agreement"),
            "llm_agreement": data.get("llm_agreement"),
            "llm_accuracy": data.get("llm_accuracy"),
            "judge_stats": data.get("judge_stats"),
            "sweep_summary": data.get("sweep_summary"),
            "cost_estimate_usd": data.get("cost_estimate_usd"),
            "accuracy_by_model": data.get("accuracy_by_model"),
        }
        counts["runs"] += 1
        if not dry_run:
            db["runs"].update_one(
                {"_id": run_id}, {"$set": run_doc}, upsert=True
            )

        # Migrate segments (metadata only, no model results nested here)
        segments = data.get("segments", [])
        for seg in segments:
            seg_id = seg.get("segment_id", "")
            if seg_id and not dry_run:
                seg_doc = dict(seg)
                seg_doc["_id"] = seg_doc.pop("segment_id")
                db["segments"].update_one(
                    {"_id": seg_doc["_id"]}, {"$set": seg_doc}, upsert=True
                )

        # Extract label results from top-level "results" array
        results = data.get("results", [])
        label_docs = []
        for r in results:
            doc = {
                "run_id": r.get("run_id", run_id),
                "video_id": r.get("video_id", ""),
                "segment_id": r.get("segment_id", ""),
                "start_time_s": r.get("start_time_s", 0.0),
                "end_time_s": r.get("end_time_s", 0.0),
                "model_name": r.get("model_name", ""),
                "provider": r.get("provider", ""),
                "primary_action": r.get("primary_action"),
                "secondary_actions": r.get("secondary_actions", []),
                "description": r.get("description"),
                "objects": r.get("objects", []),
                "environment_context": r.get("environment_context"),
                "confidence": r.get("confidence"),
                "reasoning_summary_or_notes": r.get("reasoning_summary_or_notes"),
                "uncertainty_flags": r.get("uncertainty_flags", []),
                "extraction_variant_id": r.get("extraction_variant_id", ""),
                "extraction_label": r.get("extraction_label", ""),
                "num_frames_used": r.get("num_frames_used", 0),
                "sampling_method_used": r.get("sampling_method_used", ""),
                "sweep_id": r.get("sweep_id", ""),
                "input_mode": r.get("input_mode"),
                "raw_response_text": r.get("raw_response_text"),
                "parsed_success": bool(r.get("parsed_success")),
                "parse_error": r.get("parse_error"),
                "latency_ms": r.get("latency_ms"),
                "estimated_cost": r.get("estimated_cost"),
                "prompt_version": r.get("prompt_version"),
                "timestamp": r.get("timestamp", ""),
            }
            label_docs.append(doc)

        counts["label_results"] += len(label_docs)
        if not dry_run and label_docs:
            for doc in label_docs:
                compound_filter = {
                    "run_id": doc["run_id"],
                    "segment_id": doc["segment_id"],
                    "model_name": doc["model_name"],
                    "extraction_variant_id": doc.get("extraction_variant_id", ""),
                }
                db["label_results"].update_one(
                    compound_filter, {"$set": doc}, upsert=True
                )

    return counts


def _bulk_insert(collection, docs: list[dict], name: str) -> None:
    """Insert documents, skipping duplicates."""
    try:
        collection.insert_many(docs, ordered=False)
    except BulkWriteError as e:
        n_errors = len(e.details.get("writeErrors", []))
        n_inserted = e.details.get("nInserted", 0)
        print(
            f"    {name}: inserted {n_inserted}, "
            f"skipped {n_errors} duplicates"
        )


def verify(db, source_counts: dict[str, int]) -> bool:
    """Compare MongoDB collection counts against source counts."""
    all_ok = True
    print("\nVerification:")
    for collection_name, source_count in source_counts.items():
        mongo_count = db[collection_name].count_documents({})
        status = "OK" if mongo_count >= source_count else "MISMATCH"
        if status == "MISMATCH":
            all_ok = False
        print(
            f"  {collection_name}: source={source_count}, "
            f"mongo={mongo_count} [{status}]"
        )
    return all_ok


def main() -> None:
    args = parse_args()

    if not args.artifacts and not args.json_dir:
        print("Error: provide at least one of --artifacts or --json-dir")
        sys.exit(1)

    if args.dry_run:
        print("=== DRY RUN (no writes to MongoDB) ===\n")

    client = MongoClient(args.mongodb_uri)
    db = client[args.database]

    total_counts: dict[str, int] = {}

    if args.artifacts:
        db_path = args.artifacts / "vbench.db"
        if not db_path.exists():
            print(f"SQLite database not found at {db_path}")
        else:
            print(f"Migrating from SQLite: {db_path}")
            counts = migrate_sqlite(db_path, db, args.dry_run)
            for k, v in counts.items():
                total_counts[k] = total_counts.get(k, 0) + v
            print(f"  SQLite counts: {counts}")

    if args.json_dir:
        if not args.json_dir.is_dir():
            print(f"JSON directory not found: {args.json_dir}")
        else:
            print(f"\nMigrating from JSON files: {args.json_dir}")
            counts = migrate_json_files(args.json_dir, db, args.dry_run)
            for k, v in counts.items():
                total_counts[k] = total_counts.get(k, 0) + v
            print(f"  JSON counts: {counts}")

    if not args.dry_run:
        verify(db, total_counts)
    else:
        print(f"\nDry-run totals: {total_counts}")

    client.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
