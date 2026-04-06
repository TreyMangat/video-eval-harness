"""Core logic for cleaning up junk/debug/test runs from MongoDB."""

from __future__ import annotations

from dataclasses import dataclass, field

from pymongo.database import Database

HIDDEN_RUN_PATTERNS = [
    "debug-fast",
    "debug-test",
    "run-type-accuracy-test",
    "run-type-comparison-test",
    "accuracy-e2e-test",
    "e2e-test",
    "e2e-final-test",
    "e2e-smoke",
    "stage-e2e",
    "cors-test",
    "volume-fix",
    "preview-fix",
    "warmup-test",
    "fix-test",
    "report-accuracy-live-test",
    "node-test",
]


def _normalized_run_name(doc: dict) -> str:
    """Mirror the normalizedRunName logic from deploy/frontend/lib/run-visibility.ts."""
    config = doc.get("config") or {}
    config_display_name = config.get("display_name") if isinstance(config, dict) else None

    parts = [
        doc.get("name"),
        doc.get("display_name"),
        config_display_name,
        doc.get("_id") or doc.get("run_id"),
    ]
    return " ".join(p.strip() for p in parts if isinstance(p, str) and p.strip()).lower()


def _is_junk_run(doc: dict) -> bool:
    normalized = _normalized_run_name(doc)
    return any(pattern in normalized for pattern in HIDDEN_RUN_PATTERNS)


@dataclass
class CleanupResult:
    junk_runs: list[dict] = field(default_factory=list)
    runs_deleted: int = 0
    label_results_deleted: int = 0
    orphans_cleaned: int = 0


def find_junk_runs(db: Database) -> list[dict]:
    """Return all run documents that match junk patterns."""
    all_runs = list(db["runs"].find())
    return [r for r in all_runs if _is_junk_run(r)]


def cleanup_junk_runs(
    db: Database,
    *,
    dry_run: bool = False,
    log_fn: callable = print,
) -> CleanupResult:
    """Delete junk runs and their label_results, plus orphaned label_results.

    Returns a CleanupResult with counts of what was (or would be) deleted.
    """
    result = CleanupResult()
    result.junk_runs = find_junk_runs(db)

    if not result.junk_runs:
        log_fn("No junk runs found.")
        return result

    # Preview
    log_fn(f"\nFound {len(result.junk_runs)} junk run(s):")
    for run in result.junk_runs:
        run_id = run.get("_id") or run.get("run_id", "?")
        display = run.get("display_name") or run.get("name") or run_id
        label_count = db["label_results"].count_documents({"run_id": run_id})
        log_fn(f"  {run_id}  ({display})  — {label_count} label_results")

    if dry_run:
        log_fn("\n[dry-run] No changes made.")
        return result

    # Delete
    junk_ids = [r.get("_id") or r.get("run_id") for r in result.junk_runs]

    label_del = db["label_results"].delete_many({"run_id": {"$in": junk_ids}})
    result.label_results_deleted = label_del.deleted_count

    runs_del = db["runs"].delete_many({"_id": {"$in": junk_ids}})
    result.runs_deleted = runs_del.deleted_count

    # Orphan cleanup: label_results whose run_id doesn't exist in runs
    remaining_run_ids = {
        doc["_id"] for doc in db["runs"].find({}, {"_id": 1})
    }
    if remaining_run_ids:
        orphan_del = db["label_results"].delete_many(
            {"run_id": {"$nin": list(remaining_run_ids)}}
        )
    else:
        orphan_del = db["label_results"].delete_many({})
    result.orphans_cleaned = orphan_del.deleted_count

    log_fn(f"\nDeleted {result.runs_deleted} runs")
    log_fn(f"Deleted {result.label_results_deleted} label_results")
    log_fn(f"Cleaned up {result.orphans_cleaned} orphaned label_results")

    return result
