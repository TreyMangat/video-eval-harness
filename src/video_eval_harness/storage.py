"""Local storage layer: SQLite for metadata, filesystem for artifacts."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path
from typing import Optional

from .config import get_artifacts_dir, get_settings
from .log import get_logger
from .schemas import (
    ExtractedFrames,
    RunConfig,
    Segment,
    SegmentLabelResult,
    VideoMetadata,
)

logger = get_logger(__name__)

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    duration_s REAL,
    width INTEGER,
    height INTEGER,
    fps REAL,
    codec TEXT,
    file_size_bytes INTEGER,
    ingested_at TEXT
);

CREATE TABLE IF NOT EXISTS segments (
    segment_id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    segment_index INTEGER NOT NULL,
    start_time_s REAL NOT NULL,
    end_time_s REAL NOT NULL,
    duration_s REAL NOT NULL,
    segmentation_mode TEXT,
    segmentation_config TEXT,
    FOREIGN KEY (video_id) REFERENCES videos(video_id)
);

CREATE TABLE IF NOT EXISTS extracted_frames (
    segment_id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    frame_paths TEXT,
    frame_timestamps_s TEXT,
    contact_sheet_path TEXT,
    num_frames INTEGER,
    FOREIGN KEY (segment_id) REFERENCES segments(segment_id)
);

CREATE TABLE IF NOT EXISTS label_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    video_id TEXT NOT NULL,
    segment_id TEXT NOT NULL,
    start_time_s REAL,
    end_time_s REAL,
    model_name TEXT NOT NULL,
    provider TEXT,
    primary_action TEXT,
    secondary_actions TEXT,
    description TEXT,
    objects TEXT,
    environment_context TEXT,
    confidence REAL,
    reasoning_summary_or_notes TEXT,
    uncertainty_flags TEXT,
    extraction_variant_id TEXT DEFAULT '',
    extraction_label TEXT DEFAULT '',
    num_frames_used INTEGER DEFAULT 0,
    sampling_method_used TEXT DEFAULT '',
    sweep_id TEXT DEFAULT '',
    raw_response_text TEXT,
    parsed_success INTEGER,
    parse_error TEXT,
    latency_ms REAL,
    estimated_cost REAL,
    prompt_version TEXT,
    timestamp TEXT,
    UNIQUE(run_id, segment_id, model_name, extraction_variant_id)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT,
    config_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_segments_video ON segments(video_id);
CREATE INDEX IF NOT EXISTS idx_labels_run ON label_results(run_id);
CREATE INDEX IF NOT EXISTS idx_labels_model ON label_results(model_name);
CREATE INDEX IF NOT EXISTS idx_labels_segment ON label_results(segment_id);
"""


class Storage:
    """Manages SQLite database and artifact directories."""

    def __init__(self, artifacts_dir: Optional[str | Path] = None):
        if artifacts_dir is None:
            artifacts_dir = get_artifacts_dir()
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.artifacts_dir / "vbench.db"
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(DB_SCHEMA)
            self._migrate_label_results_table(conn)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _migrate_label_results_table(self, conn: sqlite3.Connection) -> None:
        required_columns = {
            "extraction_variant_id": "TEXT DEFAULT ''",
            "extraction_label": "TEXT DEFAULT ''",
            "num_frames_used": "INTEGER DEFAULT 0",
            "sampling_method_used": "TEXT DEFAULT ''",
            "sweep_id": "TEXT DEFAULT ''",
        }

        existing_columns = self._table_columns(conn, "label_results")
        for column_name, column_sql in required_columns.items():
            if column_name not in existing_columns:
                conn.execute(
                    f"ALTER TABLE label_results ADD COLUMN {column_name} {column_sql}"
                )

        if self._label_results_unique_constraint_needs_rebuild(conn):
            self._rebuild_label_results_table(conn)

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_labels_sweep_model_variant "
            "ON label_results(sweep_id, model_name, extraction_variant_id)"
        )

    def _table_columns(self, conn: sqlite3.Connection, table_name: str) -> set[str]:
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {row["name"] for row in rows}

    def _label_results_unique_constraint_needs_rebuild(
        self, conn: sqlite3.Connection
    ) -> bool:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'label_results'"
        ).fetchone()
        if row is None or not row["sql"]:
            return False

        normalized_sql = " ".join(str(row["sql"]).lower().split())
        desired = "unique(run_id, segment_id, model_name, extraction_variant_id)"
        return desired not in normalized_sql

    def _rebuild_label_results_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE label_results_v2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                video_id TEXT NOT NULL,
                segment_id TEXT NOT NULL,
                start_time_s REAL,
                end_time_s REAL,
                model_name TEXT NOT NULL,
                provider TEXT,
                primary_action TEXT,
                secondary_actions TEXT,
                description TEXT,
                objects TEXT,
                environment_context TEXT,
                confidence REAL,
                reasoning_summary_or_notes TEXT,
                uncertainty_flags TEXT,
                extraction_variant_id TEXT DEFAULT '',
                extraction_label TEXT DEFAULT '',
                num_frames_used INTEGER DEFAULT 0,
                sampling_method_used TEXT DEFAULT '',
                sweep_id TEXT DEFAULT '',
                raw_response_text TEXT,
                parsed_success INTEGER,
                parse_error TEXT,
                latency_ms REAL,
                estimated_cost REAL,
                prompt_version TEXT,
                timestamp TEXT,
                UNIQUE(run_id, segment_id, model_name, extraction_variant_id)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO label_results_v2 (
                id,
                run_id,
                video_id,
                segment_id,
                start_time_s,
                end_time_s,
                model_name,
                provider,
                primary_action,
                secondary_actions,
                description,
                objects,
                environment_context,
                confidence,
                reasoning_summary_or_notes,
                uncertainty_flags,
                extraction_variant_id,
                extraction_label,
                num_frames_used,
                sampling_method_used,
                sweep_id,
                raw_response_text,
                parsed_success,
                parse_error,
                latency_ms,
                estimated_cost,
                prompt_version,
                timestamp
            )
            SELECT
                id,
                run_id,
                video_id,
                segment_id,
                start_time_s,
                end_time_s,
                model_name,
                provider,
                primary_action,
                secondary_actions,
                description,
                objects,
                environment_context,
                confidence,
                reasoning_summary_or_notes,
                uncertainty_flags,
                COALESCE(extraction_variant_id, ''),
                COALESCE(extraction_label, ''),
                COALESCE(num_frames_used, 0),
                COALESCE(sampling_method_used, ''),
                COALESCE(sweep_id, ''),
                raw_response_text,
                parsed_success,
                parse_error,
                latency_ms,
                estimated_cost,
                prompt_version,
                timestamp
            FROM label_results
            """
        )
        conn.execute("DROP TABLE label_results")
        conn.execute("ALTER TABLE label_results_v2 RENAME TO label_results")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_labels_run ON label_results(run_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_labels_model ON label_results(model_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_labels_segment ON label_results(segment_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_labels_sweep_model_variant "
            "ON label_results(sweep_id, model_name, extraction_variant_id)"
        )

    # --- Videos ---

    def save_video(self, video: VideoMetadata) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO videos
                (video_id, source_path, filename, duration_s, width, height, fps, codec, file_size_bytes, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (video.video_id, video.source_path, video.filename, video.duration_s,
                 video.width, video.height, video.fps, video.codec, video.file_size_bytes, video.ingested_at),
            )

    def get_video(self, video_id: str) -> Optional[VideoMetadata]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()
            if row is None:
                return None
            return VideoMetadata(**dict(row))

    def list_videos(self) -> list[VideoMetadata]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM videos ORDER BY ingested_at DESC").fetchall()
            return [VideoMetadata(**dict(r)) for r in rows]

    # --- Segments ---

    def save_segments(self, segments: list[Segment]) -> None:
        with self._conn() as conn:
            for seg in segments:
                conn.execute(
                    """INSERT OR REPLACE INTO segments
                    (segment_id, video_id, segment_index, start_time_s, end_time_s, duration_s,
                     segmentation_mode, segmentation_config)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (seg.segment_id, seg.video_id, seg.segment_index, seg.start_time_s,
                     seg.end_time_s, seg.duration_s, seg.segmentation_mode.value,
                     json.dumps(seg.segmentation_config)),
                )

    def get_segments(self, video_id: str) -> list[Segment]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM segments WHERE video_id = ? ORDER BY segment_index", (video_id,)
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["segmentation_config"] = json.loads(d["segmentation_config"] or "{}")
                results.append(Segment(**d))
            return results

    # --- Extracted Frames ---

    def save_extracted_frames(self, frames: ExtractedFrames) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO extracted_frames
                (segment_id, video_id, frame_paths, frame_timestamps_s, contact_sheet_path, num_frames)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (frames.segment_id, frames.video_id,
                 json.dumps(frames.frame_paths), json.dumps(frames.frame_timestamps_s),
                 frames.contact_sheet_path, frames.num_frames),
            )

    def get_extracted_frames(self, segment_id: str) -> Optional[ExtractedFrames]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM extracted_frames WHERE segment_id = ?", (segment_id,)
            ).fetchone()
            if row is None:
                return None
            d = dict(row)
            d["frame_paths"] = json.loads(d["frame_paths"] or "[]")
            d["frame_timestamps_s"] = json.loads(d["frame_timestamps_s"] or "[]")
            return ExtractedFrames(**d)

    # --- Label Results ---

    def save_label_result(self, result: SegmentLabelResult) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO label_results
                (run_id, video_id, segment_id, start_time_s, end_time_s, model_name, provider,
                 primary_action, secondary_actions, description, objects, environment_context,
                 confidence, reasoning_summary_or_notes, uncertainty_flags, extraction_variant_id,
                 extraction_label, num_frames_used, sampling_method_used, sweep_id,
                 raw_response_text, parsed_success, parse_error, latency_ms, estimated_cost,
                 prompt_version, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (result.run_id, result.video_id, result.segment_id, result.start_time_s,
                 result.end_time_s, result.model_name, result.provider,
                 result.primary_action, json.dumps(result.secondary_actions),
                 result.description, json.dumps(result.objects), result.environment_context,
                 result.confidence, result.reasoning_summary_or_notes,
                 json.dumps(result.uncertainty_flags), result.extraction_variant_id,
                 result.extraction_label, result.num_frames_used,
                 result.sampling_method_used, result.sweep_id, result.raw_response_text,
                 int(result.parsed_success), result.parse_error, result.latency_ms,
                 result.estimated_cost, result.prompt_version, result.timestamp),
            )

    def get_run_results(self, run_id: str) -> list[SegmentLabelResult]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM label_results WHERE run_id = ? "
                "ORDER BY segment_id, model_name, extraction_variant_id",
                (run_id,),
            ).fetchall()
            return [self._row_to_label_result(r) for r in rows]

    def get_segment_results(self, segment_id: str, run_id: Optional[str] = None) -> list[SegmentLabelResult]:
        with self._conn() as conn:
            if run_id:
                rows = conn.execute(
                    "SELECT * FROM label_results WHERE segment_id = ? AND run_id = ? "
                    "ORDER BY model_name, extraction_variant_id",
                    (segment_id, run_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM label_results WHERE segment_id = ? "
                    "ORDER BY model_name, extraction_variant_id",
                    (segment_id,),
                ).fetchall()
            return [self._row_to_label_result(r) for r in rows]

    def get_results_by_sweep(self, sweep_id: str) -> list[SegmentLabelResult]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM label_results WHERE sweep_id = ? "
                "ORDER BY model_name, extraction_variant_id, segment_id",
                (sweep_id,),
            ).fetchall()
            return [self._row_to_label_result(r) for r in rows]

    def has_result(self, run_id: str, segment_id: str, model_name: str, variant_id: str = "") -> bool:
        """Check if a result already exists (for resume support).

        When ``variant_id`` is provided (sweep runs), it's included in the
        lookup so different extraction variants aren't treated as duplicates.
        Requires the ``extraction_variant_id`` column to exist in the
        ``label_results`` table (added by the sweep schema migration).
        Falls back to the 3-column check if the column doesn't exist yet.
        """
        with self._conn() as conn:
            if variant_id:
                try:
                    row = conn.execute(
                        "SELECT 1 FROM label_results "
                        "WHERE run_id = ? AND segment_id = ? AND model_name = ? AND extraction_variant_id = ?",
                        (run_id, segment_id, model_name, variant_id),
                    ).fetchone()
                    return row is not None
                except sqlite3.OperationalError:
                    # Column doesn't exist yet — fall through to basic check.
                    # This means sweep resume won't deduplicate across variants
                    # until the schema migration adds the column.
                    pass
            row = conn.execute(
                "SELECT 1 FROM label_results WHERE run_id = ? AND segment_id = ? AND model_name = ?",
                (run_id, segment_id, model_name),
            ).fetchone()
            return row is not None

    def _row_to_label_result(self, row: sqlite3.Row) -> SegmentLabelResult:
        d = dict(row)
        d.pop("id", None)
        d["secondary_actions"] = json.loads(d.get("secondary_actions") or "[]")
        d["objects"] = json.loads(d.get("objects") or "[]")
        d["uncertainty_flags"] = json.loads(d.get("uncertainty_flags") or "[]")
        d["parsed_success"] = bool(d.get("parsed_success"))
        if d.get("timestamp") is None:
            d.pop("timestamp", None)
        return SegmentLabelResult(**d)

    # --- Runs ---

    def save_run(self, run_config: RunConfig) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs (run_id, created_at, config_json) VALUES (?, ?, ?)",
                (run_config.run_id, run_config.created_at, run_config.model_dump_json()),
            )

    def get_run(self, run_id: str) -> Optional[RunConfig]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            return RunConfig.model_validate_json(row["config_json"])

    def list_runs(self) -> list[RunConfig]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
            return [RunConfig.model_validate_json(r["config_json"]) for r in rows]

    # --- Artifact paths ---

    def run_dir(self, run_id: str) -> Path:
        d = self.artifacts_dir / "runs" / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def frames_dir(
        self, video_id: str, segment_id: str, variant_id: Optional[str] = None
    ) -> Path:
        if variant_id:
            d = self.artifacts_dir / "frames" / video_id / variant_id / segment_id
        else:
            d = self.artifacts_dir / "frames" / video_id / segment_id
        d.mkdir(parents=True, exist_ok=True)
        return d
