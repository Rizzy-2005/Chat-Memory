"""
dedup.py — SHA-256 content-hash deduplication index backed by SQLite.

Schema:
    seen_hashes (
        content_hash TEXT PRIMARY KEY,
        chunk_id     TEXT NOT NULL,
        inserted_at  TEXT NOT NULL   -- ISO-8601 UTC
    )

Public API:
    get_new_hashes(hashes, db_path)           -> list[str]   (only unseen ones)
    record_hashes(entries, db_path)           -> None        (mark as seen)
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone

from app.config import settings

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS seen_hashes (
    content_hash TEXT PRIMARY KEY,
    chunk_id     TEXT NOT NULL,
    inserted_at  TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _connect(db_path: str) -> sqlite3.Connection:
    """Open (or create) the SQLite database and ensure the table exists."""
    dir_ = os.path.dirname(os.path.abspath(db_path))
    os.makedirs(dir_, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute(_DDL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_new_hashes(
    hashes: list[str],
    db_path: str | None = None,
) -> list[str]:
    """Return only the hashes that have NOT been recorded in the dedup index.

    Order of returned hashes is preserved (same order as input).
    """
    if not hashes:
        return []

    db_path = db_path or settings.sqlite_path
    conn = _connect(db_path)
    try:
        placeholders = ",".join("?" * len(hashes))
        rows = conn.execute(
            f"SELECT content_hash FROM seen_hashes WHERE content_hash IN ({placeholders})",
            hashes,
        ).fetchall()
        seen: set[str] = {row[0] for row in rows}
        return [h for h in hashes if h not in seen]
    finally:
        conn.close()


def record_hashes(
    entries: list[tuple[str, str]],
    db_path: str | None = None,
) -> None:
    """Insert (content_hash, chunk_id) pairs into the dedup index.

    Uses INSERT OR IGNORE so re-recording the same hash is a no-op.

    Args:
        entries:  List of (content_hash, chunk_id) tuples.
        db_path:  Path to the SQLite file.  Defaults to settings.sqlite_path.
    """
    if not entries:
        return

    db_path = db_path or settings.sqlite_path
    now = datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    try:
        conn.executemany(
            "INSERT OR IGNORE INTO seen_hashes (content_hash, chunk_id, inserted_at) "
            "VALUES (?, ?, ?)",
            [(h, cid, now) for h, cid in entries],
        )
        conn.commit()
    finally:
        conn.close()
