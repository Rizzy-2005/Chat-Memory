"""
dedup.py — SHA-256 content-hash deduplication index backed by SQLite.

Schema:
    seen_hashes (
        content_hash TEXT PRIMARY KEY,
        chunk_id     TEXT NOT NULL,
        timestamp    TEXT NOT NULL,   -- message time, ISO-8601
        sender       TEXT NOT NULL
    )
    meta (key TEXT PRIMARY KEY, value TEXT)   -- e.g. data_version

Because every stored message is recorded here with its sender and time, this
table also answers the cheap /stats questions (counts, participants, span).

Public API:
    get_new_hashes(hashes, db_path)    -> list[str]   (only unseen ones)
    record_hashes(entries, db_path)    -> None        (mark as seen)
    get_stats(db_path)                 -> dict
    get_meta / set_meta / reset_index
"""

from __future__ import annotations

import os
import sqlite3

from app.config import settings

_DDL = """
CREATE TABLE IF NOT EXISTS seen_hashes (
    content_hash TEXT PRIMARY KEY,
    chunk_id     TEXT NOT NULL,
    timestamp    TEXT NOT NULL,
    sender       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def _connect(db_path: str | None) -> sqlite3.Connection:
    """Open (or create) the SQLite database and ensure the tables exist."""
    db_path = db_path or settings.sqlite_path
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.executescript(_DDL)
    return conn


def get_new_hashes(hashes: list[str], db_path: str | None = None) -> list[str]:
    """Return only the hashes that have NOT been recorded (input order kept)."""
    if not hashes:
        return []
    conn = _connect(db_path)
    try:
        seen: set[str] = set()
        # SQLite caps bound parameters per statement — query in batches.
        for i in range(0, len(hashes), 900):
            batch = hashes[i : i + 900]
            placeholders = ",".join("?" * len(batch))
            rows = conn.execute(
                f"SELECT content_hash FROM seen_hashes WHERE content_hash IN ({placeholders})",
                batch,
            ).fetchall()
            seen.update(r[0] for r in rows)
        return [h for h in hashes if h not in seen]
    finally:
        conn.close()


def record_hashes(entries: list[tuple[str, str, str, str]], db_path: str | None = None) -> None:
    """Insert (content_hash, chunk_id, timestamp_iso, sender) rows.
    INSERT OR IGNORE makes re-recording the same hash a no-op."""
    if not entries:
        return
    conn = _connect(db_path)
    try:
        conn.executemany(
            "INSERT OR IGNORE INTO seen_hashes (content_hash, chunk_id, timestamp, sender) "
            "VALUES (?, ?, ?, ?)",
            entries,
        )
        conn.commit()
    finally:
        conn.close()


def get_stats(db_path: str | None = None) -> dict:
    """Message count, participants (by message count desc) and date span."""
    conn = _connect(db_path)
    try:
        total, first, last = conn.execute(
            "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM seen_hashes"
        ).fetchone()
        participants = [
            r[0]
            for r in conn.execute(
                "SELECT sender, COUNT(*) AS n FROM seen_hashes GROUP BY sender ORDER BY n DESC, sender"
            ).fetchall()
        ]
    finally:
        conn.close()
    return {
        "total_messages": total or 0,
        "participants": participants,
        "first_message": first[:10] if first else None,
        "last_message": last[:10] if last else None,
    }


def get_meta(key: str, db_path: str | None = None) -> str | None:
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def set_meta(key: str, value: str, db_path: str | None = None) -> None:
    conn = _connect(db_path)
    try:
        conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
        conn.commit()
    finally:
        conn.close()


def reset_index(db_path: str | None = None) -> None:
    """Drop every recorded hash (used when the stored data format changes)."""
    conn = _connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS seen_hashes")
        conn.executescript(_DDL)
        conn.commit()
    finally:
        conn.close()
