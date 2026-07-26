"""
dedup.py — SHA-256 content-hash deduplication index backed by SQLite.
Stage 1: placeholder. Stage 2 will implement:
  - init_db(): create the SQLite table if not exists
  - is_duplicate(content_hash): bool
  - mark_stored(content_hash): void
Hash key: sha256(timestamp + sender + text)
"""
# TODO (Stage 2): implement SQLite-backed dedup logic
