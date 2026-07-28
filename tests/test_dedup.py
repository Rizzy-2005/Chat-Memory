"""
tests/test_dedup.py — unit tests for app/core/dedup.py

Covers:
    - All hashes new when DB is empty
    - Already-seen hashes are filtered out
    - Empty input is a no-op
    - Double-recording the same hash is silently ignored
    - Re-upload simulation: second upload reports zero new
    - Partial overlap: only truly new hashes returned
"""

from __future__ import annotations

import pytest

from app.core.dedup import get_new_hashes, record_hashes


# ---------------------------------------------------------------------------
# Fixture: fresh temporary SQLite path per test
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path) -> str:
    """Return a path to a fresh (non-existent) SQLite file in a temp directory."""
    return str(tmp_path / "dedup_test.db")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetNewHashes:

    def test_all_new_when_db_is_empty(self, db):
        hashes = ["aaa", "bbb", "ccc"]
        result = get_new_hashes(hashes, db)
        assert set(result) == {"aaa", "bbb", "ccc"}

    def test_empty_input_returns_empty(self, db):
        assert get_new_hashes([], db) == []

    def test_seen_hashes_are_filtered(self, db):
        record_hashes([("aaa", "chunk1"), ("bbb", "chunk1")], db)
        result = get_new_hashes(["aaa", "bbb", "ccc"], db)
        assert result == ["ccc"]

    def test_all_seen_returns_empty(self, db):
        record_hashes([("x", "c1"), ("y", "c1"), ("z", "c1")], db)
        assert get_new_hashes(["x", "y", "z"], db) == []

    def test_order_preserved(self, db):
        """Returned hashes keep the same order as the input list."""
        record_hashes([("b", "c1")], db)
        result = get_new_hashes(["a", "b", "c", "d"], db)
        assert result == ["a", "c", "d"]


class TestRecordHashes:

    def test_record_empty_is_no_op(self, db):
        record_hashes([], db)  # must not raise
        assert get_new_hashes(["aaa"], db) == ["aaa"]

    def test_double_record_ignored(self, db):
        """INSERT OR IGNORE: recording the same hash twice is safe."""
        record_hashes([("aaa", "chunk1")], db)
        record_hashes([("aaa", "chunk2")], db)  # ignored — first chunk_id wins
        # Hash is already seen regardless
        assert get_new_hashes(["aaa"], db) == []

    def test_multiple_entries_in_one_call(self, db):
        record_hashes([("h1", "c1"), ("h2", "c1"), ("h3", "c1")], db)
        assert get_new_hashes(["h1", "h2", "h3", "h4"], db) == ["h4"]


class TestReUploadSimulation:

    def test_second_upload_returns_zero_new(self, db):
        """Core dedup guarantee: uploading the same export twice → 0 new."""
        hashes = ["hash_a", "hash_b", "hash_c", "hash_d"]

        # First upload
        new_first = get_new_hashes(hashes, db)
        assert len(new_first) == 4
        record_hashes([(h, "chunk_x") for h in new_first], db)

        # Second upload — same hashes
        new_second = get_new_hashes(hashes, db)
        assert new_second == []

    def test_partial_overlap(self, db):
        """If new messages are added to an existing chat export, only new ones are ingested."""
        old_hashes = ["h1", "h2", "h3"]
        new_hashes_added = ["h4", "h5"]

        # First upload
        record_hashes([(h, "chunk_old") for h in old_hashes], db)

        # Second upload with 3 old + 2 new messages
        combined = old_hashes + new_hashes_added
        result = get_new_hashes(combined, db)
        assert set(result) == {"h4", "h5"}
