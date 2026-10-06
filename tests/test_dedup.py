"""
tests/test_dedup.py — app/core/dedup.py (SQLite seen-hashes index + stats).
"""
from __future__ import annotations

import pytest

from app.core.dedup import get_meta, get_new_hashes, get_stats, record_hashes, reset_index, set_meta


@pytest.fixture
def db(tmp_path) -> str:
    return str(tmp_path / "dedup_test.db")


def _e(h: str, cid: str = "c1", ts: str = "2026-05-09T10:00:00", sender: str = "Ann"):
    return (h, cid, ts, sender)


class TestGetNewHashes:
    def test_all_new_when_db_is_empty(self, db):
        assert get_new_hashes(["a", "b"], db) == ["a", "b"]

    def test_empty_input(self, db):
        assert get_new_hashes([], db) == []

    def test_seen_filtered_order_preserved(self, db):
        record_hashes([_e("b")], db)
        assert get_new_hashes(["a", "b", "c"], db) == ["a", "c"]

    def test_many_hashes_batched(self, db):
        hashes = [f"h{i}" for i in range(2500)]  # > SQLite's bound-parameter limit
        record_hashes([_e(h) for h in hashes[:1200]], db)
        assert get_new_hashes(hashes, db) == hashes[1200:]


class TestRecordHashes:
    def test_double_record_ignored(self, db):
        record_hashes([_e("a", "c1")], db)
        record_hashes([_e("a", "c2")], db)
        assert get_new_hashes(["a"], db) == []
        assert get_stats(db)["total_messages"] == 1

    def test_reupload_reports_zero_new(self, db):
        hashes = ["h1", "h2", "h3"]
        record_hashes([_e(h) for h in get_new_hashes(hashes, db)], db)
        assert get_new_hashes(hashes, db) == []

    def test_partial_overlap(self, db):
        record_hashes([_e(h) for h in ["h1", "h2"]], db)
        assert get_new_hashes(["h1", "h2", "h3", "h4"], db) == ["h3", "h4"]


class TestStatsAndMeta:
    def test_empty_stats(self, db):
        assert get_stats(db) == {"total_messages": 0, "participants": [], "first_message": None, "last_message": None}

    def test_stats(self, db):
        record_hashes([
            _e("1", ts="2026-05-09T10:00:00", sender="Ann"),
            _e("2", ts="2026-05-10T10:00:00", sender="Bob"),
            _e("3", ts="2026-07-01T10:00:00", sender="Bob"),
        ], db)
        s = get_stats(db)
        assert s["total_messages"] == 3
        assert s["participants"] == ["Bob", "Ann"]  # most active first
        assert (s["first_message"], s["last_message"]) == ("2026-05-09", "2026-07-01")

    def test_meta_and_reset(self, db):
        set_meta("data_version", "2", db)
        record_hashes([_e("a")], db)
        reset_index(db)
        assert get_meta("data_version", db) == "2"
        assert get_new_hashes(["a"], db) == ["a"]
