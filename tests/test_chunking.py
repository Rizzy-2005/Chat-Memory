"""
tests/test_chunking.py — unit tests for app/core/chunking.py

Covers:
    - Single message → one chunk
    - Gap below threshold → same session
    - Gap above threshold → new session
    - Messages passed out of order are sorted correctly
    - Participants are correct and deduplicated
    - chunk_id is deterministic
    - text contains sender names and message content
    - Empty input returns empty list
    - message_hashes are in chronological order
"""

from __future__ import annotations

import hashlib
from datetime import datetime

import pytest

from app.core.chunking import chunk_messages
from app.core.models import ParsedMessage


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _msg(hour: int, minute: int, sender: str, text: str) -> ParsedMessage:
    """Build a ParsedMessage with a correctly computed content_hash."""
    ts = datetime(2026, 5, 9, hour, minute)
    raw = f"{ts.isoformat()}|{sender}|{text}"
    h = hashlib.sha256(raw.encode()).hexdigest()
    return ParsedMessage(timestamp=ts, sender=sender, text=text, content_hash=h)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChunkMessages:

    def test_empty_input_returns_empty(self):
        assert chunk_messages([]) == []

    def test_single_message_creates_one_chunk(self):
        chunks = chunk_messages([_msg(10, 0, "Alice", "Hello")], gap_hours=2.0)
        assert len(chunks) == 1
        assert chunks[0].message_count == 1

    def test_small_gap_stays_in_same_session(self):
        msgs = [
            _msg(10, 0, "Alice", "Hello"),
            _msg(10, 30, "Bob", "Hi"),
            _msg(11, 45, "Alice", "How are you"),
        ]
        # Max gap is 1h15m — below the 2h threshold
        chunks = chunk_messages(msgs, gap_hours=2.0)
        assert len(chunks) == 1
        assert chunks[0].message_count == 3

    def test_big_gap_creates_new_session(self):
        msgs = [
            _msg(10, 0, "Alice", "Morning"),
            _msg(10, 5, "Bob", "Hey"),
            _msg(14, 0, "Alice", "Back again"),  # 3h55m gap → exceeds 2h
        ]
        chunks = chunk_messages(msgs, gap_hours=2.0)
        assert len(chunks) == 2
        assert chunks[0].message_count == 2
        assert chunks[1].message_count == 1

    def test_three_sessions(self):
        msgs = [
            _msg(8, 0, "Alice", "A1"),
            _msg(8, 10, "Bob", "B1"),
            _msg(12, 0, "Alice", "A2"),   # 3h50m gap
            _msg(12, 5, "Bob", "B2"),
            _msg(18, 0, "Alice", "A3"),   # 5h55m gap
        ]
        chunks = chunk_messages(msgs, gap_hours=2.0)
        assert len(chunks) == 3

    def test_messages_sorted_by_timestamp(self):
        """Out-of-order input must be sorted internally."""
        msgs = [
            _msg(10, 30, "Bob", "Second"),
            _msg(10, 0, "Alice", "First"),
        ]
        chunks = chunk_messages(msgs, gap_hours=2.0)
        assert len(chunks) == 1
        assert chunks[0].start_ts == datetime(2026, 5, 9, 10, 0)
        assert chunks[0].end_ts == datetime(2026, 5, 9, 10, 30)

    def test_participants_deduplicated_and_sorted(self):
        msgs = [
            _msg(10, 0, "Zara", "Hi"),
            _msg(10, 5, "Alice", "Hey"),
            _msg(10, 10, "Zara", "Bye"),
        ]
        chunk = chunk_messages(msgs, gap_hours=2.0)[0]
        assert chunk.participants == ["Alice", "Zara"]  # sorted, no duplicates

    def test_text_contains_sender_and_body(self):
        msgs = [_msg(10, 0, "Alice", "Hello world")]
        chunk = chunk_messages(msgs, gap_hours=2.0)[0]
        assert "Alice" in chunk.text
        assert "Hello world" in chunk.text

    def test_text_format_has_timestamp_prefix(self):
        msgs = [_msg(10, 5, "Alice", "Test")]
        chunk = chunk_messages(msgs, gap_hours=2.0)[0]
        # Must include formatted timestamp
        assert "[2026-05-09 10:05]" in chunk.text

    def test_chunk_id_is_deterministic(self):
        """Same set of messages always produces the same chunk_id."""
        msgs = [_msg(10, 0, "Alice", "Hello"), _msg(10, 5, "Bob", "Hi")]
        assert chunk_messages(msgs)[0].chunk_id == chunk_messages(msgs)[0].chunk_id

    def test_message_hashes_in_chunk(self):
        msgs = [
            _msg(10, 0, "Alice", "Hello"),
            _msg(10, 5, "Bob", "Hi"),
        ]
        chunk = chunk_messages(msgs, gap_hours=2.0)[0]
        expected_hashes = {m.content_hash for m in msgs}
        assert set(chunk.message_hashes) == expected_hashes
        assert chunk.message_count == 2

    def test_start_and_end_ts(self):
        msgs = [
            _msg(10, 0, "Alice", "First"),
            _msg(10, 30, "Bob", "Last"),
        ]
        chunk = chunk_messages(msgs, gap_hours=2.0)[0]
        assert chunk.start_ts == datetime(2026, 5, 9, 10, 0)
        assert chunk.end_ts == datetime(2026, 5, 9, 10, 30)

    def test_exact_gap_boundary(self):
        """Gap exactly equal to threshold is NOT a new session (strictly greater)."""
        msgs = [
            _msg(10, 0, "Alice", "A"),
            _msg(12, 0, "Bob", "B"),    # gap = exactly 2.0 hours
        ]
        chunks = chunk_messages(msgs, gap_hours=2.0)
        # 2.0 > 2.0 is False → same session
        assert len(chunks) == 1


class TestSizeCap:

    def test_long_session_is_split_at_message_boundaries(self):
        msgs = [_msg(10, i, "Alice", "x" * 300) for i in range(10)]  # ~3,300 chars, one session
        chunks = chunk_messages(msgs, gap_hours=2.0, max_chars=1000)
        assert len(chunks) == 4
        assert all(len(c.text) <= 1000 for c in chunks)
        assert sum(c.message_count for c in chunks) == 10
        assert chunks[0].end_ts <= chunks[1].start_ts

    def test_single_oversized_message_kept_whole(self):
        chunks = chunk_messages([_msg(10, 0, "Alice", "y" * 5000)], gap_hours=2.0, max_chars=1000)
        assert len(chunks) == 1 and "y" * 5000 in chunks[0].text

    def test_zero_disables_splitting(self):
        msgs = [_msg(10, i, "Alice", "x" * 300) for i in range(10)]
        assert len(chunk_messages(msgs, gap_hours=2.0, max_chars=0)) == 1
