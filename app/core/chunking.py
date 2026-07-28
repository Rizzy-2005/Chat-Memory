"""
chunking.py — groups parsed messages into session-based chunks.

A new session starts whenever the gap between consecutive messages (sorted by
timestamp) exceeds SESSION_GAP_HOURS.  Each session becomes one SessionChunk
whose `text` field is the concatenation of all messages in the readable format
    [YYYY-MM-DD HH:MM] Sender: body

Public API:
    chunk_messages(messages, gap_hours) -> list[SessionChunk]
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from app.config import settings
from app.core.models import ParsedMessage, SessionChunk


def _make_chunk_id(message_hashes: list[str]) -> str:
    """Deterministic chunk ID: sha256 of the sorted message hashes.

    Sorting ensures the same set of messages always produces the same ID
    regardless of the order they were passed in.
    """
    combined = "|".join(sorted(message_hashes))
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _build_chunk(session_msgs: list[ParsedMessage]) -> SessionChunk:
    """Convert a list of chronologically-ordered messages into a SessionChunk."""
    text_lines = [
        f"[{m.timestamp.strftime('%Y-%m-%d %H:%M')}] {m.sender}: {m.text}"
        for m in session_msgs
    ]
    message_hashes = [m.content_hash for m in session_msgs]
    return SessionChunk(
        chunk_id=_make_chunk_id(message_hashes),
        text="\n".join(text_lines),
        participants=sorted({m.sender for m in session_msgs}),
        start_ts=session_msgs[0].timestamp,
        end_ts=session_msgs[-1].timestamp,
        message_count=len(session_msgs),
        message_hashes=message_hashes,
    )


def chunk_messages(
    messages: list[ParsedMessage],
    gap_hours: float | None = None,
) -> list[SessionChunk]:
    """Group messages into session chunks separated by time gaps.

    Args:
        messages:   Any ordering — sorted internally by timestamp.
        gap_hours:  Gap threshold in hours.  Defaults to settings.session_gap_hours.

    Returns:
        List of SessionChunk objects in chronological order.
        Returns [] if messages is empty.
    """
    if not messages:
        return []

    if gap_hours is None:
        gap_hours = settings.session_gap_hours

    # Sort chronologically first
    sorted_msgs = sorted(messages, key=lambda m: m.timestamp)

    sessions: list[list[ParsedMessage]] = []
    current: list[ParsedMessage] = [sorted_msgs[0]]

    for msg in sorted_msgs[1:]:
        gap_seconds = (msg.timestamp - current[-1].timestamp).total_seconds()
        if gap_seconds / 3600 > gap_hours:
            # Gap exceeds threshold → flush current session, start a new one
            sessions.append(current)
            current = [msg]
        else:
            current.append(msg)

    sessions.append(current)  # flush the final session

    return [_build_chunk(s) for s in sessions]
