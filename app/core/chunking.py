"""
chunking.py — groups parsed messages into session-based chunks.

A new session starts whenever the gap between consecutive messages (sorted by
timestamp) exceeds SESSION_GAP_HOURS.  A session whose text would exceed
MAX_CHUNK_CHARS is split at message boundaries, because all-MiniLM-L6-v2 only
reads roughly the first 256 word-pieces and would ignore the rest.  Each
(sub-)session becomes one SessionChunk whose `text` is the concatenation of
its messages in the readable format
    [YYYY-MM-DD HH:MM] Sender: body

Public API:
    chunk_messages(messages, gap_hours, max_chars) -> list[SessionChunk]
"""

from __future__ import annotations

import hashlib

from app.config import settings
from app.core.models import ParsedMessage, SessionChunk


def format_line(m: ParsedMessage) -> str:
    """The canonical one-message text representation used everywhere."""
    return f"[{m.timestamp.strftime('%Y-%m-%d %H:%M')}] {m.sender}: {m.text}"


def _make_chunk_id(message_hashes: list[str]) -> str:
    """Deterministic chunk ID: sha256 of the sorted message hashes."""
    combined = "|".join(sorted(message_hashes))
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _build_chunk(session_msgs: list[ParsedMessage]) -> SessionChunk:
    """Convert a list of chronologically-ordered messages into a SessionChunk."""
    message_hashes = [m.content_hash for m in session_msgs]
    return SessionChunk(
        chunk_id=_make_chunk_id(message_hashes),
        text="\n".join(format_line(m) for m in session_msgs),
        participants=sorted({m.sender for m in session_msgs}),
        start_ts=session_msgs[0].timestamp,
        end_ts=session_msgs[-1].timestamp,
        message_count=len(session_msgs),
        message_hashes=message_hashes,
    )


def _split_by_size(session: list[ParsedMessage], max_chars: int) -> list[list[ParsedMessage]]:
    """Greedily pack messages into pieces whose text stays within max_chars.
    A single message longer than max_chars becomes its own piece."""
    pieces: list[list[ParsedMessage]] = []
    current: list[ParsedMessage] = []
    size = 0
    for m in session:
        line_len = len(format_line(m)) + 1  # +1 for the joining newline
        if current and size + line_len > max_chars:
            pieces.append(current)
            current, size = [], 0
        current.append(m)
        size += line_len
    if current:
        pieces.append(current)
    return pieces


def chunk_messages(
    messages: list[ParsedMessage],
    gap_hours: float | None = None,
    max_chars: int | None = None,
) -> list[SessionChunk]:
    """Group messages into session chunks separated by time gaps.

    Args:
        messages:   Any ordering — sorted internally by timestamp.
        gap_hours:  Gap threshold in hours.  Defaults to settings.session_gap_hours.
        max_chars:  Size cap per chunk.  Defaults to settings.max_chunk_chars.
                    Pass 0 to disable splitting.

    Returns:
        List of SessionChunk objects in chronological order ([] for no input).
    """
    if not messages:
        return []

    if gap_hours is None:
        gap_hours = settings.session_gap_hours
    if max_chars is None:
        max_chars = settings.max_chunk_chars

    sorted_msgs = sorted(messages, key=lambda m: m.timestamp)

    sessions: list[list[ParsedMessage]] = []
    current: list[ParsedMessage] = [sorted_msgs[0]]
    for msg in sorted_msgs[1:]:
        gap_seconds = (msg.timestamp - current[-1].timestamp).total_seconds()
        if gap_seconds / 3600 > gap_hours:
            sessions.append(current)
            current = [msg]
        else:
            current.append(msg)
    sessions.append(current)

    chunks: list[SessionChunk] = []
    for session in sessions:
        pieces = _split_by_size(session, max_chars) if max_chars else [session]
        chunks.extend(_build_chunk(p) for p in pieces)
    return chunks
