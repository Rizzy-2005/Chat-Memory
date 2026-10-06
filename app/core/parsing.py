"""
parsing.py — WhatsApp .zip / .txt parser.

Supported export line formats (header line of each message):
    Android 12h:  09/05/26, 2:51 pm - Sender Name: message text
    Android 24h:  09/05/2026, 14:51 - Sender Name: message text
    iOS:          [09/05/26, 2:51:03 PM] Sender Name: message text
Date separators '/', '.', '-' are accepted.  Day-first vs month-first is
detected from the whole file (any first field > 12 → day-first, any second
field > 12 → month-first, otherwise day-first) or forced with DATE_ORDER.

Lines without a timestamp header are continuation lines of the previous
message.  Timestamped lines without a "Name: " structure are system events
and are dropped.  Media placeholders ("<Media omitted>", "This message was
deleted", ...) are dropped too — the app is text-only by design.

A file in which no line matches any known header format raises ValueError.

Public API:
    parse_zip(file_bytes: bytes)  -> list[ParsedMessage]
    parse_text(content: str)      -> list[ParsedMessage]
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from collections import Counter
from datetime import datetime

from app.core.models import ParsedMessage

# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------

_DATE = r"\d{1,4}[./-]\d{1,2}[./-]\d{1,4}"
_TIME = r"\d{1,2}[:.]\d{2}(?:[:.]\d{2})?(?:\s*[aApP]\.?\s?[mM]\.?)?"

# Android: "09/05/26, 2:51 pm - body"
_ANDROID_RE = re.compile(rf"^(?P<date>{_DATE}),?\s+(?P<time>{_TIME})\s+[-–]\s+(?P<body>.*)$")
# iOS: "[09/05/26, 2:51:03 PM] body"
_IOS_RE = re.compile(rf"^\[(?P<date>{_DATE}),?\s+(?P<time>{_TIME})\]\s*(?P<body>.*)$")

# "Sender Name: message text" — split on the FIRST colon followed by a space.
_SENDER_RE = re.compile(r"^([^:]{1,80}?):\s(.*)$", re.DOTALL)

# Fallback system detection for actors who never sent a message themselves.
_SYSTEM_VERB_RE = re.compile(
    r"\b(added|created|removed|left|joined|changed|pinned|deleted|missed|"
    r"blocked|unblocked|reset|started|ended|turned|invited|promoted|demoted)\b",
    re.IGNORECASE,
)

# Unicode directional / invisible characters WhatsApp sometimes inserts.
_UNICODE_JUNK = re.compile(r"[‎‏‪‫‬‭‮﻿]")

# Bodies that carry no text content (media / deleted placeholders).
_PLACEHOLDERS = {
    "<media omitted>",
    "this message was deleted",
    "you deleted this message",
    "null",
    "waiting for this message. this may take a while.",
    "image omitted",
    "video omitted",
    "audio omitted",
    "sticker omitted",
    "gif omitted",
    "document omitted",
    "contact card omitted",
    "<view once voice message omitted>",
}
_PLACEHOLDER_RE = re.compile(r"^(<attached: [^>]+>|.+ \(file attached\)|<.+ omitted>)$", re.IGNORECASE)
_EDITED_SUFFIX_RE = re.compile(r"\s*<This message was edited>\s*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_hash(timestamp: datetime, sender: str, text: str, occurrence: int = 0) -> str:
    """sha256 of timestamp|sender|text.  WhatsApp timestamps have minute
    precision, so genuinely repeated identical messages in the same minute get
    an occurrence suffix to keep them distinct (and still stable on re-upload)."""
    raw = f"{timestamp.isoformat()}|{sender}|{text}"
    if occurrence:
        raw += f"|{occurrence}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _match_header(line: str) -> re.Match | None:
    return _ANDROID_RE.match(line) or _IOS_RE.match(line)


def _split_date(date_str: str) -> tuple[int, int, int]:
    a, b, c = (int(p) for p in re.split(r"[./-]", date_str))
    return a, b, c


def _detect_date_order(date_strs: list[str], forced: str = "auto") -> str:
    """Return 'YMD', 'DMY' or 'MDY' for the whole file."""
    forced = (forced or "auto").upper()
    if forced in ("DMY", "MDY", "YMD"):
        return forced
    parts = [_split_date(d) for d in date_strs]
    if any(a > 31 for a, _, _ in parts):
        return "YMD"
    if any(a > 12 for a, _, _ in parts):
        return "DMY"
    if any(b > 12 for _, b, _ in parts):
        return "MDY"
    return "DMY"


def _parse_timestamp(date_str: str, time_str: str, order: str) -> datetime:
    """Build a datetime from the header's date and time strings.

    Raises ValueError (with the offending text) on impossible values.
    """
    a, b, c = _split_date(date_str)
    if order == "YMD":
        year, month, day = a, b, c
    elif order == "MDY":
        month, day, year = a, b, c
    else:
        day, month, year = a, b, c
    if year < 100:
        year += 2000

    t = re.sub(r"\s+", " ", time_str.strip()).upper().replace(".", ":")
    ampm = None
    m = re.search(r"([AP])\s?:?M:?$", t)
    if m:
        ampm = m.group(1)
        t = t[: m.start()].strip().rstrip(":")
    nums = [int(x) for x in t.split(":") if x]
    hour, minute = nums[0], nums[1]
    if ampm:
        if not 1 <= hour <= 12:
            raise ValueError(f"Invalid 12-hour time '{time_str}'")
        hour = hour % 12 + (12 if ampm == "P" else 0)

    try:
        # Seconds (iOS) are dropped so the same message hashes identically
        # across exports and matches Android's minute precision.
        return datetime(year, month, day, hour, minute)
    except ValueError as exc:
        raise ValueError(
            f"Unrecognised timestamp '{date_str}, {time_str}' ({exc}). "
            "Set DATE_ORDER=DMY or MDY if your phone uses a different date order."
        ) from exc


def _is_placeholder(text: str) -> bool:
    t = text.strip()
    return t.lower() in _PLACEHOLDERS or bool(_PLACEHOLDER_RE.match(t))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_text(content: str, date_order: str = "auto") -> list[ParsedMessage]:
    """Parse the raw text of a WhatsApp export and return all real messages.

    System messages and media placeholders are dropped.  Multi-line message
    bodies are joined with '\\n'.  Returns [] for empty input.

    Raises ValueError if the text is non-empty but contains no recognisable
    WhatsApp header line, or if a timestamp cannot be interpreted.
    """
    lines = [_UNICODE_JUNK.sub("", raw) for raw in content.splitlines()]

    # ── Pass 1: group lines into (date, time, body_lines) records ───────────
    records: list[tuple[str, str, list[str]]] = []
    for line in lines:
        m = _match_header(line)
        if m:
            records.append((m.group("date"), m.group("time"), [m.group("body").strip()]))
        elif records:
            records[-1][2].append(line)
        # Lines before the first header (rare) are ignored.

    if not records:
        if any(line.strip() for line in lines):
            raise ValueError(
                "Unrecognised chat format: no line looks like a WhatsApp message "
                "(expected e.g. '09/05/26, 2:51 pm - Name: text'). Please upload an "
                "unmodified 'Export chat' file."
            )
        return []

    order = _detect_date_order([r[0] for r in records], date_order)

    # ── Pass 2: split bodies into sender / text, collect real sender names ──
    candidates: list[tuple[datetime, str | None, list[str]]] = []
    for date_str, time_str, body_lines in records:
        ts = _parse_timestamp(date_str, time_str, order)
        first = body_lines[0]
        sm = _SENDER_RE.match(first)
        if sm:
            candidates.append((ts, sm.group(1).strip(), [sm.group(2)] + body_lines[1:]))
        else:
            candidates.append((ts, None, body_lines))  # no "Name: " → system event

    sender_counts = Counter(s for _, s, _ in candidates if s)
    real_senders = set(sender_counts)

    def _is_system_sender(sender: str) -> bool:
        # Structural: "Alice added you to a group: X" is another sender's
        # name followed by a lowercase action phrase.
        for other in real_senders:
            if other != sender and sender.startswith(other + " "):
                rest = sender[len(other) + 1:]
                if rest[:1].islower() and _SYSTEM_VERB_RE.search(rest):
                    return True
        # Fallback for actors who never wrote a message: a long "name" that
        # reads like an action sentence and appears only once.
        return (
            sender_counts[sender] == 1
            and len(sender.split()) >= 3
            and bool(_SYSTEM_VERB_RE.search(sender))
        )

    system_senders = {s for s in real_senders if _is_system_sender(s)}

    # ── Pass 3: build messages ───────────────────────────────────────────────
    messages: list[ParsedMessage] = []
    seen: Counter[tuple] = Counter()
    for ts, sender, body_lines in candidates:
        if sender is None or sender in system_senders:
            continue
        text = _EDITED_SUFFIX_RE.sub("", "\n".join(body_lines)).strip()
        if not text or _is_placeholder(text):
            continue
        key = (ts, sender, text)
        occurrence = seen[key]
        seen[key] += 1
        messages.append(
            ParsedMessage(
                timestamp=ts,
                sender=sender,
                text=text,
                content_hash=_compute_hash(ts, sender, text, occurrence),
            )
        )
    return messages


def parse_zip(file_bytes: bytes, date_order: str = "auto", max_txt_bytes: int = 50 * 1024 * 1024) -> list[ParsedMessage]:
    """Extract the chat .txt from a WhatsApp .zip export and parse it.
    Every other file in the zip (media) is ignored.

    Raises:
        ValueError – not a zip, no .txt inside, .txt too large, or bad format.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(file_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError("The uploaded file is not a valid .zip archive.") from exc

    with zf:
        txt_infos = [i for i in zf.infolist() if i.filename.lower().endswith(".txt")]
        if not txt_infos:
            raise ValueError(
                "No .txt file found inside the uploaded .zip. "
                "Please upload a WhatsApp 'Export chat' zip file."
            )
        # WhatsApp produces exactly one .txt (_chat.txt or "WhatsApp Chat with X.txt").
        chat_info = next((i for i in txt_infos if "chat" in i.filename.lower()), txt_infos[0])
        if chat_info.file_size > max_txt_bytes:
            raise ValueError("The chat text file inside the zip is too large.")
        raw_bytes = zf.read(chat_info)

    # WhatsApp exports are UTF-8; some devices prepend a BOM → 'utf-8-sig'.
    content = raw_bytes.decode("utf-8-sig", errors="replace")
    return parse_text(content, date_order=date_order)
