"""
parsing.py — WhatsApp .zip / .txt parser.

Detected export format (Android, 12-hour clock, DD/MM/YY):
    09/05/26, 2:51 pm - Sender Name: message text
    09/05/26, 2:51 pm - System message (no "Sender: " structure)
    <continuation line of the previous message>

Public API:
    parse_zip(file_bytes: bytes)  -> list[ParsedMessage]
    parse_text(content: str)      -> list[ParsedMessage]
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from datetime import datetime

from app.core.models import ParsedMessage

# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------

# Matches the timestamp prefix of a new WhatsApp line.
# Groups: (1) date  (2) time including am/pm
# Example: "09/05/26, 2:51 pm"
_LINE_RE = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s+"   # date (DD/MM/YY or DD/MM/YYYY)
    r"(\d{1,2}:\d{2}\s*[aApP][mM])"      # time  (H:MM am/pm, case-insensitive)
    r"\s+-\s+"                            # separator " - "
    r"(.+)$",                             # body (everything after " - ")
    re.DOTALL,
)

# Splits "Sender Name: message text" on the FIRST colon-space.
# If the text before the colon contains action verbs it is a system message
# (handled separately below).
_SENDER_RE = re.compile(r"^([^:]+?):\s(.+)$", re.DOTALL)

# If a "sender candidate" (text before the first ": ") matches any of these
# words it is actually a WhatsApp system action string, not a real sender.
_SYSTEM_VERB_RE = re.compile(
    r"\b("
    r"added|created|removed|left|joined|changed|pinned|deleted|missed"
    r"|waiting|encrypted|blocked|unblocked|reset|started|ended|security"
    r")\b",
    re.IGNORECASE,
)

# Unicode directional / invisible characters WhatsApp sometimes inserts.
_UNICODE_JUNK = re.compile(r"[\u200e\u200f\u202a\u202b\u202c\u202d\u202e\ufeff]")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_hash(timestamp: datetime, sender: str, text: str) -> str:
    raw = f"{timestamp.isoformat()}|{sender}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _parse_timestamp(date_str: str, time_str: str) -> datetime:
    """Parse date + time strings from the export into a datetime object.

    Tries DD/MM/YY (2-digit year) first, then DD/MM/YYYY (4-digit) as fallback.
    Raises ValueError loudly on any unrecognised format.
    """
    # Normalise: strip stray spaces, force uppercase AM/PM for strptime %p.
    time_norm = time_str.strip().upper()
    # Ensure exactly one space before AM/PM: "2:51PM" → "2:51 PM"
    time_norm = re.sub(r"(\d)(AM|PM)$", r"\1 \2", time_norm)

    raw = f"{date_str.strip()}, {time_norm}"
    for fmt in ("%d/%m/%y, %I:%M %p", "%d/%m/%Y, %I:%M %p"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue

    raise ValueError(
        f"Unrecognised timestamp format: '{raw}'. "
        "Expected 'DD/MM/YY, H:MM am/pm' (WhatsApp Android export)."
    )


def _is_system_body(body: str) -> bool:
    """Return True if the body (after ' - ') is a system message.

    Two cases:
      1. No "Sender: " structure at all        → definitely system.
      2. Has a colon, but the pseudo-sender
         contains action verbs                 → system disguised with colon.
    """
    m = _SENDER_RE.match(body)
    if not m:
        return True  # No "Sender: text" → system message

    sender_candidate = m.group(1).strip()
    if _SYSTEM_VERB_RE.search(sender_candidate):
        return True  # e.g. "Arathi TKM CSE added you to a group in the community"

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_text(content: str) -> list[ParsedMessage]:
    """Parse the raw text of a WhatsApp export and return all real messages.

    System messages are silently dropped.
    Multi-line message bodies are joined with '\\n'.

    Raises ValueError on any line whose timestamp cannot be parsed.
    """
    messages: list[ParsedMessage] = []

    # State machine: track the "in-progress" message being accumulated.
    current_ts: datetime | None = None
    current_sender: str | None = None
    current_lines: list[str] = []

    def _flush() -> None:
        """Commit the buffered message (if any) to the output list."""
        if current_sender is not None and current_lines:
            text = "\n".join(current_lines).strip()
            if text:
                messages.append(
                    ParsedMessage(
                        timestamp=current_ts,
                        sender=current_sender,
                        text=text,
                        content_hash=_compute_hash(current_ts, current_sender, text),
                    )
                )

    for raw_line in content.splitlines():
        line = _UNICODE_JUNK.sub("", raw_line)  # strip invisible chars

        m = _LINE_RE.match(line)
        if not m:
            # ── Continuation line: belongs to the current message ──────────
            if current_sender is not None:
                current_lines.append(line)
            # If current_sender is None we are inside a system message block;
            # continuation lines of system messages are also dropped.
            continue

        # ── New timestamped line detected ───────────────────────────────────
        _flush()
        date_str, time_str, body = m.group(1), m.group(2), m.group(3)
        body = body.strip()

        ts = _parse_timestamp(date_str, time_str)

        if _is_system_body(body):
            # System message — reset state, do not store.
            current_ts = None
            current_sender = None
            current_lines = []
        else:
            sender_m = _SENDER_RE.match(body)
            # _is_system_body returned False so sender_m is guaranteed to match.
            current_ts = ts
            current_sender = sender_m.group(1).strip()
            current_lines = [sender_m.group(2).strip()]

    _flush()  # commit the last message
    return messages


def parse_zip(file_bytes: bytes) -> list[ParsedMessage]:
    """Extract the single .txt from a WhatsApp .zip export and parse it.

    Raises:
        ValueError – if no .txt file is found inside the zip.
    """
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
        txt_names = [n for n in zf.namelist() if n.lower().endswith(".txt")]

        if not txt_names:
            raise ValueError(
                "No .txt file found inside the uploaded .zip. "
                "Please upload a WhatsApp 'Export chat' zip file."
            )

        # WhatsApp always produces exactly one .txt (_chat.txt or
        # "WhatsApp Chat with X.txt").  If somehow multiple exist, pick the
        # one whose name contains 'chat' (case-insensitive); otherwise first.
        chat_txt = next(
            (n for n in txt_names if "chat" in n.lower()), txt_names[0]
        )

        raw_bytes = zf.read(chat_txt)

    # WhatsApp exports are UTF-8; some devices prepend a BOM → use 'utf-8-sig'.
    content = raw_bytes.decode("utf-8-sig")
    return parse_text(content)
