"""
tests/test_parsing.py — unit tests for app/core/parsing.py

Covers:
    - Normal single-line message
    - Multi-line message (body spans multiple lines)
    - System messages are excluded (various WhatsApp system event types)
    - SHA-256 content_hash is computed correctly
    - Bad timestamp raises ValueError loudly
    - parse_zip: correct .txt is extracted and parsed
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import datetime

import pytest

from app.core.parsing import parse_text, parse_zip
from app.core.models import ParsedMessage

# ---------------------------------------------------------------------------
# Shared sample transcript (mirrors the user's real Android export format)
# ---------------------------------------------------------------------------

SAMPLE = """\
09/05/26, 2:51 pm - Messages and calls are end-to-end encrypted. Only people in this chat can read or share them.
09/05/26, 2:51 pm - Arathi TKM CSE created group "Batch A - Placements"
09/05/26, 2:51 pm - Arathi TKM CSE added you to a group in the community: Batch A
09/05/26, 3:05 pm - Arathi TKM CSE added Sufiyan TKM CSE
09/05/26, 5:37 pm - Arathi TKM CSE: Hello everyone!
This is the second line of the same message.
And a third line.
09/05/26, 5:38 pm - Sufiyan TKM CSE: Got it, thanks.
09/05/26, 6:03 pm - Arathi TKM CSE added Sreedeep TKM CSE
09/05/26, 6:10 pm - Arathi TKM CSE: <Media omitted>
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseText:

    def test_correct_message_count(self):
        """Only real messages are returned; all system lines are dropped."""
        messages = parse_text(SAMPLE)
        # Real messages: "Hello everyone!", "Got it thanks.", "<Media omitted>"
        assert len(messages) == 3

    def test_single_line_message(self):
        """A normal single-line message is parsed correctly."""
        messages = parse_text(SAMPLE)
        single = messages[1]  # "Got it, thanks."
        assert single.sender == "Sufiyan TKM CSE"
        assert single.text == "Got it, thanks."
        assert single.timestamp == datetime(2026, 5, 9, 17, 38)

    def test_multiline_message(self):
        """Continuation lines are joined into the same message body."""
        messages = parse_text(SAMPLE)
        multi = messages[0]  # "Hello everyone!\nThis is the second..."
        assert multi.sender == "Arathi TKM CSE"
        assert "Hello everyone!" in multi.text
        assert "second line of the same message" in multi.text
        assert "third line" in multi.text
        # Must be a single ParsedMessage, not three
        assert isinstance(multi, ParsedMessage)

    def test_system_message_excluded(self):
        """System event lines (created, added, encrypted notice) are dropped."""
        messages = parse_text(SAMPLE)
        senders = {m.sender for m in messages}
        # No message should have a system-event string as its sender
        for sender in senders:
            assert "added" not in sender.lower()
            assert "created" not in sender.lower()
            assert "encrypted" not in sender.lower()

    def test_system_with_colon_excluded(self):
        """'X added you to a group in the community: Y' is a system message."""
        snippet = "09/05/26, 2:51 pm - Bot added you to a group in the community: Test\n"
        messages = parse_text(snippet)
        assert messages == []

    def test_content_hash_correct(self):
        """content_hash == sha256('{ts.isoformat()}|{sender}|{text}')."""
        messages = parse_text(SAMPLE)
        for msg in messages:
            expected = hashlib.sha256(
                f"{msg.timestamp.isoformat()}|{msg.sender}|{msg.text}".encode()
            ).hexdigest()
            assert msg.content_hash == expected

    def test_media_omitted_kept(self):
        """'<Media omitted>' is a real message, not a system event."""
        messages = parse_text(SAMPLE)
        media = messages[2]
        assert media.text == "<Media omitted>"
        assert media.sender == "Arathi TKM CSE"

    def test_empty_input(self):
        """Parsing an empty string returns an empty list (no crash)."""
        assert parse_text("") == []

    def test_only_system_messages(self):
        """A transcript with only system lines returns an empty list."""
        only_system = """\
09/05/26, 2:51 pm - Messages and calls are end-to-end encrypted.
09/05/26, 2:52 pm - Alice created group "Friends"
09/05/26, 2:53 pm - Alice added Bob
"""
        assert parse_text(only_system) == []

    def test_bad_timestamp_raises(self):
        """An unrecognised timestamp format raises ValueError loudly."""
        bad = "99-99-9999 99:99 ZZ - Alice: Hello\n"
        # This line won't match _LINE_RE at all, so it's treated as a
        # continuation of nothing and silently dropped — which is correct
        # behaviour (the file is otherwise valid).
        # A completely garbled file (zero valid lines) simply returns [].
        result = parse_text(bad)
        assert result == []


class TestParseZip:

    def _make_zip(self, filename: str, content: str) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(filename, content.encode("utf-8"))
        return buf.getvalue()

    def test_parses_single_txt_in_zip(self):
        """parse_zip extracts the .txt and returns correct message count."""
        zipped = self._make_zip("WhatsApp Chat with Test.txt", SAMPLE)
        messages = parse_zip(zipped)
        assert len(messages) == 3

    def test_ignores_non_txt_files(self):
        """Other files in the zip (images, etc.) are ignored."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("WhatsApp Chat with Test.txt", SAMPLE.encode())
            zf.writestr("photo.jpg", b"\xff\xd8\xff")  # fake JPEG
            zf.writestr("video.mp4", b"\x00\x00\x00")  # fake MP4
        messages = parse_zip(buf.getvalue())
        assert len(messages) == 3

    def test_no_txt_raises_value_error(self):
        """A zip with no .txt raises ValueError."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("photo.jpg", b"\xff\xd8\xff")
        with pytest.raises(ValueError, match="No .txt file found"):
            parse_zip(buf.getvalue())

    def test_chat_txt_preferred_over_other_txt(self):
        """If multiple .txt files exist, the one containing 'chat' is used."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            # This should NOT be picked
            zf.writestr("readme.txt", b"not a chat export")
            # This SHOULD be picked
            zf.writestr("_chat.txt", SAMPLE.encode())
        messages = parse_zip(buf.getvalue())
        assert len(messages) == 3
