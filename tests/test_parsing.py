"""
tests/test_parsing.py — app/core/parsing.py

Covers: single and multi-line messages, structural system-message removal
(including the "X added you to a group: Y" colon case), media placeholders,
hashes (incl. repeated identical messages), Android 12h/24h, iOS and
month-first formats, loud failure on unknown formats, and zip handling.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import datetime

import pytest

from app.core.parsing import parse_text, parse_zip

# Mirrors the real Android export (note the narrow no-break space before "pm").
SAMPLE = """\
09/05/26, 2:51 pm - Messages and calls are end-to-end encrypted. Only people in this chat can read or share them.
09/05/26, 2:51 pm - Arathi TKM CSE created group "Batch A - Placements"
09/05/26, 2:51 pm - Arathi TKM CSE added you to a group in the community: Batch A
09/05/26, 3:05 pm - Arathi TKM CSE added Sufiyan TKM CSE
09/05/26, 5:37 pm - Arathi TKM CSE: Hello everyone!
This is the second line of the same message.
And a third line.
09/05/26, 5:38 pm - Sufiyan TKM CSE: Got it, thanks.
09/05/26, 6:03 pm - Arathi TKM CSE added Sreedeep TKM CSE
09/05/26, 6:10 pm - Arathi TKM CSE: <Media omitted>
09/05/26, 6:11 pm - Sufiyan TKM CSE: This message was deleted
09/05/26, 6:12 pm - Sufiyan TKM CSE: Meet at 5? <This message was edited>
"""


def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


class TestParseText:
    def test_only_real_text_messages_kept(self):
        msgs = parse_text(SAMPLE)
        assert [m.text.split("\n")[0] for m in msgs] == ["Hello everyone!", "Got it, thanks.", "Meet at 5?"]

    def test_single_line_message(self):
        m = parse_text(SAMPLE)[1]
        assert m.sender == "Sufiyan TKM CSE"
        assert m.timestamp == datetime(2026, 5, 9, 17, 38)

    def test_multiline_message_joined(self):
        m = parse_text(SAMPLE)[0]
        assert m.text == "Hello everyone!\nThis is the second line of the same message.\nAnd a third line."

    def test_system_message_with_colon_dropped(self):
        senders = {m.sender for m in parse_text(SAMPLE)}
        assert senders == {"Arathi TKM CSE", "Sufiyan TKM CSE"}

    def test_system_actor_who_never_wrote(self):
        text = "09/05/26, 2:51 pm - Bot Admin added you to a group in the community: Test\n09/05/26, 2:52 pm - Ann: hi\n"
        assert [m.sender for m in parse_text(text)] == ["Ann"]

    def test_similar_real_names_both_kept(self):
        text = "09/05/26, 2:51 pm - Akshay: hi\n09/05/26, 2:52 pm - Akshay S TKM CSE: hello\n"
        assert [m.sender for m in parse_text(text)] == ["Akshay", "Akshay S TKM CSE"]

    def test_edited_suffix_stripped(self):
        assert parse_text(SAMPLE)[-1].text == "Meet at 5?"

    def test_content_hash(self):
        for m in parse_text(SAMPLE):
            expected = hashlib.sha256(f"{m.timestamp.isoformat()}|{m.sender}|{m.text}".encode()).hexdigest()
            assert m.content_hash == expected

    def test_repeated_identical_messages_get_distinct_stable_hashes(self):
        text = "09/05/26, 2:51 pm - Ann: ok\n09/05/26, 2:51 pm - Ann: ok\n"
        first, second = parse_text(text)
        assert first.content_hash != second.content_hash
        assert [m.content_hash for m in parse_text(text)] == [first.content_hash, second.content_hash]

    def test_empty_input(self):
        assert parse_text("") == []

    def test_only_system_messages(self):
        text = "09/05/26, 2:51 pm - Messages are end-to-end encrypted.\n09/05/26, 2:52 pm - Alice created group \"F\"\n"
        assert parse_text(text) == []

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError, match="Unrecognised chat format"):
            parse_text("hello there\nthis is not a whatsapp export\n")

    def test_android_24h(self):
        m = parse_text("21/07/2026, 14:05 - Ann: lunch?\n")[0]
        assert m.timestamp == datetime(2026, 7, 21, 14, 5)

    def test_ios_format(self):
        text = "[09/05/26, 2:51:03 PM] Ann: hi\n[09/05/26, 2:52:10 PM] Bob: hey\n"
        msgs = parse_text(text)
        assert [(m.sender, m.timestamp) for m in msgs] == [
            ("Ann", datetime(2026, 5, 9, 14, 51)),
            ("Bob", datetime(2026, 5, 9, 14, 52)),
        ]

    def test_month_first_detected(self):
        text = "5/9/26, 2:51 pm - Ann: hi\n12/31/26, 9:00 am - Ann: new year eve\n"
        assert parse_text(text)[0].timestamp == datetime(2026, 5, 9, 14, 51)

    def test_day_first_detected(self):
        text = "13/05/26, 9:00 am - Ann: hi\n"
        assert parse_text(text)[0].timestamp == datetime(2026, 5, 13, 9, 0)

    def test_forced_date_order(self):
        text = "05/09/26, 9:00 am - Ann: hi\n"
        assert parse_text(text, date_order="MDY")[0].timestamp == datetime(2026, 5, 9, 9, 0)

    def test_impossible_date_raises(self):
        with pytest.raises(ValueError):
            parse_text("13/13/26, 9:00 am - Ann: hi\n")


class TestParseZip:
    def test_parses_txt_and_ignores_media(self):
        data = _zip({"WhatsApp Chat with Test.txt": SAMPLE.encode(), "photo.jpg": b"\xff\xd8", "v.mp4": b"\0"})
        assert len(parse_zip(data)) == 3

    def test_bom_handled(self):
        data = _zip({"_chat.txt": "﻿09/05/26, 2:51 pm - Ann: hi\n".encode("utf-8")})
        assert parse_zip(data)[0].sender == "Ann"

    def test_no_txt_raises(self):
        with pytest.raises(ValueError, match="No .txt file found"):
            parse_zip(_zip({"photo.jpg": b"\xff\xd8"}))

    def test_not_a_zip_raises(self):
        with pytest.raises(ValueError, match="not a valid .zip"):
            parse_zip(b"definitely not a zip")

    def test_chat_txt_preferred(self):
        data = _zip({"readme.txt": b"not a chat export", "_chat.txt": SAMPLE.encode()})
        assert len(parse_zip(data)) == 3
