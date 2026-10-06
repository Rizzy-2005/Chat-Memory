"""
tests/test_store_and_api.py — real Chroma + SQLite in a temp dir (fake
embeddings): upload → dedup → stats, metadata filters, neighbours, the
data-version reset, and the tool helpers.
"""
from __future__ import annotations

import io
import zipfile
from datetime import datetime

from fastapi.testclient import TestClient

from app.agent.tools import clean_arg, parse_date_arg, resolve_senders
from app.core import dedup

CHAT = """\
01/07/26, 9:00 am - Priya: Trip planning thread
01/07/26, 9:05 am - Sam: I can do 15th to 18th August
01/07/26, 9:06 am - Priya: https://example.com/booking
01/07/26, 9:07 am - Priya: booking link above
10/07/26, 6:00 pm - Sam: Dinner on Friday?
10/07/26, 6:02 pm - Priya: Sure
20/07/26, 8:00 am - Priya Nair: different Priya here
"""


def _zip(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("WhatsApp Chat with Trip.txt", text.encode())
    return buf.getvalue()


def _client():
    from app.main import app
    return TestClient(app)


def test_upload_dedup_and_stats(store):
    with _client() as c:
        r = c.post("/upload", files={"file": ("chat.zip", _zip(CHAT), "application/zip")})
        assert r.status_code == 200
        assert r.json() == {"new_messages": 7, "new_sessions": 3, "skipped_duplicates": 0, "total_in_file": 7}

        again = c.post("/upload", files={"file": ("chat.zip", _zip(CHAT), "application/zip")}).json()
        assert again["new_messages"] == 0 and again["skipped_duplicates"] == 7

        newer = CHAT + "25/07/26, 1:00 pm - Sam: new message\n"
        r3 = c.post("/upload", files={"file": ("chat.zip", _zip(newer), "application/zip")}).json()
        assert (r3["new_messages"], r3["skipped_duplicates"]) == (1, 7)

        s = c.get("/stats").json()
        assert s["total_messages"] == 8 and s["total_sessions"] == 4
        assert s["first_message"] == "2026-07-01" and s["last_message"] == "2026-07-25"
        assert set(s["participants"]) == {"Priya", "Sam", "Priya Nair"}


def test_upload_rejects_bad_files(store):
    with _client() as c:
        assert c.post("/upload", files={"file": ("chat.txt", b"x", "text/plain")}).status_code == 400
        r = c.post("/upload", files={"file": ("chat.zip", b"nope", "application/zip")})
        assert r.status_code == 422 and "not a valid .zip" in r.json()["detail"]
        r = c.post("/upload", files={"file": ("chat.zip", _zip("just some notes\n"), "application/zip")})
        assert r.status_code == 422 and "Unrecognised chat format" in r.json()["detail"]


def test_filters_neighbours_and_range(store):
    with _client() as c:
        c.post("/upload", files={"file": ("chat.zip", _zip(CHAT), "application/zip")})

    only_sam = store.search_sessions("dinner", senders=["Sam"])
    assert only_sam and all("Sam" in s["participants"] for s in only_sam)

    july10 = store.search_sessions("anything", start=datetime(2026, 7, 10), end=datetime(2026, 7, 10, 23, 59))
    assert [m["text"] for s in july10 for m in s["messages"]] == ["Dinner on Friday?", "Sure"]

    hits = store.search_messages("booking", senders=["Priya"], k=10)
    assert hits and {h["sender"] for h in hits} == {"Priya"}

    link = next(h for h in hits if h["text"].startswith("https://"))
    window = store.message_window(link)
    assert [m["text"] for m in window] == ["I can do 15th to 18th August", "https://example.com/booking", "booking link above"]

    in_range = store.messages_in_range(datetime(2026, 7, 1), datetime(2026, 7, 9, 23, 59))
    assert len(in_range) == 4 and in_range[0]["text"] == "Trip planning thread"


def test_data_version_reset(store):
    with _client() as c:
        c.post("/upload", files={"file": ("chat.zip", _zip(CHAT), "application/zip")})
    dedup.set_meta("data_version", "old")
    store.ensure_data_version()
    assert store.collection_counts() == {"sessions": 0, "messages": 0}
    assert dedup.get_stats()["total_messages"] == 0
    assert dedup.get_meta("data_version") == store.DATA_VERSION


def test_resolve_senders():
    people = ["Priya", "Priya Nair", "Sam Thomas", "Arathi TKM CSE"]
    assert resolve_senders("priya", people) == ["Priya"]                  # exact wins
    assert resolve_senders("nair", people) == ["Priya Nair"]              # substring
    assert resolve_senders("Arathi", people) == ["Arathi TKM CSE"]
    assert resolve_senders("Arati", people) == ["Arathi TKM CSE"]         # fuzzy first name
    assert resolve_senders("Zed", people) == []


def test_arg_helpers():
    assert clean_arg("null") is None and clean_arg(" ") is None and clean_arg("Sam") == "Sam"
    assert parse_date_arg("2026-07-01") == datetime(2026, 7, 1)
    assert parse_date_arg("2026-07-01", end_of_day=True) == datetime(2026, 7, 1, 23, 59, 59)
    assert parse_date_arg("1 July 2026") == datetime(2026, 7, 1)
    assert parse_date_arg(None) is None
