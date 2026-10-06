"""
tests/test_summarize.py — POST /summarize (validation, empty range, decision
citations mapped to real messages, map-reduce over large ranges).
"""
from __future__ import annotations

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agent import summarize as summ
from app.api.summarize import router

app_for_testing = FastAPI()
app_for_testing.include_router(router)
client = TestClient(app_for_testing)

MSGS = [
    {"content_hash": "h1", "timestamp": "2026-05-09T17:37:00", "sender": "Arathi", "text": "Let's meet next Friday for placement prep.", "chunk_id": "c"},
    {"content_hash": "h2", "timestamp": "2026-05-09T17:40:00", "sender": "Sam", "text": "Agreed.", "chunk_id": "c"},
]


def test_validation():
    assert client.post("/summarize", json={"start_date": "", "end_date": ""}).status_code == 400
    assert client.post("/summarize", json={"start_date": "2026-13-01", "end_date": "2026-05-01"}).status_code == 400
    r = client.post("/summarize", json={"start_date": "2026-06-01", "end_date": "2026-05-01"})
    assert r.status_code == 400 and "on or before" in r.json()["detail"]


@patch("app.agent.summarize.vector_store.messages_in_range", return_value=[])
def test_empty_range(_):
    r = client.post("/summarize", json={"start_date": "2026-01-01", "end_date": "2026-01-05"})
    assert r.status_code == 200
    d = r.json()
    assert d["found"] is False and d["decisions_and_plans"] == [] and "No messages" in d["narrative_summary"]


@patch("app.agent.summarize.vector_store.messages_in_range", return_value=MSGS)
def test_decisions_cite_real_messages(_):
    seen_prompts = []

    def fake(schema, build):
        prompt, payload = build(30000)
        seen_prompts.append(prompt)
        return summ._SummaryOut(
            narrative_summary="Arathi proposed placement prep; Sam agreed.",
            decisions_and_plans=[
                summ._Decision(description="Placement prep meeting on Fri 15 May", message_id=1),
                summ._Decision(description="Invented decision", message_id=42),  # dropped
            ],
        ), payload

    with patch.object(summ, "invoke_structured_budgeted", fake):
        r = client.post("/summarize", json={"start_date": "2026-05-01", "end_date": "2026-05-31"})
    d = r.json()
    assert r.status_code == 200 and d["found"] is True and d["message_count"] == 2
    assert d["decisions_and_plans"] == [{
        "description": "Placement prep meeting on Fri 15 May",
        "timestamp": "2026-05-09T17:37:00",
        "sender": "Arathi",
        "excerpt": "Let's meet next Friday for placement prep.",
    }]
    assert "2026-05-15" in seen_prompts[0]  # deterministic date note reached the prompt


def test_large_range_is_map_reduced():
    many = [{**MSGS[0], "content_hash": f"h{i}", "timestamp": f"2026-05-{1 + i // 10:02d}T10:{i % 10:02d}:00", "text": "x" * 500} for i in range(60)]
    calls = []

    def fake_budgeted(schema, build):
        prompt, payload = build(10000)
        calls.append(len(payload))
        return summ._SummaryOut(narrative_summary=f"part {len(calls)}", decisions_and_plans=[]), payload

    with patch.object(summ.vector_store, "messages_in_range", return_value=many), \
         patch.object(summ, "context_budget", return_value=10000), \
         patch.object(summ, "invoke_structured_budgeted", fake_budgeted), \
         patch.object(summ, "invoke_structured", return_value=summ._MergeOut(narrative_summary="merged")) as merge:
        r = client.post("/summarize", json={"start_date": "2026-05-01", "end_date": "2026-05-31"})
    assert r.json()["narrative_summary"] == "merged"
    assert sum(calls) == 60 and len(calls) > 1
    merge.assert_called_once()


@patch("app.agent.summarize.vector_store.messages_in_range", return_value=MSGS)
def test_rate_limit_is_429(_):
    from app.agent.llm import LLMRateLimited

    with patch.object(summ, "invoke_structured_budgeted", side_effect=LLMRateLimited("429")):
        r = client.post("/summarize", json={"start_date": "2026-05-01", "end_date": "2026-05-31"})
    assert r.status_code == 429 and "quota" in r.json()["detail"]
