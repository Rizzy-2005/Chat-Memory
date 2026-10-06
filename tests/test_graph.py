"""
tests/test_graph.py — the LangGraph agent (app/agent/graph.py).

The model is replaced by scripted replies and retrieval by fake tool results,
so these tests check the plumbing: routing, the tool loop, the round limit,
fallbacks, citation mapping and the not-found contract.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage

from app.agent import graph
from app.agent.graph import NOT_FOUND, ComposerOutput, run_query
from app.agent.llm import LLMRateLimited

MSGS = [
    {"content_hash": "h1", "timestamp": "2026-07-02T19:14:00", "sender": "Priya", "text": "let's book 15th to 18th", "chunk_id": "c1"},
    {"content_hash": "h2", "timestamp": "2026-07-02T19:20:00", "sender": "You", "text": "done, booking tonight", "chunk_id": "c1"},
]


def call(name: str, args: dict, i: int = 0) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}_{i}", "type": "tool_call"}])


class Script:
    """Feeds scripted agent replies, records what the agent saw."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0

    def __call__(self, tools, messages):
        self.calls += 1
        reply = self.replies.pop(0) if self.replies else AIMessage(content="done")
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def fake_retrieval():
    """search_sessions / message_pinpoint return MSGS; stats are fixed."""
    with patch("app.agent.tools.vector_store.search_sessions", return_value=[
        {"chunk_id": "c1", "start_ts": MSGS[0]["timestamp"], "end_ts": MSGS[1]["timestamp"],
         "participants": ["Priya", "You"], "score": 0.1, "messages": MSGS}
    ]) as ss, patch("app.agent.tools.vector_store.search_messages", return_value=[MSGS[0]]), \
         patch("app.agent.tools.vector_store.message_window", side_effect=lambda m: [m]), \
         patch("app.agent.tools.dedup.get_stats", return_value={"total_messages": 2, "participants": ["Priya", "You"],
                                                               "first_message": "2026-07-02", "last_message": "2026-07-02"}), \
         patch("app.agent.graph.dedup.get_stats", return_value={"total_messages": 2, "participants": ["Priya", "You"],
                                                               "first_message": "2026-07-02", "last_message": "2026-07-02"}):
        yield ss


def composer_returns(out: ComposerOutput):
    def fake(schema, build):
        _prompt, payload = build(30000)
        return out, payload
    return fake


def test_sender_and_date_filters_in_one_call(fake_retrieval):
    agent = Script(call("search_sessions", {"question": "the trip", "sender": "priya", "start_date": "2026-07-01", "end_date": "2026-07-15"}))
    with patch.object(graph, "invoke_with_tools", agent), patch.object(
        graph, "invoke_structured_budgeted", composer_returns(ComposerOutput(found=True, answer="Booked 15–18 Aug.", citation_ids=[2, 1]))
    ):
        out = run_query("what did Priya say about the trip in early July?")

    args = fake_retrieval.call_args.args
    assert args[1] == ["Priya"]                                      # fuzzy name → exact participant
    assert args[2].date().isoformat() == "2026-07-01" and args[3].date().isoformat() == "2026-07-15"
    assert out["found"] is True
    assert out["mode_used"] == "search_sessions"
    assert out["filters"] == {"sender": "Priya", "start_date": "2026-07-01", "end_date": "2026-07-15"}
    # citations come from the real messages, chronological
    assert [(c["sender"], c["timestamp"]) for c in out["citations"]] == [("Priya", MSGS[0]["timestamp"]), ("You", MSGS[1]["timestamp"])]
    assert out["citations"][0]["excerpt"] == MSGS[0]["text"]
    assert agent.calls == 2  # tool round, then "done"


def test_pinpoint_route(fake_retrieval):
    agent = Script(call("message_pinpoint", {"question": "shared the link"}))
    with patch.object(graph, "invoke_with_tools", agent), patch.object(
        graph, "invoke_structured_budgeted", composer_returns(ComposerOutput(found=True, answer="Priya did.", citation_ids=[1]))
    ):
        out = run_query("find the message where someone shared the link")
    assert out["mode_used"] == "message_pinpoint"
    assert out["tool_calls"] == ["message_pinpoint"]


def test_invented_citation_ids_mean_not_found(fake_retrieval):
    agent = Script(call("search_sessions", {"question": "x"}))
    with patch.object(graph, "invoke_with_tools", agent), patch.object(
        graph, "invoke_structured_budgeted", composer_returns(ComposerOutput(found=True, answer="Something", citation_ids=[99]))
    ):
        out = run_query("q")
    assert out["found"] is False and out["answer"] == NOT_FOUND and out["citations"] == []


def test_composer_not_found(fake_retrieval):
    agent = Script(call("search_sessions", {"question": "cake"}))
    with patch.object(graph, "invoke_with_tools", agent), patch.object(
        graph, "invoke_structured_budgeted", composer_returns(ComposerOutput(found=False, answer=NOT_FOUND, citation_ids=[]))
    ):
        out = run_query("chocolate cake recipe?")
    assert out == {**out, "found": False, "answer": NOT_FOUND, "citations": []}


def test_nothing_retrieved_skips_composer_llm(fake_retrieval):
    fake_retrieval.return_value = []
    agent = Script(call("search_sessions", {"question": "x"}))
    with patch.object(graph, "invoke_with_tools", agent), patch.object(graph, "invoke_structured_budgeted") as comp:
        out = run_query("q")
    comp.assert_not_called()
    assert out["found"] is False and out["answer"] == NOT_FOUND


def test_model_skipping_tools_still_searches(fake_retrieval):
    agent = Script(AIMessage(content="I think the answer is 42"))  # no tool call
    with patch.object(graph, "invoke_with_tools", agent), patch.object(
        graph, "invoke_structured_budgeted", composer_returns(ComposerOutput(found=True, answer="ok", citation_ids=[1]))
    ):
        out = run_query("what was decided?")
    assert out["tool_calls"] == ["search_sessions"]
    assert agent.calls == 1  # forced search goes straight to the composer


def test_agent_error_falls_back_to_plain_search(fake_retrieval):
    agent = Script(RuntimeError("503 overloaded"))
    with patch.object(graph, "invoke_with_tools", agent), patch.object(
        graph, "invoke_structured_budgeted", composer_returns(ComposerOutput(found=True, answer="ok", citation_ids=[1]))
    ):
        out = run_query("q")
    assert out["found"] is True and out["tool_calls"] == ["search_sessions"]


def test_rate_limit_propagates(fake_retrieval):
    with patch.object(graph, "invoke_with_tools", Script(LLMRateLimited("429"))):
        with pytest.raises(LLMRateLimited):
            run_query("q")


def test_round_limit(fake_retrieval, monkeypatch):
    monkeypatch.setattr(graph.settings, "max_tool_rounds", 2)
    agent = Script(*[call("search_sessions", {"question": f"variant {i}"}, i) for i in range(10)])
    with patch.object(graph, "invoke_with_tools", agent), patch.object(
        graph, "invoke_structured_budgeted", composer_returns(ComposerOutput(found=True, answer="ok", citation_ids=[1]))
    ):
        out = run_query("q")
    assert out["tool_calls"] == ["search_sessions", "search_sessions"]


def test_duplicate_call_goes_to_composer(fake_retrieval):
    same = {"question": "trip"}
    agent = Script(call("search_sessions", same, 0), call("search_sessions", {"question": "  TRIP "}, 1))
    with patch.object(graph, "invoke_with_tools", agent), patch.object(
        graph, "invoke_structured_budgeted", composer_returns(ComposerOutput(found=True, answer="ok", citation_ids=[1]))
    ):
        out = run_query("q")
    assert out["tool_calls"] == ["search_sessions"]


def test_date_tool_result_reaches_answer(fake_retrieval):
    agent = Script(
        call("search_sessions", {"question": "meeting"}),
        call("resolve_date_reference", {"reference_text": "next Friday", "message_timestamp": "2026-05-09 17:37"}, 1),
    )
    with patch.object(graph, "invoke_with_tools", agent), patch.object(
        graph, "invoke_structured_budgeted", composer_returns(ComposerOutput(found=True, answer="Fri 15 May", citation_ids=[1]))
    ):
        out = run_query("when is the meeting?")
    assert out["tool_calls"] == ["search_sessions", "resolve_date_reference"]
    assert out["date_resolutions"][0]["resolved_date"] == "2026-05-15"


def test_unknown_participant_message(fake_retrieval):
    from app.agent.tools import search_sessions

    text = search_sessions.invoke({"question": "x", "sender": "Zebediah"})
    assert "No participant matches 'Zebediah'" in text and "Priya" in text


def test_rate_limit_after_retrieval_still_answers(fake_retrieval):
    agent = Script(call("search_sessions", {"question": "trip"}), LLMRateLimited("429"))
    with patch.object(graph, "invoke_with_tools", agent), patch.object(
        graph, "invoke_structured_budgeted", composer_returns(ComposerOutput(found=True, answer="ok", citation_ids=[1]))
    ):
        out = run_query("q")
    assert out["found"] is True and out["tool_calls"] == ["search_sessions"]
