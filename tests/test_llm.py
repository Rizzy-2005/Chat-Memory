"""
tests/test_llm.py — provider chain and failover (app/agent/llm.py).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.agent import llm
from app.agent.llm import LLMNotConfigured, LLMRateLimited


@pytest.fixture
def keys(monkeypatch):
    s = llm.settings
    monkeypatch.setattr(s, "llm_provider", "auto")
    monkeypatch.setattr(s, "gemini_api_key", "g")
    monkeypatch.setattr(s, "groq_api_key", "q")
    monkeypatch.setattr(s, "gemini_model", "gem-main")
    monkeypatch.setattr(s, "gemini_fallback_models", "gem-lite")
    monkeypatch.setattr(s, "groq_model", "groq-model")
    monkeypatch.setattr(s, "max_context_chars", 0)
    return s


def test_chain_order_and_budgets(keys, monkeypatch):
    names = [c.name for c in llm.candidates()]
    assert names == ["gemini:gem-main", "gemini:gem-lite", "groq:groq-model"]
    assert llm.context_budget() == 30000
    monkeypatch.setattr(keys, "llm_provider", "groq")
    assert [c.name for c in llm.candidates()] == ["groq:groq-model"]
    assert llm.context_budget() == 12000


def test_no_keys(monkeypatch):
    monkeypatch.setattr(llm.settings, "gemini_api_key", "")
    monkeypatch.setattr(llm.settings, "groq_api_key", "")
    with pytest.raises(LLMNotConfigured):
        llm.candidates()
    assert llm.model_name() == "none"


def _chain(monkeypatch, behaviours):
    """Replace candidates with fakes whose structured call does `behaviour`."""
    cands = []
    for i, b in enumerate(behaviours):
        model = MagicMock()
        runnable = model.with_structured_output.return_value
        if isinstance(b, Exception):
            runnable.invoke.side_effect = b
        else:
            runnable.invoke.return_value = b
        cands.append(llm.Candidate(f"m{i}", 1000 * (i + 1), lambda m=model: m))
    monkeypatch.setattr(llm, "candidates", lambda: cands)


def test_failover_to_next_model_with_its_budget(monkeypatch):
    _chain(monkeypatch, [RuntimeError("503 overloaded"), "answer"])
    budgets = []
    out, payload = llm.invoke_structured_budgeted(object, lambda b: (budgets.append(b) or "p", b))
    assert out == "answer" and payload == 2000 and budgets == [1000, 2000]


def test_all_rate_limited(monkeypatch):
    _chain(monkeypatch, [RuntimeError("429 RESOURCE_EXHAUSTED"), RuntimeError("503")])
    with pytest.raises(LLMRateLimited):
        llm.invoke_structured(object, "p")


def test_all_misconfigured(monkeypatch):
    _chain(monkeypatch, [RuntimeError("404 model_not_found"), RuntimeError("401 invalid api key")])
    with pytest.raises(LLMNotConfigured):
        llm.invoke_structured(object, "p")


def test_none_output_fails_over(monkeypatch):
    _chain(monkeypatch, [None, "ok"])
    assert llm.invoke_structured(object, "p") == "ok"
