"""
summarize.py — date-range summary: narrative + separate decisions/plans list.

Fetches every stored message in the range with a metadata filter (no semantic
search), numbers them, and asks the model for a structured summary whose
decisions cite message numbers; numbers are mapped back to real messages.

If the range is larger than the context budget it is summarized in batches
(map) and the partial narratives are merged in one final call (reduce).
"""

from __future__ import annotations

from datetime import date, datetime, time

from pydantic import BaseModel, Field

from app.agent import context as ctx
from app.agent.llm import context_budget, invoke_structured, invoke_structured_budgeted
from app.retrieval import vector_store


class _Decision(BaseModel):
    description: str = Field(description="What was decided, planned, agreed or scheduled — one concise sentence.")
    message_id: int = Field(description="Number of the message where it was stated or agreed.")


class _SummaryOut(BaseModel):
    narrative_summary: str = Field(description="A readable narrative of what was discussed, in chronological order.")
    decisions_and_plans: list[_Decision] = Field(default_factory=list)


class _MergeOut(BaseModel):
    narrative_summary: str


_SUMMARY_PROMPT = """\
You summarize a WhatsApp chat for the period {start} to {end}{part}.
Use ONLY the numbered messages below. Never invent facts, names or dates.

1. narrative_summary: one to three short paragraphs describing what was discussed, in
   chronological order, mentioning who said what and the dates involved.
2. decisions_and_plans: every concrete decision, plan, agreement, deadline or scheduled event,
   each with the number of the message where it was stated. Use the resolved dates provided
   for phrases like "next Friday". Return an empty list if there are none.

Messages:
{messages}

{date_notes}"""

_MERGE_PROMPT = """\
Merge these partial summaries of consecutive parts of one WhatsApp chat ({start} to {end})
into one coherent narrative (one to three paragraphs, chronological). Use only what they say.

{parts}"""


def _batches(messages: list[dict], budget: int) -> list[list[dict]]:
    out: list[list[dict]] = []
    rest = messages
    while rest:
        batch = ctx.fit_budget(rest, budget)
        out.append(batch)
        rest = rest[len(batch):]
    return out


def summarize_range(start: date, end: date) -> dict:
    messages = vector_store.messages_in_range(
        datetime.combine(start, time(0, 0)), datetime.combine(end, time(23, 59, 59))
    )
    if not messages:
        return {
            "narrative_summary": f"No messages were found between {start.isoformat()} and {end.isoformat()}.",
            "decisions_and_plans": [],
            "message_count": 0,
            "found": False,
        }

    batches = _batches(messages, context_budget())
    narratives: list[str] = []
    decisions: list[dict] = []
    for n, batch in enumerate(batches, 1):
        part = f" (part {n} of {len(batches)})" if len(batches) > 1 else ""

        def build(budget: int, batch=batch, part=part):
            # A fallback model with a smaller budget sees a trimmed batch; the
            # exact list it numbered comes back with the answer.
            msgs = ctx.fit_budget(batch, budget)
            prompt = _SUMMARY_PROMPT.format(
                start=start.isoformat(),
                end=end.isoformat(),
                part=part,
                messages=ctx.numbered_lines(msgs),
                date_notes=ctx.date_notes_text(ctx.auto_date_resolutions(msgs, limit=25)),
            )
            return prompt, msgs

        out, batch = invoke_structured_budgeted(_SummaryOut, build)
        narratives.append(out.narrative_summary.strip())
        for d in out.decisions_and_plans:
            ids = ctx.valid_ids([d.message_id], len(batch))
            if not ids:
                continue  # uncited decision → dropped rather than trusted
            m = batch[ids[0] - 1]
            decisions.append({"description": d.description.strip(), "timestamp": m["timestamp"], "sender": m["sender"], "excerpt": ctx.citation(m)["excerpt"]})

    if len(narratives) == 1:
        narrative = narratives[0]
    else:
        merged: _MergeOut = invoke_structured(
            _MergeOut,
            _MERGE_PROMPT.format(
                start=start.isoformat(),
                end=end.isoformat(),
                parts="\n\n".join(f"Part {i}:\n{t}" for i, t in enumerate(narratives, 1)),
            ),
        )
        narrative = merged.narrative_summary.strip()

    decisions.sort(key=lambda d: d["timestamp"])
    return {
        "narrative_summary": narrative,
        "decisions_and_plans": decisions,
        "message_count": len(messages),
        "found": True,
    }
