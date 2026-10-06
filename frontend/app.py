"""
app.py — Streamlit frontend for Chat Memory.

Talks to the FastAPI backend only through its HTTP endpoints:
    GET /health · GET /stats · POST /upload · POST /query · POST /summarize
"""
from __future__ import annotations

import os
import sys
import time
from datetime import date, timedelta

import requests
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ui  # noqa: E402  (sibling module)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").strip().rstrip("/")
if not BACKEND_URL.startswith(("http://", "https://")):
    BACKEND_URL = "https://" + BACKEND_URL  # e.g. "chat-memory-api.onrender.com"

st.set_page_config(page_title="Chat Memory", page_icon=ui.LOGO_PATH, layout="centered", initial_sidebar_state="expanded")
ui.inject_css()

# ---------------------------------------------------------------------------
# Backend access
# ---------------------------------------------------------------------------

class BackendError(Exception):
    pass


def api(method: str, path: str, timeout: float = 30, **kwargs) -> dict:
    """Call the backend and turn every failure into a friendly BackendError."""
    try:
        resp = requests.request(method, f"{BACKEND_URL}{path}", timeout=timeout, **kwargs)
    except requests.exceptions.Timeout as exc:
        raise BackendError("The server took too long to answer. Please try again.") from exc
    except requests.exceptions.ConnectionError as exc:
        raise BackendError("Can't reach the server. It may be waking up — try again in a minute.") from exc
    if resp.ok:
        return resp.json()
    try:
        detail = resp.json().get("detail", resp.text)
    except ValueError:
        detail = resp.text
    if isinstance(detail, list):  # FastAPI validation errors
        detail = "; ".join(str(d.get("msg", d)) for d in detail)
    if resp.status_code == 404 and path == "/stats":
        raise BackendError(
            f"The server at {BACKEND_URL} answered, but it isn't a Chat Memory backend. "
            "Set BACKEND_URL to your own chat-memory-api URL (it may have a suffix, "
            "e.g. https://chat-memory-api-xxxx.onrender.com)."
        )
    if resp.status_code == 429:
        raise BackendError("The free AI quota is used up for now. Try again in a minute.")
    if resp.status_code == 503:
        raise BackendError(f"The AI model isn't available: {detail}")
    if resp.status_code >= 500:
        raise BackendError(f"Something went wrong on the server. {detail}")
    raise BackendError(str(detail))


def backend_alive() -> bool:
    try:
        return requests.get(f"{BACKEND_URL}/health", timeout=4).ok
    except requests.exceptions.RequestException:
        return False


@st.cache_data(ttl=30, show_spinner=False)
def fetch_stats() -> dict:
    return api("GET", "/stats", timeout=15)


def ensure_backend() -> None:
    """Free hosts sleep; wait up to ~75 s for /health before giving up."""
    if st.session_state.get("backend_ok_at", 0) > time.time() - 60:
        return
    if not backend_alive():
        with st.status("Waking the server, this takes up to a minute…", expanded=False) as status:
            deadline = time.time() + 75
            while time.time() < deadline and not backend_alive():
                time.sleep(3)
            if not backend_alive():
                status.update(label="The server is not responding.", state="error")
                st.error(f"Couldn't reach the backend at {BACKEND_URL}. Start it with `uvicorn app.main:app` or try again shortly.")
                if st.button("Retry"):
                    st.rerun()
                st.stop()
            status.update(label="Server is awake.", state="complete")
    st.session_state.backend_ok_at = time.time()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

st.session_state.setdefault("messages", [])
st.session_state.setdefault("upload_result", None)
st.session_state.setdefault("summary", None)
st.session_state.setdefault("seen_data", False)
st.session_state.setdefault("pill_key", 0)

ensure_backend()
try:
    stats = fetch_stats()
except BackendError as exc:
    st.error(str(exc))
    st.stop()

has_data = stats.get("total_messages", 0) > 0
data_was_reset = not has_data and st.session_state.seen_data
if has_data:
    st.session_state.seen_data = True

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(ui.brand(), unsafe_allow_html=True)

    st.markdown(ui.side_heading("Upload chat"), unsafe_allow_html=True)
    uploaded = st.file_uploader("WhatsApp export (.zip)", type=["zip"], label_visibility="collapsed")
    st.caption("Export → *Without media*. Text only; photos and voice notes are ignored.")
    if uploaded is not None and st.button("Process chat", type="primary", use_container_width=True, icon=":material/bolt:"):
        with st.status("Processing your chat…", expanded=True) as status:
            st.write(f"Reading **{ui.esc(uploaded.name)}** ({uploaded.size / 1024:,.0f} KB)")
            st.write("Parsing, de-duplicating and embedding — the first run can take a minute.")
            try:
                result = api(
                    "POST", "/upload", timeout=900,
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/zip")},
                )
            except BackendError as exc:
                status.update(label="Upload failed", state="error")
                st.error(str(exc))
            else:
                st.session_state.upload_result = result
                fetch_stats.clear()
                if result["new_messages"] == 0:
                    status.update(label="Already up to date", state="complete", expanded=False)
                    st.toast("Already up to date — nothing new in this export.", icon="✅")
                else:
                    status.update(label="Chat processed", state="complete", expanded=False)
                    st.toast(f"Added {result['new_messages']:,} new messages.", icon="✅")
                time.sleep(0.6)
                st.rerun()

    res = st.session_state.upload_result
    if res:
        st.markdown(ui.upload_tiles(res), unsafe_allow_html=True)
        if res["new_messages"] == 0 and res["skipped_duplicates"]:
            st.caption("✅ Already up to date.")

    st.markdown(ui.side_heading("Your chat"), unsafe_allow_html=True)
    if has_data:
        st.markdown(ui.stat_tiles(stats), unsafe_allow_html=True)
        st.markdown(ui.participant_chips(stats.get("participants", [])), unsafe_allow_html=True)
    else:
        st.caption("Nothing uploaded yet.")

    st.markdown(ui.side_heading("Conversation"), unsafe_allow_html=True)
    if st.button("Clear conversation", use_container_width=True, icon=":material/delete_sweep:",
                 help="Empties the on-screen chat only; stored data is kept."):
        st.session_state.messages = []
        st.rerun()
    st.markdown(f'<div class="cm-foot">Model · {ui.esc(stats.get("llm", "n/a"))}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header + navigation
# ---------------------------------------------------------------------------

st.markdown(ui.hero(stats), unsafe_allow_html=True)

VIEWS = {"chat": ":material/forum: Chat", "summary": ":material/event_note: Summary", "how": ":material/auto_awesome: How it works"}
view = st.segmented_control(
    "View", list(VIEWS), format_func=VIEWS.get, default="chat", key="view", label_visibility="collapsed",
) or "chat"


def render_answer(d: dict) -> None:
    """Assistant bubble body: text, routing badges, date callouts, sources."""
    if not d.get("found"):
        st.markdown(ui.NOT_FOUND_CARD, unsafe_allow_html=True)
        st.markdown(ui.badges(d.get("mode_used", ""), d.get("filters"), d.get("tool_calls")), unsafe_allow_html=True)
        return
    st.markdown(ui.md_safe(d["answer"]))
    for r in d.get("date_resolutions", []):
        st.markdown(ui.date_callout(r), unsafe_allow_html=True)
    st.markdown(ui.badges(d.get("mode_used", ""), d.get("filters"), d.get("tool_calls")), unsafe_allow_html=True)
    cites = d.get("citations", [])
    if cites:
        with st.expander(f"Sources ({len(cites)})", icon=":material/format_quote:"):
            st.markdown("".join(ui.citation_card(c) for c in cites), unsafe_allow_html=True)


def describe_step(t: dict) -> str:
    name = {"search_sessions": "Searched conversation sessions", "message_pinpoint": "Searched individual messages",
            "resolve_date_reference": "Resolved a relative date"}.get(t["tool"], t["tool"])
    bits = []
    f = t.get("filters") or {}
    if f.get("sender"):
        bits.append(f["sender"])
    rng = ui.friendly_range(f.get("start_date"), f.get("end_date"))
    if rng:
        bits.append(rng)
    if "results" in t:
        bits.append(f"{t['results']} messages")
    if t["tool"] == "resolve_date_reference" and t.get("result", {}).get("resolved_date"):
        r = t["result"]
        bits.append(f"“{r.get('reference_text')}” → {ui.friendly_date(r['resolved_date'], weekday=True)}")
    return name + (f" — {', '.join(bits)}" if bits else "")


def suggestions() -> list[str]:
    people = stats.get("participants", [])
    first_name = people[0].split()[0] if people else "someone"
    last = ui.parse_dt(stats.get("last_message") or "")
    month = f"{last:%B %Y}" if last else "the last month"
    return [
        f"What did {first_name} announce most recently?",
        "Find the message with a registration link",
        f"What was decided in {month}?",
        "What deadlines were mentioned?",
    ]


# ---------------------------------------------------------------------------
# Chat view
# ---------------------------------------------------------------------------

if view == "chat":
    # Top-level chat_input is pinned to the bottom of the page, wherever it's called.
    typed = st.chat_input("Ask about your chat…", disabled=not has_data)
    question = typed or st.session_state.pop("pending_question", None)

    if not has_data:
        st.markdown(ui.onboarding_card(reset=data_was_reset), unsafe_allow_html=True)
    elif not st.session_state.messages and not question:
        st.markdown(ui.empty_state(), unsafe_allow_html=True)
        pick = st.pills("Try asking", suggestions(), key=f"pills_{st.session_state.pill_key}", label_visibility="collapsed")
        if pick:
            st.session_state.pending_question = pick
            st.session_state.pill_key += 1
            st.rerun()

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(ui.user_bubble(msg["content"]), unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar=ui.LOGO_PATH):
                if msg.get("error"):
                    st.error(msg["error"])
                else:
                    render_answer(msg["data"])

    if question and has_data:
        st.session_state.messages.append({"role": "user", "content": question})
        st.markdown(ui.user_bubble(question), unsafe_allow_html=True)
        with st.chat_message("assistant", avatar=ui.LOGO_PATH):
            started = time.time()
            error = None
            with st.status("Choosing a search and reading your chat…", expanded=False) as status:
                try:
                    data = api("POST", "/query", json={"question": question}, timeout=180)
                except BackendError as exc:
                    status.update(label="Couldn't answer", state="error")
                    data, error = None, str(exc)
                else:
                    for t in data.get("trace", []):
                        st.write("• " + describe_step(t))
                    st.write("• Wrote the answer from the retrieved messages")
                    status.update(label=f"Answered in {time.time() - started:.1f} s", state="complete")
            if data is None:
                st.error(error)
                st.session_state.messages.append({"role": "assistant", "error": error})
            else:
                render_answer(data)
                st.session_state.messages.append({"role": "assistant", "data": data})

# ---------------------------------------------------------------------------
# Summary view
# ---------------------------------------------------------------------------

elif view == "summary":
    if not has_data:
        st.markdown(ui.onboarding_card(reset=data_was_reset), unsafe_allow_html=True)
    else:
        first = date.fromisoformat(stats["first_message"])
        last = date.fromisoformat(stats["last_message"])

        def set_range(days: int | None) -> None:
            st.session_state.sum_start = first if days is None else max(first, last - timedelta(days=days - 1))
            st.session_state.sum_end = last

        if "sum_start" not in st.session_state or not (first <= st.session_state.sum_start <= last):
            set_range(30)
        if not (first <= st.session_state.get("sum_end", last) <= last):
            st.session_state.sum_end = last

        with st.container(border=True):
            st.markdown('<div class="cm-eyebrow">Date-range summary</div>', unsafe_allow_html=True)
            st.markdown("#### What happened, and what was decided?")
            p1, p2, p3 = st.columns(3)
            p1.button("Last 7 days", on_click=set_range, args=(7,), use_container_width=True)
            p2.button("Last 30 days", on_click=set_range, args=(30,), use_container_width=True)
            p3.button("Whole chat", on_click=set_range, args=(None,), use_container_width=True)
            d1, d2 = st.columns(2)
            start = d1.date_input("From", min_value=first, max_value=last, key="sum_start", format="DD/MM/YYYY")
            end = d2.date_input("To", min_value=first, max_value=last, key="sum_end", format="DD/MM/YYYY")
            st.caption("“Last” is measured from the newest message in the chat.")
            go = st.button("Summarize", type="primary", use_container_width=True, icon=":material/auto_awesome:")

        if go:
            if start > end:
                st.error("“From” must be on or before “To”.")
            else:
                with st.spinner(f"Reading messages from {ui.friendly_date(start)} to {ui.friendly_date(end)}…"):
                    try:
                        st.session_state.summary = api(
                            "POST", "/summarize", timeout=300,
                            json={"start_date": start.isoformat(), "end_date": end.isoformat()},
                        ) | {"range": (start.isoformat(), end.isoformat())}
                    except BackendError as exc:
                        st.session_state.summary = None
                        st.error(str(exc))

        s = st.session_state.summary
        if s:
            rng = ui.friendly_range(*s["range"])
            count = f" · {s.get('message_count', 0):,} messages" if s.get("found", True) else ""
            st.markdown(
                f'<div class="cm-card"><div class="cm-eyebrow">Narrative · {ui.esc(rng)}{count}</div>'
                f'<div class="cm-narrative">{ui.esc(s["narrative_summary"]).replace(chr(10), "<br>")}</div></div>',
                unsafe_allow_html=True,
            )
            items = s.get("decisions_and_plans", [])
            body = (
                '<div class="cm-timeline">' + "".join(ui.decision_card(d) for d in items) + "</div>"
                if items else '<div class="cm-muted">No clear decisions in this range.</div>'
            )
            st.markdown(f'<div class="cm-card"><div class="cm-eyebrow">Decisions and plans · {len(items)}</div>{body}</div>',
                        unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# How it works view
# ---------------------------------------------------------------------------

else:
    st.markdown(
        """
<div class="cm-how">
  <div class="cm-card"><div class="cm-eyebrow">Once per upload</div><h4>Remember</h4><ol>
    <li>Read the <b>.txt</b> inside the zip; media is ignored.</li>
    <li>Parse each line into time, sender and text; drop system lines.</li>
    <li>Hash every message and skip ones already stored.</li>
    <li>Group into conversations (2 h of silence starts a new one).</li>
    <li>Embed locally with MiniLM and store in Chroma.</li></ol></div>
  <div class="cm-card"><div class="cm-eyebrow">Every question</div><h4>Recall</h4><ol>
    <li>An agent picks a tool: <b>session search</b> (with person / date filters) or <b>message pinpoint</b>.</li>
    <li>Relative dates like “next Friday” are resolved by code, not guessed.</li>
    <li>A second model call answers <b>only</b> from the retrieved messages.</li>
    <li>Every claim cites a real message — or it says it couldn't find anything.</li></ol></div>
</div>
""",
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.graphviz_chart(
            """
digraph {
  rankdir=LR; bgcolor="transparent"; pad=0.2;
  node [shape=box, style="rounded,filled", fillcolor="#FFFFFF", color="#CDEBE3", fontname="Helvetica", fontsize=11, fontcolor="#111B21"];
  edge [color="#8AA6A0", fontname="Helvetica", fontsize=9, fontcolor="#54656F"];
  subgraph cluster_up { label="Upload"; color="#E5DED3"; fontname="Helvetica"; fontsize=10; fontcolor="#128C7E";
    zip [label=".zip"]; parse [label="parse"]; dedup [label="dedup\\n(SQLite)"]; chunk [label="sessions"]; embed [label="embed\\n(MiniLM)"];
    chroma [label="Chroma", fillcolor="#E7F6F2"];
    zip -> parse -> dedup -> chunk -> embed -> chroma; }
  subgraph cluster_q { label="Question"; color="#E5DED3"; fontname="Helvetica"; fontsize=10; fontcolor="#128C7E";
    q [label="question"]; agent [label="agent\\n(picks a tool)", fillcolor="#E7F6F2"]; tools [label="tools\\nsearch · pinpoint · date"];
    composer [label="composer\\n(cited answer)", fillcolor="#E7F6F2"];
    q -> agent; agent -> tools [label="tool call"]; tools -> agent [label="results"]; agent -> composer [label="done"]; }
  chroma -> tools [style=dashed, label="vector + metadata search"];
}
"""
        )
    st.caption("FastAPI · LangGraph · LangChain · Chroma · MiniLM (ONNX) · Gemini (Groq fallback) · SQLite · Streamlit · Docker")
