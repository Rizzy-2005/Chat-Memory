"""
ui.py — presentation helpers for the Streamlit app: CSS, logo, cards, badges,
friendly dates.  Everything user- or model-provided is HTML-escaped here.
"""
from __future__ import annotations

import base64
import hashlib
import html
import os
from datetime import date, datetime

import streamlit as st

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
LOGO_PATH = os.path.join(ASSETS, "logo.svg")
USER_AVATAR_PATH = os.path.join(ASSETS, "user.svg")

_AVATAR_COLORS = ["#128C7E", "#3B6FB6", "#8E5BB5", "#C2502F", "#9A6A12", "#2F7A4D", "#B0306F", "#4A5568"]


def _data_uri(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/svg+xml;base64," + base64.b64encode(f.read()).decode()


LOGO_URI = _data_uri(LOGO_PATH)

CSS = """
<style>
/* ── Canvas: WhatsApp-style warm wallpaper with a faint dot texture ───────── */
[data-testid="stAppViewContainer"] > .stMain, .stMain {
  background-color: #EFEAE2;
  background-image: radial-gradient(rgba(18,140,126,0.08) 1px, transparent 1.2px);
  background-size: 22px 22px;
}
.block-container { max-width: 880px; padding-top: 1.6rem; padding-bottom: 6rem; }
[data-testid="stHeader"] { background: transparent; }

/* ── Hero ─────────────────────────────────────────────────────────────────── */
.cm-hero { position: relative; overflow: hidden; border-radius: 22px; padding: 22px 26px;
  background: linear-gradient(120deg, #075E54 0%, #0E7A6D 45%, #19A67E 100%);
  box-shadow: 0 10px 30px rgba(7,94,84,0.25); color: #FFFFFF; margin-bottom: 14px; }
.cm-hero::after { content: ""; position: absolute; right: -60px; top: -60px; width: 220px; height: 220px;
  border-radius: 50%; background: rgba(255,255,255,0.08); }
.cm-hero::before { content: ""; position: absolute; right: 70px; bottom: -90px; width: 180px; height: 180px;
  border-radius: 50%; background: rgba(37,211,102,0.18); }
.cm-hero-row { display: flex; align-items: center; gap: 16px; position: relative; z-index: 1; }
.cm-hero img { width: 54px; height: 54px; filter: drop-shadow(0 4px 10px rgba(0,0,0,0.18)); }
.cm-hero h1 { color: #FFFFFF !important; font-size: 1.75rem !important; font-weight: 800 !important;
  margin: 0 !important; padding: 0 !important; letter-spacing: -0.02em; }
.cm-hero p { margin: 2px 0 0 0; color: rgba(255,255,255,0.86); font-size: 0.95rem; }
.cm-hero-stats { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; position: relative; z-index: 1; }
.cm-hero-stat { background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.22); color: #FFFFFF;
  border-radius: 999px; padding: 4px 12px; font-size: 0.8rem; font-weight: 600; backdrop-filter: blur(4px); }

/* ── Navigation (segmented control) ───────────────────────────────────────── */
div[data-testid="stButtonGroup"] { margin: 2px 0 10px 0; }

/* ── Cards ────────────────────────────────────────────────────────────────── */
.cm-card { background: #FFFFFF; border: 1px solid #E5DED3; border-radius: 16px; padding: 18px 20px; margin: 8px 0 14px 0;
  box-shadow: 0 1px 2px rgba(17,27,33,0.04), 0 6px 18px rgba(17,60,55,0.06); }
.cm-card h3, .cm-card h4 { margin: 0 0 6px 0 !important; padding: 0 !important; }
.cm-muted { color: #54656F; font-size: 0.9rem; }
.cm-eyebrow { color: #128C7E; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }

/* ── Chat ─────────────────────────────────────────────────────────────────── */
.cm-user { display: flex; justify-content: flex-end; margin: 14px 0 6px 0; }
.cm-user-b { background: #D9FDD3; color: #111B21; border-radius: 16px 16px 4px 16px; padding: 10px 14px; max-width: 78%;
  box-shadow: 0 1px 1.5px rgba(11,20,26,0.13); font-size: 0.97rem; line-height: 1.45; overflow-wrap: anywhere; white-space: pre-wrap; }
[data-testid="stChatMessage"] { background: #FFFFFF; border: 1px solid #E5DED3; border-radius: 4px 16px 16px 16px;
  padding: 14px 16px; box-shadow: 0 1px 1.5px rgba(11,20,26,0.10); margin: 6px 0 4px 0; }
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarCustom"] img,
[data-testid="stChatMessage"] img[alt="assistant avatar"] { border-radius: 10px; }
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: transparent !important; }
[data-testid="stChatInput"] { border-radius: 26px !important; box-shadow: 0 6px 24px rgba(17,60,55,0.14); background: #FFFFFF; }

.cm-badges { display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 2px 0; }
.cm-badge { display: inline-flex; align-items: center; gap: 4px; font-size: 0.76rem; font-weight: 600; line-height: 1.4;
  padding: 3px 10px; border-radius: 999px; background: #E7F6F2; color: #0B6157; border: 1px solid #C6E9E1; }
.cm-badge.neutral { background: #F3F0EA; color: #3B4A54; border-color: #E2DCD1; }

.cm-cite { display: flex; gap: 10px; align-items: flex-start; background: #F9F7F3; border: 1px solid #ECE6DC;
  border-radius: 12px; padding: 10px 12px; margin: 8px 0; }
.cm-avatar { flex: 0 0 30px; width: 30px; height: 30px; border-radius: 50%; color: #FFFFFF; font-weight: 700; font-size: 0.85rem;
  display: inline-flex; align-items: center; justify-content: center; }
.cm-cite-head { font-size: 0.82rem; color: #54656F; }
.cm-cite-head b { color: #111B21; }
.cm-cite-text { font-size: 0.9rem; color: #111B21; margin-top: 3px; overflow-wrap: anywhere; }

.cm-callout { display: flex; align-items: center; flex-wrap: wrap; gap: 6px; background: #F2FBF8; border: 1px solid #CDEBE3;
  border-left: 4px solid #128C7E; border-radius: 10px; padding: 8px 12px; margin: 10px 0 4px 0; font-size: 0.87rem; color: #111B21; }
.cm-pill { display: inline-block; font-size: 0.7rem; font-weight: 700; padding: 2px 9px; border-radius: 999px; letter-spacing: 0.02em; }
.cm-pill.upcoming { background: #128C7E; color: #FFFFFF; }
.cm-pill.today { background: #F5B83D; color: #3D2A00; }
.cm-pill.past { background: #E3DFD8; color: #3B4A54; }

.cm-notfound { background: #FBF9F5; border: 1px dashed #CFC6B8; border-radius: 12px; padding: 12px 16px; color: #3B4A54; }
.cm-notfound b { color: #111B21; }
.cm-notfound ul { margin: 6px 0 0 18px; padding: 0; font-size: 0.88rem; }

/* ── Empty state ──────────────────────────────────────────────────────────── */
.cm-empty { text-align: center; padding: 26px 20px 10px 20px; }
.cm-empty img { width: 64px; height: 64px; margin-bottom: 8px; }
.cm-empty h3 { margin: 0 !important; padding: 0 !important; font-weight: 800 !important; }
.cm-empty p { color: #54656F; margin: 4px auto 0 auto; max-width: 520px; }
div[data-testid="stPills"] { display: flex; justify-content: center; }

/* ── Onboarding steps ─────────────────────────────────────────────────────── */
.cm-steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 12px; }
@media (max-width: 700px) { .cm-steps { grid-template-columns: 1fr; } }
.cm-step { background: #F9F7F3; border: 1px solid #ECE6DC; border-radius: 14px; padding: 14px; font-size: 0.9rem; }
.cm-step-n { width: 28px; height: 28px; border-radius: 50%; background: linear-gradient(135deg, #25D366, #128C7E); color: #FFFFFF;
  font-weight: 800; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 8px; }

/* ── Summary ──────────────────────────────────────────────────────────────── */
.cm-narrative { font-size: 0.98rem; line-height: 1.65; color: #111B21; }
.cm-timeline { position: relative; margin: 6px 0 0 8px; padding-left: 20px; border-left: 2px solid #CDEBE3; }
.cm-decision { position: relative; background: #FFFFFF; border: 1px solid #E5DED3; border-radius: 12px; padding: 10px 14px; margin: 10px 0;
  box-shadow: 0 1px 2px rgba(17,27,33,0.04); }
.cm-decision::before { content: ""; position: absolute; left: -28px; top: 14px; width: 12px; height: 12px; border-radius: 50%;
  background: #25D366; border: 3px solid #EFEAE2; }
.cm-decision b { color: #111B21; }

/* ── How it works ─────────────────────────────────────────────────────────── */
.cm-how { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
@media (max-width: 700px) { .cm-how { grid-template-columns: 1fr; } }
.cm-how .cm-card { margin: 0; }
.cm-how ol { margin: 6px 0 0 18px; padding: 0; font-size: 0.9rem; color: #3B4A54; }
.cm-how li { margin: 3px 0; }

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] .block-container, section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 0.6rem; }
.cm-brand { display: flex; align-items: center; gap: 10px; margin: 4px 0 4px 0; }
.cm-brand img { width: 38px; height: 38px; }
.cm-brand b { font-size: 1.15rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.01em; }
.cm-brand span { display: block; font-size: 0.78rem; color: rgba(227,241,238,0.7); }
.cm-side-h { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: rgba(227,241,238,0.6);
  margin: 18px 0 8px 0; }
.cm-tiles { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.cm-tile { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.09); border-radius: 12px; padding: 10px 12px; }
.cm-tile b { display: block; font-size: 1.35rem; font-weight: 800; color: #FFFFFF; line-height: 1.2; }
.cm-tile span { font-size: 0.74rem; color: rgba(227,241,238,0.7); }
.cm-tile.accent b { color: #25D366; }
.cm-span { font-size: 0.8rem; color: rgba(227,241,238,0.75); margin: 10px 0 6px 0; }
.cm-chip { display: inline-flex; align-items: center; gap: 6px; background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12);
  border-radius: 999px; padding: 3px 10px 3px 3px; margin: 3px 4px 3px 0; font-size: 0.8rem; color: #E3F1EE; }
.cm-chip .cm-avatar { flex-basis: 22px; width: 22px; height: 22px; font-size: 0.68rem; }
.cm-foot { font-size: 0.72rem; color: rgba(227,241,238,0.5); margin-top: 14px; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(text) -> str:
    return html.escape(str(text if text is not None else ""))


def md_safe(text: str) -> str:
    """Stop '$' in chat text from being rendered as LaTeX by st.markdown."""
    return (text or "").replace("$", "\\$")


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

def parse_dt(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace(" ", "T")[:19])
    except ValueError:
        return None


def friendly_dt(value: str) -> str:
    """'2026-07-02T19:14:00' → 'Thu 2 Jul 2026, 7:14 pm'."""
    d = parse_dt(value)
    if not d:
        return str(value)
    hour = d.hour % 12 or 12
    return f"{d:%a} {d.day} {d:%b %Y}, {hour}:{d:%M} {'am' if d.hour < 12 else 'pm'}"


def friendly_date(value: str | date | None, weekday: bool = False) -> str:
    """'2026-07-02' → '2 Jul 2026' (or 'Thu 2 Jul 2026')."""
    if value is None:
        return ""
    d = value if isinstance(value, date) else (parse_dt(value).date() if parse_dt(value) else None)
    if not d:
        return str(value)
    return (f"{d:%a} " if weekday else "") + f"{d.day} {d:%b %Y}"


def friendly_range(start: str | None, end: str | None) -> str:
    if start and end:
        a, b = parse_dt(start), parse_dt(end)
        if a and b and a.date() == b.date():
            return friendly_date(start)
        if a and b and (a.year, a.month) == (b.year, b.month):
            return f"{a.day}–{b.day} {b:%b %Y}"
        return f"{friendly_date(start)} – {friendly_date(end)}"
    if start:
        return f"from {friendly_date(start)}"
    if end:
        return f"until {friendly_date(end)}"
    return ""


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def hero(stats: dict | None) -> str:
    pills = ""
    if stats and stats.get("total_messages"):
        n_people = len(stats.get("participants", []))
        items = [
            f"{stats['total_messages']:,} messages",
            f"{stats.get('total_sessions', 0):,} conversations",
            f"{n_people} {'person' if n_people == 1 else 'people'}",
            friendly_range(stats.get("first_message"), stats.get("last_message")),
        ]
        pills = '<div class="cm-hero-stats">' + "".join(f'<span class="cm-hero-stat">{esc(i)}</span>' for i in items if i) + "</div>"
    return (
        f'<div class="cm-hero"><div class="cm-hero-row"><img src="{LOGO_URI}" alt="Chat Memory logo"/>'
        '<div><h1>Chat Memory</h1><p>Ask your WhatsApp history anything — every answer shows its sources.</p></div></div>'
        f"{pills}</div>"
    )


def brand() -> str:
    return (
        f'<div class="cm-brand"><img src="{LOGO_URI}" alt=""/>'
        "<div><b>Chat Memory</b><span>WhatsApp RAG assistant</span></div></div>"
    )


def side_heading(text: str) -> str:
    return f'<div class="cm-side-h">{esc(text)}</div>'


def stat_tiles(stats: dict) -> str:
    tiles = [
        ("Messages", f"{stats.get('total_messages', 0):,}", "accent"),
        ("Conversations", f"{stats.get('total_sessions', 0):,}", ""),
    ]
    html_tiles = "".join(f'<div class="cm-tile {cls}"><b>{esc(v)}</b><span>{esc(k)}</span></div>' for k, v, cls in tiles)
    span = f"{friendly_date(stats.get('first_message'))} → {friendly_date(stats.get('last_message'))}"
    return f'<div class="cm-tiles">{html_tiles}</div><div class="cm-span">{esc(span)}</div>'


def upload_tiles(res: dict) -> str:
    tiles = [("New", res["new_messages"], "accent"), ("Skipped", res["skipped_duplicates"], ""), ("Sessions", res["new_sessions"], "")]
    inner = "".join(f'<div class="cm-tile {c}"><b>{v:,}</b><span>{k}</span></div>' for k, v, c in tiles)
    return f'<div class="cm-tiles" style="grid-template-columns:1fr 1fr 1fr">{inner}</div>'


def avatar(name: str) -> str:
    color = _AVATAR_COLORS[int(hashlib.sha1(name.encode("utf-8")).hexdigest(), 16) % len(_AVATAR_COLORS)]
    initial = (name.strip()[:1] or "?").upper()
    return f'<span class="cm-avatar" style="background:{color}">{esc(initial)}</span>'


def participant_chips(names: list[str], limit: int = 12) -> str:
    chips = "".join(f'<span class="cm-chip">{avatar(n)}{esc(n)}</span>' for n in names[:limit])
    more = f'<span class="cm-span"> +{len(names) - limit} more</span>' if len(names) > limit else ""
    return f"<div>{chips}{more}</div>"


def user_bubble(text: str) -> str:
    return f'<div class="cm-user"><div class="cm-user-b">{esc(text)}</div></div>'


_MODE_LABELS = {"search_sessions": "Session search", "message_pinpoint": "Message pinpoint"}


def badges(mode_used: str, filters: dict | None, tool_calls: list[str] | None) -> str:
    items = []
    if mode_used in _MODE_LABELS:
        items.append(f'<span class="cm-badge">{_MODE_LABELS[mode_used]}</span>')
    f = filters or {}
    if f.get("sender"):
        items.append(f'<span class="cm-badge neutral">Person · {esc(f["sender"])}</span>')
    rng = friendly_range(f.get("start_date"), f.get("end_date"))
    if rng:
        items.append(f'<span class="cm-badge neutral">Dates · {esc(rng)}</span>')
    if tool_calls and "resolve_date_reference" in tool_calls:
        items.append('<span class="cm-badge neutral">Date tool</span>')
    return f'<div class="cm-badges">{"".join(items)}</div>' if items else ""


def citation_card(c: dict) -> str:
    text = esc(c.get("excerpt", "")).replace("\n", "<br>")
    return (
        f'<div class="cm-cite">{avatar(c.get("sender", "?"))}<div>'
        f'<div class="cm-cite-head"><b>{esc(c.get("sender"))}</b> · {esc(friendly_dt(c.get("timestamp", "")))}</div>'
        f'<div class="cm-cite-text">{text}</div></div></div>'
    )


def date_callout(r: dict) -> str:
    status = r.get("status", "past")
    label = {"upcoming": "UPCOMING", "today": "TODAY", "past": "PAST"}.get(status, status.upper())
    return (
        f'<div class="cm-callout">“{esc(r.get("reference_text"))}”, sent {esc(friendly_date(r.get("reference_timestamp")))}'
        f' → <b>{esc(friendly_date(r.get("resolved_date"), weekday=True))}</b>'
        f'<span class="cm-muted">· {esc(r.get("delta_description"))}</span>'
        f'<span class="cm-pill {esc(status)}">{label}</span></div>'
    )


NOT_FOUND_CARD = """
<div class="cm-notfound"><b>ⓘ I couldn't find anything in the chat about that.</b>
<ul><li>Name a person — “What did Priya say about the trip?”</li>
<li>Add a date range — “… between 1 and 15 July”</li></ul></div>
"""


def empty_state() -> str:
    return (
        f'<div class="cm-card cm-empty"><img src="{LOGO_URI}" alt=""/>'
        "<h3>Ask your chat anything</h3>"
        "<p>Answers quote the exact messages they come from. Name a person or a date to narrow the search.</p></div>"
    )


def decision_card(d: dict) -> str:
    return (
        f'<div class="cm-decision"><b>{esc(d.get("description"))}</b>'
        f'<div class="cm-muted">{esc(d.get("sender"))} · {esc(friendly_dt(d.get("timestamp", "")))}</div></div>'
    )


def onboarding_card(reset: bool = False) -> str:
    title = "Your data was reset — re-upload to continue" if reset else "Get started in three steps"
    steps = [
        "In WhatsApp, open the chat → ⋮ <b>More</b> → <b>Export chat</b> → <b>Without media</b>.",
        "Upload the <b>.zip</b> in the sidebar and press <b>Process chat</b>.",
        "Ask questions, or summarize any date range.",
    ]
    rows = "".join(f'<div class="cm-step"><span class="cm-step-n">{i}</span><div>{s}</div></div>' for i, s in enumerate(steps, 1))
    return f'<div class="cm-card"><div class="cm-eyebrow">Welcome</div><h3>{title}</h3><div class="cm-steps">{rows}</div></div>'
