# Chat Memory

**Chat Memory** lets you upload a WhatsApp chat export once and then ask questions
about it whenever you like. Every answer cites the exact messages (date + sender)
it came from, and the app says *"I couldn't find anything in the chat about that"*
instead of guessing.

It's a RAG chatbot that:

- **chunks by conversation, not character count**: a 2-hour silence starts a new session (capped at ~1,000 characters)
- **deduplicates on re-upload**: a SHA-256 hash per message in SQLite, so a newer export only processes what's new
- **picks its own retrieval strategy** per question, using a LangGraph agent
- **cites every claim** and refuses honestly when nothing relevant is found
- **does date arithmetic deterministically**: "next Friday" in a message sent on 9 May is resolved by code, not by the LLM

**Stack:** FastAPI · LangGraph · LangChain · Chroma · sentence-transformers
(`all-MiniLM-L6-v2`, local) · Gemini free tier (with Groq as fallback) · SQLite ·
Streamlit · Docker. Everything is free; no paid API is needed.

---

## Quick start (Docker)

```bash
cp .env.example .env          # then paste your GEMINI_API_KEY (and optionally GROQ_API_KEY)
docker compose up --build     # the first build is slow: torch + the embedding model
```

- UI: <http://localhost:8501>
- API docs: <http://localhost:8000/docs>

Your data lives in the `chat_data` Docker volume, so it survives restarts.

### Getting your export

In WhatsApp, open the chat → ⋮ → **More** → **Export chat** → **Without media**,
then upload the `.zip` in the sidebar. Only the `.txt` inside is read; media is ignored.

## Running without Docker

```bash
pip install -r requirements.txt -r frontend/requirements.txt
uvicorn app.main:app --reload                 # backend on :8000
cd frontend && streamlit run app.py           # UI on :8501 (run from frontend/ so the theme loads)
```

Tests (offline, using fake embeddings and a mocked LLM):

```bash
pytest
```

---

## How it works

**Upload (runs once per export):**
`.zip` → parse lines into `{timestamp, sender, text}` (multi-line messages joined; system
lines and media placeholders dropped) → hash each message and skip the ones already
seen → group into sessions → embed locally → store in two Chroma collections
(`sessions` and `messages`) → record the hashes.

**Question (runs every time):**

```
START → agent ──tool_calls?──yes──▶ tools ──▶ agent   (max 3 rounds)
                 └──────no───────▶ composer ──▶ END
```

- **agent** is the LLM with three tools bound. Its reply either contains tool calls or it doesn't; the graph just checks that.
  - `search_sessions(question, sender?, start_date?, end_date?)`: semantic search over sessions. Sender and date filters can be combined in one call.
  - `message_pinpoint(question, sender?)`: finds one specific message, returned with its neighbouring messages for context.
  - `resolve_date_reference(reference_text, message_timestamp)`: deterministic; returns `upcoming` / `today` / `past` plus how many days away.
- **composer** is a second, structured LLM call that sees only the retrieved messages, numbered `[1]…[n]`.
  - It returns the numbers it relied on, and the code maps those back to real messages. The model therefore can't invent a citation.
  - Nothing relevant means the fixed not-found answer with no citations.

**Summaries** (`/summarize`) fetch every message in the date range with a metadata
filter (no semantic search) and return a narrative plus a separate, cited list of
decisions and plans. Large ranges are summarized in batches and then merged.

### Free-tier resilience

LLM calls go through a failover chain: `GEMINI_MODEL` → `GEMINI_FALLBACK_MODELS` →
`GROQ_MODEL`. When a model is overloaded, retired or rate-limited, the next one is
tried immediately. Each model gets a context budget sized to its tokens-per-minute
limit. If every model is out of quota, the UI says *"The free AI quota is used up for
now"*.

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/health` | | `{"status": "ok"}` |
| POST | `/upload` | multipart `file` (.zip) | `{new_messages, new_sessions, skipped_duplicates, total_in_file}` |
| GET | `/stats` | | `{total_messages, total_sessions, participants, first_message, last_message, llm}` |
| POST | `/query` | `{"question": "..."}` | `{answer, citations[{timestamp, sender, excerpt}], mode_used, filters, tool_calls, found, date_resolutions, trace}` |
| POST | `/summarize` | `{"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}` | `{narrative_summary, decisions_and_plans[{description, timestamp, sender, excerpt}], message_count, found}` |

## Configuration

All settings live in `app/config.py` and are documented in `.env.example`. The main ones:

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | | Primary LLM ([get one](https://aistudio.google.com/app/apikey)) |
| `GROQ_API_KEY` | | Optional fallback LLM ([get one](https://console.groq.com)) |
| `GEMINI_MODEL` / `GEMINI_FALLBACK_MODELS` / `GROQ_MODEL` | `gemini-3.5-flash` / `gemini-flash-lite-latest` / `openai/gpt-oss-120b` | Models in the failover chain |
| `SESSION_GAP_HOURS` | `2` | Silence that starts a new session |
| `MAX_CHUNK_CHARS` | `1000` | Session size cap before splitting |
| `DATE_ORDER` | `auto` | `DMY` / `MDY` if your phone's dates are misread |
| `MAX_TOOL_ROUNDS` | `3` | Agent loop limit |
| `BACKEND_URL` | `http://localhost:8000` | Where the frontend finds the API |

> If you upgrade from an older version, the backend notices the old storage format
> on startup and resets the index. Re-upload your chat afterwards.

## Deploying to Render (free)

1. Push to GitHub. `.env`, `*.zip` and `app/data/` are git-ignored.
2. In Render: **New → Blueprint**, then pick the repo. `render.yaml` defines both services.
3. Set `GEMINI_API_KEY` (and optionally `GROQ_API_KEY`) on **chat-memory-api**.
4. Set `BACKEND_URL` on **chat-memory-ui** to the API's public URL, e.g. `https://chat-memory-api.onrender.com`.

**Free-tier limits:**

- The backend sleeps after about 15 minutes idle. The UI shows *"Waking the server"* and retries automatically.
- There is no persistent disk, so uploads are lost when the backend restarts. The UI detects this and asks you to re-upload.
- `torch` plus the model is close to the 512 MB memory limit.

## Known limits

- Text only: images, voice notes and documents are ignored by design.
- One chat at a time. Two contacts with the same display name are treated as one sender.
- A conversation that continues across two exports is stored as two sessions, because only new messages are chunked.
- Relative dates are resolved from when a message was sent. "Next Friday" means the first Friday after that day.
