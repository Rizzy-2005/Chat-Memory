# Chat Memory

**Chat Memory** is a RAG (Retrieval-Augmented Generation) chatbot that lets you ask natural-language questions about your own WhatsApp conversations. Upload a WhatsApp "Export chat" `.zip` file, and the app parses it into structured messages, embeds them locally with `sentence-transformers`, and stores them in an on-disk Chroma vector database — deduplicating on every re-upload so nothing is processed twice. At query time, a LangGraph agent routes each question to the right retrieval mode, calls a dedicated MCP server for any relative date resolution, and returns grounded answers with per-claim citations (timestamp + sender) — never guessing when information isn't in the chat.

**Stack:** FastAPI · LangGraph · LangChain · Chroma · sentence-transformers (all-MiniLM-L6-v2) · Google Gemini free tier · SQLite dedup · MCP (stdio) · Streamlit frontend · Docker.

## How to run locally

```bash
# 1. Copy the example env file and fill in your Gemini API key
cp .env.example .env

# 2. Build and start both containers (app + mcp-server)
docker compose up --build

# 3. Verify the backend is healthy
curl localhost:8000/health
# → {"status":"ok"}

# 4. Open the Streamlit frontend (when implemented)
#    docker compose up frontend  (added in a later stage)
```

> **Note:** All vector data and the dedup SQLite database are stored in the `chat_data` named Docker volume, so they survive container restarts.
