"""
main.py — FastAPI application entry point.
Creates the app, registers the routers and prepares storage on startup.
"""
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import health, query, stats, summarize, upload
from app.retrieval.vector_store import ensure_data_version, get_embeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_data_version()
    # Load the embedding model in the background so the first upload/question
    # is fast, without delaying /health (Render needs the port open quickly).
    threading.Thread(target=get_embeddings, daemon=True).start()
    yield


app = FastAPI(
    title="Chat Memory",
    description="RAG chatbot over your WhatsApp export, with citations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router, tags=["Health"])
app.include_router(upload.router, tags=["Upload"])
app.include_router(stats.router, tags=["Stats"])
app.include_router(query.router, tags=["Query"])
app.include_router(summarize.router, tags=["Summarize"])
