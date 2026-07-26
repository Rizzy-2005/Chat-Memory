"""
main.py — FastAPI application entry point.
Creates the app instance and registers all four API routers.
"""
from fastapi import FastAPI

from app.api import health, upload, query, summarize

app = FastAPI(
    title="Chat Memory",
    description="RAG chatbot over your WhatsApp export.",
    version="0.1.0",
)

# Register routers — each module owns its own prefix and tags.
app.include_router(health.router, tags=["Health"])
app.include_router(upload.router, tags=["Upload"])
app.include_router(query.router, tags=["Query"])
app.include_router(summarize.router, tags=["Summarize"])
