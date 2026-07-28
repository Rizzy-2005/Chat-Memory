"""
upload.py — handles WhatsApp .zip export uploads.
Stage 3: full pipeline — parse → dedup → chunk → embed → store.
"""
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.core.chunking import chunk_messages
from app.core.dedup import get_new_hashes, record_hashes
from app.core.parsing import parse_zip
from app.retrieval.vector_store import upsert_chunks, upsert_messages

router = APIRouter()


@router.post("/upload")
async def upload_chat(file: UploadFile = File(...)):
    """Accept a WhatsApp 'Export chat' .zip and run the full ingestion pipeline.

    Pipeline:
        1. Parse the .zip → list[ParsedMessage]
        2. Dedup check    → drop messages already in the index
        3. Chunk          → group new messages into SessionChunks
        4. Embed + store  → upsert into Chroma 'sessions' and 'messages' collections
        5. Record         → mark new hashes in the SQLite dedup index

    Returns:
        {
            "new_messages":       int,   # messages ingested this upload
            "new_sessions":       int,   # session chunks created this upload
            "skipped_duplicates": int,   # messages already seen, skipped
        }
    """
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="Only .zip files are accepted. Please upload a WhatsApp 'Export chat' zip.",
        )

    raw = await file.read()

    # ── Step 1: Parse ────────────────────────────────────────────────────────
    try:
        all_messages = parse_zip(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not all_messages:
        return {"new_messages": 0, "new_sessions": 0, "skipped_duplicates": 0}

    # ── Step 2: Dedup ────────────────────────────────────────────────────────
    all_hashes = [m.content_hash for m in all_messages]
    new_hash_list = get_new_hashes(all_hashes, settings.sqlite_path)
    new_hash_set = set(new_hash_list)

    new_messages = [m for m in all_messages if m.content_hash in new_hash_set]
    skipped = len(all_messages) - len(new_messages)

    if not new_messages:
        return {"new_messages": 0, "new_sessions": 0, "skipped_duplicates": skipped}

    # ── Step 3: Chunk ────────────────────────────────────────────────────────
    chunks = chunk_messages(new_messages, gap_hours=settings.session_gap_hours)

    # Build hash → chunk_id lookup for message metadata
    hash_to_chunk: dict[str, str] = {
        h: chunk.chunk_id
        for chunk in chunks
        for h in chunk.message_hashes
    }

    # ── Step 4: Embed + store ────────────────────────────────────────────────
    upsert_chunks(chunks)
    upsert_messages(new_messages, hash_to_chunk)

    # ── Step 5: Record in dedup index ────────────────────────────────────────
    record_hashes(
        [(h, hash_to_chunk[h]) for h in new_hash_list],
        settings.sqlite_path,
    )

    return {
        "new_messages": len(new_messages),
        "new_sessions": len(chunks),
        "skipped_duplicates": skipped,
    }
