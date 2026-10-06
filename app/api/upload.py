"""
upload.py — POST /upload: parse → dedup → chunk → embed → store → record.
"""
import threading

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import settings
from app.core.chunking import chunk_messages
from app.core.dedup import get_new_hashes, record_hashes
from app.core.parsing import parse_zip
from app.retrieval.vector_store import upsert_chunks, upsert_messages

router = APIRouter()

# One ingestion at a time: two concurrent uploads of overlapping exports
# would otherwise both see the same messages as "new".
_ingest_lock = threading.Lock()


@router.post("/upload")
def upload_chat(file: UploadFile = File(...)):
    """Accept a WhatsApp 'Export chat' .zip and run the ingestion pipeline.

    Returns {new_messages, new_sessions, skipped_duplicates, total_in_file}.
    """
    if not (file.filename or "").lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are accepted. Please upload a WhatsApp 'Export chat' zip.")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    raw = file.file.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File is larger than {settings.max_upload_mb} MB.")

    # ── 1. Parse (only the .txt; media and other files are ignored) ─────────
    try:
        all_messages = parse_zip(raw, date_order=settings.date_order)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    with _ingest_lock:
        # ── 2. Dedup against everything ever stored ──────────────────────────
        new_hashes = set(get_new_hashes([m.content_hash for m in all_messages]))
        new_messages = [m for m in all_messages if m.content_hash in new_hashes]
        skipped = len(all_messages) - len(new_messages)
        if not new_messages:
            return {"new_messages": 0, "new_sessions": 0, "skipped_duplicates": skipped, "total_in_file": len(all_messages)}

        # ── 3. Chunk the new messages into sessions (size-capped) ────────────
        chunks = chunk_messages(new_messages, gap_hours=settings.session_gap_hours, max_chars=settings.max_chunk_chars)
        hash_to_chunk = {h: c.chunk_id for c in chunks for h in c.message_hashes}

        # ── 4. Embed + store.  Upserts are idempotent and hashes are recorded
        #       only after both succeed, so a failed upload is safe to retry.
        upsert_chunks(chunks)
        upsert_messages(new_messages, hash_to_chunk)

        # ── 5. Record in the dedup index ─────────────────────────────────────
        record_hashes([
            (m.content_hash, hash_to_chunk[m.content_hash], m.timestamp.isoformat(), m.sender)
            for m in new_messages
        ])

    return {
        "new_messages": len(new_messages),
        "new_sessions": len(chunks),
        "skipped_duplicates": skipped,
        "total_in_file": len(all_messages),
    }
