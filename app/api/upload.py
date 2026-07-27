"""
upload.py — handles WhatsApp .zip export uploads.
Stage 2: accepts a .zip, extracts the .txt, parses messages, returns count.
"""
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.parsing import parse_zip

router = APIRouter()


@router.post("/upload")
async def upload_chat(file: UploadFile = File(...)):
    """Accept a WhatsApp 'Export chat' .zip file and parse it into messages.

    Returns the count of real (non-system) messages found.
    Stage 3 will wire deduplication + vector-store embedding here.
    """
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="Only .zip files are accepted. Please upload a WhatsApp 'Export chat' zip.",
        )

    data = await file.read()

    try:
        messages = parse_zip(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"parsed_messages": len(messages)}
