"""
upload.py — handles WhatsApp .zip export uploads.
Stage 1 stub: returns 'not implemented' until Stage 2 wires in parsing + dedup.
"""
from fastapi import APIRouter

router = APIRouter()


@router.post("/upload")
async def upload_chat():
    # TODO (Stage 2): accept UploadFile, parse .zip → messages, dedup, embed, store.
    return {"status": "not implemented"}
