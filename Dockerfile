# Single-container image for Hugging Face Spaces (one public port: 7860).
# FastAPI runs privately on 127.0.0.1:8000; Streamlit is served on 7860.
# Local development still uses docker-compose.yml (two containers).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf-cache \
    BACKEND_URL=http://127.0.0.1:8000 \
    CHROMA_PERSIST_DIR=/tmp/chat-data/chroma \
    SQLITE_PATH=/tmp/chat-data/dedup.db

WORKDIR /workspace

# CPU-only torch first, so sentence-transformers doesn't pull the CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt ./requirements.txt
COPY frontend/requirements.txt ./frontend-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r frontend-requirements.txt

# Bake the embedding model into the image (no download at runtime).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
    && chmod -R a+rwX /opt/hf-cache

COPY app/ ./app/
COPY frontend/ ./frontend/
COPY docker/start-space.sh ./start-space.sh

# Spaces run containers as uid 1000.
RUN useradd -m -u 1000 user && chown -R user /workspace
USER user
ENV HOME=/home/user

EXPOSE 7860
CMD ["sh", "start-space.sh"]
