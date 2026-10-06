FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the ONNX embedding model (all-MiniLM-L6-v2) into the image:
# no download at runtime, faster cold starts.
# Retried because the model download (~80 MB) occasionally drops on build hosts.
RUN for i in 1 2 3; do       python -c "from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2 as E; E()(['warm up'])" && break;       echo "model download failed, retrying ($i)"; sleep 5;     done

COPY app/ ./app/

EXPOSE 8000

# Render (and other hosts) inject $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
