FROM python:3.11-slim

WORKDIR /workspace

# Pre-install CPU-only torch to avoid pulling the 526 MB GPU wheel from PyPI.
# sentence-transformers depends on torch; if torch is already present pip won't
# re-download it when processing requirements.txt.
RUN pip install --no-cache-dir \
    torch --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (layer-cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/

# Expose FastAPI port
EXPOSE 8000

# Run uvicorn; host 0.0.0.0 makes it reachable from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
