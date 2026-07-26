FROM python:3.11-slim

WORKDIR /workspace

# Install frontend dependencies
RUN pip install --no-cache-dir streamlit requests

# Copy frontend source
COPY frontend/ ./frontend/

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit; address 0.0.0.0 makes it reachable from outside the container
CMD ["streamlit", "run", "frontend/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
