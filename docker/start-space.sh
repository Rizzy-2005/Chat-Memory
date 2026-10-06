#!/bin/sh
# Start the API in the background, then serve the UI on the public port.
uvicorn app.main:app --host 127.0.0.1 --port 8000 &
cd frontend
# XSRF/CORS off: Spaces embeds the app in an iframe, which otherwise breaks file uploads.
exec streamlit run app.py \
  --server.port="${PORT:-7860}" --server.address=0.0.0.0 --server.headless=true \
  --server.enableXsrfProtection=false --server.enableCORS=false
