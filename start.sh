#!/bin/bash
# Runs both halves of the app in a single container:
#  - FastAPI backend on 127.0.0.1:8000 (internal only)
#  - Streamlit frontend on 0.0.0.0:$PORT (exposed by the host platform)
set -e

PORT="${PORT:-7860}"

echo "Starting FastAPI backend on 127.0.0.1:8000 ..."
uvicorn api_backend:app --host 127.0.0.1 --port 8000 &
BACKEND_PID=$!

sleep 3

echo "Starting Streamlit frontend on 0.0.0.0:${PORT} ..."
streamlit run streamlit_app.py \
    --server.port="${PORT}" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false

kill "$BACKEND_PID" 2>/dev/null || true
