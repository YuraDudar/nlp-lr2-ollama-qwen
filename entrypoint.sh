#!/bin/bash
set -e

MODEL_NAME="qwen2.5:0.5b"

echo "[entrypoint] Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

echo "[entrypoint] Waiting for Ollama to be ready..."
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "[entrypoint] Ollama is ready."
        break
    fi
    echo "[entrypoint] Attempt $i/30 — waiting 2s..."
    sleep 2
done

echo "[entrypoint] Pulling model: $MODEL_NAME ..."
ollama pull "$MODEL_NAME"
echo "[entrypoint] Model pulled successfully."

echo "[entrypoint] Starting FastAPI service on port 8000..."
exec uvicorn main:app --host 0.0.0.0 --port 8000
