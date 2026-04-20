#!/bin/bash
# test_ollama.sh — проверка работоспособности Ollama напрямую (внутри контейнера или через проброшенный порт)

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
MODEL="${MODEL:-qwen2.5:0.5b}"

echo "=== [1] Проверка доступности Ollama ==="
curl -s "${OLLAMA_URL}/api/tags" | python3 -m json.tool || {
    echo "ОШИБКА: Ollama недоступна по адресу ${OLLAMA_URL}"
    exit 1
}

echo ""
echo "=== [2] Тестовая генерация через Ollama API ==="
curl -s -X POST "${OLLAMA_URL}/api/generate" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"${MODEL}\",
        \"prompt\": \"Is this SMS spam? Reply with 1 for spam or 0 for ham. SMS: You won a FREE prize! Call 12345 now!\",
        \"stream\": false
    }" | python3 -m json.tool

echo ""
echo "=== Тест завершён ==="
