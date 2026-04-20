# SMS Spam Detection with LLM — Лабораторная работа 2 (NLP)

| Студент             | Группа     | Оценка |
| ------------------- | ---------- | ------ |
| Дударь Юрий Мохсенович | М8О-409Б-22 |        |

Proof-of-concept прототип классификации SMS-сообщений на спам/не-спам
с помощью LLM **Qwen2.5:0.5B** через сервер **Ollama**, развёрнутый в Docker-контейнере.
Дополнительная часть — сравнительное исследование четырёх техник промптинга.

---

## Стек

| Компонент | Версия |
| --------- | ------ |
| Ubuntu    | 22.04  |
| Python    | 3.12   |
| Ollama    | latest |
| Qwen2.5   | 0.5B   |
| FastAPI   | 0.115  |
| Docker    | 24+    |

---

## Структура проекта

```text
nlp-lr2-ollama-qwen/
├── Dockerfile                   # Ubuntu 22.04 + Python 3.12 + Ollama + FastAPI
├── entrypoint.sh                # Запуск: Ollama → pull модели → FastAPI
├── docker-compose.yml           # Проброс портов 8000 и 11434
├── app/
│   ├── main.py                  # FastAPI-сервис (единственный эндпоинт /generate)
│   └── requirements.txt         # fastapi, uvicorn, httpx, pydantic
├── research/
│   ├── prompts.py               # Системные промпты: zero-shot, CoT, few-shot, CoT+few-shot
│   ├── run_evaluation.py        # Инференс + метрики на датасете
│   └── data/                    # ← сюда положить spam.csv (Kaggle)
├── scripts/
│   ├── test_ollama.sh           # Тест Ollama напрямую (curl)
│   └── test_service.py          # Тест FastAPI (requests)
└── requirements.txt             # pandas, scikit-learn, requests (для research)
```

---

## Часть 1 — Инженерная: LLM-сервис в Docker

### Шаг 1 — Сборка и запуск контейнера

```bash
# Клонировать репозиторий
git clone <repo-url>
cd nlp-lr2-ollama-qwen

# Собрать образ и запустить (первый запуск скачает Qwen2.5:0.5B ~350MB)
docker compose up --build
```

> Первый запуск занимает 3–5 минут: сборка образа + скачивание модели.
> Последующие запуски используют кешированный volume `ollama_models`.

При успешном старте в логах вы увидите:

```text
[entrypoint] Ollama is ready.
[entrypoint] Model pulled successfully.
[entrypoint] Starting FastAPI service on port 8000...
INFO:     Application startup complete.
```

### Шаг 2 — Тест Ollama напрямую (порт 11434)

```bash
# Проверить доступность сервера
curl http://localhost:11434/api/tags

# Тестовая генерация через Ollama API
bash scripts/test_ollama.sh
```

Ожидаемый ответ:

```json
{
  "model": "qwen2.5:0.5b",
  "response": "{\"reasoning\": \"Contains prize offer...\", \"verdict\": 1}",
  "done": true
}
```

### Шаг 3 — Тест FastAPI-сервиса (порт 8000)

```bash
# Установить зависимости (вне контейнера)
pip install requests

# Запустить тест
python scripts/test_service.py
```

Либо curl:

```bash
# Проверка health
curl http://localhost:8000/health

# Генерация через FastAPI
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Is this SMS spam? Reply with 1 or 0. SMS: You won a FREE prize!", "system": ""}'
```

Ожидаемый ответ:

```json
{
  "response": "{\"reasoning\": \"Contains 'FREE', 'won', prize claim\", \"verdict\": 1}",
  "model": "qwen2.5:0.5b",
  "done": true
}
```

### API Reference

| Метод | Путь        | Описание                         |
| ----- | ----------- | -------------------------------- |
| GET   | `/health`   | Проверка работоспособности       |
| GET   | `/models`   | Список моделей в Ollama          |
| POST  | `/generate` | Отправка запроса к LLM           |

**POST /generate — тело запроса:**

```json
{
  "prompt": "текст запроса (обязательно)",
  "system": "системный промпт (опционально, по умолчанию пустой)",
  "model": "qwen2.5:0.5b",
  "stream": false
}
```

---

## Часть 2 — Исследовательская: Техники промптинга

### Описание техник

| # | Техника          | Описание                                              |
| - | ---------------- | ----------------------------------------------------- |
| 1 | **zero-shot**    | Прямой вопрос без примеров и рассуждений              |
| 2 | **CoT**          | Пошаговое рассуждение (Chain-of-Thought)              |
| 3 | **few-shot**     | 4 примера спама + 4 примера не-спама в контексте      |
| 4 | **CoT+few-shot** | Примеры с развёрнутым пошаговым рассуждением          |

Все техники возвращают ответ в формате JSON:

```json
{"reasoning": "объяснение решения", "verdict": 0}
```

где `0` = ham (не спам), `1` = spam.

### Датасет

[SMS Spam Collection Dataset — Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

| Класс     | Кол-во | Доля  |
| --------- | ------ | ----- |
| ham       | 4825   | 86.6% |
| spam      | 747    | 13.4% |
| **Итого** | **5572** | —   |

Для оценки используется сбалансированная выборка: 100 spam + 100 ham = **200 сообщений**.

### Запуск исследования

```bash
# Установить зависимости
pip install -r requirements.txt

# Скачать датасет с Kaggle и разместить в research/data/spam.csv

# Оценить одну технику
python research/run_evaluation.py --technique zero_shot --samples 200

# Оценить все техники сразу
python research/run_evaluation.py --all --samples 200

# Результаты сохраняются в research/results/
```

### Результаты (тестовая выборка, n=200, сбалансированная: 100 spam + 100 ham)

| Техника      | Accuracy | Precision | Recall | F1     | Parse Fails |
| ------------ | -------- | --------- | ------ | ------ | ----------- |
| zero_shot    | 0.4900   | 0.4930    | 0.7000 | 0.5785 | 2           |
| cot          | 0.5550   | 0.6486    | 0.2400 | 0.3504 | 2           |
| few_shot     | 0.5450   | 0.6452    | 0.2000 | 0.3053 | 0           |
| cot_few_shot | 0.5350   | 0.6842    | 0.1300 | 0.2185 | 2           |

### Системные промпты

#### Zero-shot

Минимальный системный промпт без примеров. Модель опирается только на свои знания.

#### CoT (Chain-of-Thought)

Системный промпт задаёт 5 шагов анализа:

1. Триггерные слова (free, winner, prize, urgent)
2. Форматирование (CAPS, `!!!`, короткие коды)
3. Намерение (запрос персональных данных, денег)
4. Естественность языка
5. Итоговый вердикт

#### Few-shot

4 размеченных примера спама + 4 примера ham в системном промпте.

#### CoT + Few-shot

Примеры из few-shot дополнены пошаговым рассуждением в стиле CoT.

### Вывод

Результаты опровергли исходную гипотезу: `cot_few_shot` **не** оказался лучшим —
напротив, он показал **наихудший F1 (0.219)**. Лучшей по F1 стала `zero_shot` (0.579).

**Анализ по паттернам поведения:**

`zero_shot` агрессивно предсказывает спам: recall=0.70, но precision≈0.49 — модель
помечает спамом большинство сообщений подряд (поведение близкое к «всё спам»).
Accuracy=0.49 ≈ случайное угадывание на сбалансированной выборке.

`cot`, `few_shot`, `cot_few_shot` демонстрируют противоположный паттерн:
precision растёт (0.65–0.68), но recall катастрофически падает (0.13–0.24).
Это означает, что структурированные промпты **перегружают** 0.5B-модель:
она начинает по умолчанию предсказывать ham, изредка классифицируя очевидный спам.

**Причина:** Qwen2.5:0.5B содержит ~500M параметров — этого недостаточно для
сложного контекстного рассуждения. Длинные системные промпты CoT и few-shot
превышают эффективную «рабочую память» модели, и она теряет нить задачи.
Более простой zero-shot промпт, как ни парадоксально, позволяет модели
хоть как-то проявить встроенные знания о спаме.

**Итог:** для задачи классификации SMS-спама Qwen2.5:0.5B недостаточно мощна
ни при какой технике промптинга (лучший F1=0.579 при случайном accuracy).
Для production-применения необходима модель от 7B параметров.
Тем не менее эксперимент наглядно демонстрирует **обратный эффект усложнения
промпта на малых моделях** — ценное наблюдение для PoC.

---

## Остановка контейнера

```bash
docker compose down        # остановить контейнер
docker compose down -v     # остановить + удалить volume с моделью
```
