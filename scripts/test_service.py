"""
test_service.py — скрипт для тестирования FastAPI LLM-сервиса вне контейнера.

Запускается локально и отправляет HTTP-запросы к проброшенному порту 8000.

Использование:
    python scripts/test_service.py
    python scripts/test_service.py --url http://localhost:8000
"""

import argparse
import json
import sys

import requests

SERVICE_URL = "http://localhost:8000"

SPAM_SMS = "Congratulations! You've won a FREE £1000 Tesco gift card. Click http://bit.ly/win123 to claim NOW!"
HAM_SMS = "Hey, are we still meeting at 3pm today? Let me know if you're running late."


def check_health(base_url: str) -> bool:
    """
    Ping the /health endpoint to verify the service is running.

    Args:
        base_url: Base URL of the FastAPI service.

    Returns:
        True if the service responds with status 200, False otherwise.
    """
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        resp.raise_for_status()
        print(f"[health] {resp.json()}")
        return True
    except requests.RequestException as exc:
        print(f"[health] FAILED: {exc}")
        return False


def list_models(base_url: str) -> None:
    """
    Call /models to verify Ollama has the required model loaded.

    Args:
        base_url: Base URL of the FastAPI service.
    """
    resp = requests.get(f"{base_url}/models", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    models = [m["name"] for m in data.get("models", [])]
    print(f"[models] Available models: {models}")


def generate(base_url: str, prompt: str, system: str = "") -> str:
    """
    Send a generation request to /generate and return the model's response text.

    Args:
        base_url: Base URL of the FastAPI service.
        prompt: User prompt to send to the model.
        system: Optional system prompt.

    Returns:
        The model's response as a string.

    Raises:
        requests.HTTPError: If the service returns a non-2xx status.
    """
    payload = {"prompt": prompt, "system": system}
    resp = requests.post(f"{base_url}/generate", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"]


def run_spam_test(base_url: str, sms: str, label: str) -> None:
    """
    Run a zero-shot spam classification test for a single SMS message.

    Prints the SMS label, text, and model verdict to stdout.

    Args:
        base_url: Base URL of the FastAPI service.
        sms: The SMS text to classify.
        label: Human-readable label for the test case (e.g. 'SPAM' or 'HAM').
    """
    print(f"\n--- Test: {label} ---")
    print(f"SMS: {sms}")
    prompt = (
        f'Classify this SMS as spam (1) or ham (0).\n'
        f'SMS: {sms}\n'
        f'Output ONLY valid JSON: {{"reasoning": "...", "verdict": 0}}'
    )
    response = generate(base_url, prompt)
    print(f"Model response: {response}")
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        parsed = json.loads(response[start:end])
        verdict = parsed.get("verdict")
        print(f"Parsed verdict: {verdict} ({'spam' if verdict == 1 else 'ham'})")
    except (json.JSONDecodeError, ValueError):
        print("Could not parse JSON from response.")


def main() -> None:
    """
    Entry point: runs health check, model list, and two spam classification tests.
    """
    parser = argparse.ArgumentParser(description="Test the LLM spam detection service.")
    parser.add_argument("--url", default=SERVICE_URL, help="Base URL of the FastAPI service.")
    args = parser.parse_args()

    print(f"Testing service at: {args.url}\n")

    print("=== [1] Health Check ===")
    if not check_health(args.url):
        print("Service is not available. Exiting.")
        sys.exit(1)

    print("\n=== [2] Available Models ===")
    try:
        list_models(args.url)
    except requests.RequestException as exc:
        print(f"Could not list models: {exc}")

    print("\n=== [3] Spam Classification Tests ===")
    run_spam_test(args.url, SPAM_SMS, "EXPECTED SPAM")
    run_spam_test(args.url, HAM_SMS, "EXPECTED HAM")

    print("\n=== All tests completed ===")


if __name__ == "__main__":
    main()
