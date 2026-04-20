"""
run_evaluation.py — оценка техник промптинга для классификации SMS-спама.

Загружает датасет SMS Spam Collection, запускает инференс через FastAPI-сервис
для каждой выбранной техники промптинга, вычисляет метрики и сохраняет отчёт.

Датасет: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
  - Файл: spam.csv (или SMSSpamCollection.tsv без заголовка)
  - Разместите файл в папке research/data/spam.csv перед запуском.

Использование:
    python research/run_evaluation.py --technique zero_shot
    python research/run_evaluation.py --technique cot --samples 100
    python research/run_evaluation.py --all --samples 200
    python research/run_evaluation.py --all --samples 200 --output research/results

Выходные файлы (в папке --output):
    {technique}_predictions.json  — предсказания и метки
    {technique}_metrics.json      — accuracy / precision / recall / f1
    summary_report.json           — сводная таблица всех техник
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).parent))
from prompts import TECHNIQUES, get_prompt

SERVICE_URL = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(data_path: Path, n_samples: int, seed: int = 42) -> pd.DataFrame:
    """
    Load and balance the SMS Spam Collection dataset.

    Reads a CSV file with 'label' (ham/spam) and 'message' columns,
    then draws an equal number of spam and ham samples for unbiased evaluation.

    Args:
        data_path: Path to the CSV file (spam.csv or SMSSpamCollection.tsv).
        n_samples: Total number of samples (split evenly between ham and spam).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns ['label', 'message', 'label_int'],
        where label_int is 1 for spam and 0 for ham.

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If the file has fewer spam samples than requested.
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}.\n"
            "Download from https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset\n"
            "and place spam.csv in research/data/"
        )

    # Support both CSV (with header) and TSV (no header, tab-separated)
    if data_path.suffix.lower() in (".tsv", ".txt"):
        df = pd.read_csv(data_path, sep="\t", header=None, names=["label", "message"])
    else:
        df = pd.read_csv(data_path, encoding="latin-1")
        df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "message"})
        df = df[["label", "message"]]

    df["label"] = df["label"].str.strip().str.lower()
    df["label_int"] = df["label"].map({"spam": 1, "ham": 0})

    per_class = n_samples // 2
    spam_df = df[df["label"] == "spam"].sample(n=per_class, random_state=seed)
    ham_df = df[df["label"] == "ham"].sample(n=per_class, random_state=seed)

    balanced = pd.concat([spam_df, ham_df]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return balanced


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def call_service(base_url: str, system: str, prompt: str, retries: int = 3) -> str:
    """
    Send a generation request to the FastAPI service with retry logic.

    Args:
        base_url: Base URL of the running FastAPI service.
        system: System prompt string (may be empty).
        prompt: User prompt string.
        retries: Number of retry attempts on failure.

    Returns:
        The model's raw text response string.

    Raises:
        RuntimeError: If all retry attempts fail.
    """
    payload = {"prompt": prompt, "system": system}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(f"{base_url}/generate", json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()["response"]
        except requests.RequestException as exc:
            if attempt == retries:
                raise RuntimeError(f"Service call failed after {retries} attempts: {exc}") from exc
            time.sleep(2)
    return ""


def parse_verdict(raw_response: str) -> int | None:
    """
    Extract the integer verdict (0 or 1) from a raw LLM response string.

    Tries three strategies in order:
    1. JSON parsing of the first {...} block.
    2. Regex search for '"verdict": <digit>'.
    3. Regex search for standalone '0' or '1' as the last digit in the response.

    Args:
        raw_response: Raw text output from the LLM.

    Returns:
        0 or 1 if a verdict could be extracted, None otherwise.
    """
    # Strategy 1: parse JSON block
    try:
        start = raw_response.find("{")
        end = raw_response.rfind("}") + 1
        if start != -1 and end > start:
            data = json.loads(raw_response[start:end])
            verdict = data.get("verdict")
            if verdict in (0, 1):
                return int(verdict)
    except (json.JSONDecodeError, TypeError):
        pass

    # Strategy 2: regex for "verdict": 0 or 1
    match = re.search(r'"verdict"\s*:\s*([01])', raw_response)
    if match:
        return int(match.group(1))

    # Strategy 3: last standalone digit
    digits = re.findall(r'\b([01])\b', raw_response)
    if digits:
        return int(digits[-1])

    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict:
    """
    Compute classification metrics for binary spam detection.

    Args:
        y_true: Ground-truth labels (0 = ham, 1 = spam).
        y_pred: Model predictions (0 = ham, 1 = spam).

    Returns:
        Dict with keys: accuracy, precision, recall, f1, n_total, n_parsed.
    """
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "n_total": len(y_true),
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate_technique(
    technique: str,
    df: pd.DataFrame,
    base_url: str,
    output_dir: Path,
) -> dict:
    """
    Run inference for a single prompting technique and compute metrics.

    For each SMS in df, builds the prompt, calls the LLM service, parses the
    verdict, and accumulates results. Saves per-sample predictions and aggregate
    metrics as JSON files in output_dir.

    Args:
        technique: Name of the prompting technique (key in TECHNIQUES).
        df: DataFrame with 'message' and 'label_int' columns.
        base_url: Base URL of the FastAPI service.
        output_dir: Directory where results are saved.

    Returns:
        Dict with metric values for this technique (also written to disk).
    """
    print(f"\n{'='*60}")
    print(f"Technique: {technique.upper()}  ({len(df)} samples)")
    print("=" * 60)

    predictions = []
    y_true: list[int] = []
    y_pred: list[int] = []
    n_parse_fail = 0

    for idx, row in df.iterrows():
        system, user = get_prompt(technique, row["message"])
        raw = call_service(base_url, system, user)
        verdict = parse_verdict(raw)

        if verdict is None:
            print(f"  [{idx}] PARSE FAIL — raw: {raw[:80]!r}")
            n_parse_fail += 1
            verdict = 0  # default to ham on parse failure

        label = int(row["label_int"])
        y_true.append(label)
        y_pred.append(verdict)

        status = "✓" if verdict == label else "✗"
        print(f"  [{idx:4d}] {status} true={label} pred={verdict}  {row['message'][:60]!r}")

        predictions.append(
            {
                "index": int(idx),
                "message": row["message"],
                "true_label": label,
                "predicted": verdict,
                "raw_response": raw,
            }
        )

    metrics = compute_metrics(y_true, y_pred)
    metrics["n_parse_fail"] = n_parse_fail

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{technique}_predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / f"{technique}_metrics.json").write_text(
        json.dumps({"technique": technique, **metrics}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nMetrics: {metrics}")
    return metrics


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def save_summary(results: dict[str, dict], output_dir: Path) -> None:
    """
    Save a combined JSON and Markdown summary of all evaluated techniques.

    Args:
        results: Mapping from technique name to its metrics dict.
        output_dir: Directory to write summary_report.json and summary_report.md.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "summary_report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Markdown table
    header = "| Technique | Accuracy | Precision | Recall | F1 | Parse Fails |\n"
    sep = "|-----------|----------|-----------|--------|----|-------------|\n"
    rows = ""
    for tech, m in results.items():
        rows += (
            f"| {tech} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['f1']:.4f} | {m.get('n_parse_fail', 0)} |\n"
        )

    md = f"# Prompting Techniques — Evaluation Summary\n\n{header}{sep}{rows}"
    (output_dir / "summary_report.md").write_text(md, encoding="utf-8")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(header + sep + rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate LLM prompting techniques on SMS spam dataset.")
    parser.add_argument(
        "--technique",
        choices=list(TECHNIQUES),
        help="Single prompting technique to evaluate.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all four techniques sequentially.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Total number of samples (balanced: half spam, half ham). Default: 200.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("research/data/spam.csv"),
        help="Path to the SMS Spam Collection CSV file.",
    )
    parser.add_argument(
        "--url",
        default=SERVICE_URL,
        help="Base URL of the FastAPI service.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("research/results"),
        help="Directory for output files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset sampling.",
    )
    return parser.parse_args()


def main() -> None:
    """
    CLI entry point for the evaluation pipeline.

    Loads the dataset, runs inference for selected technique(s),
    and writes metrics and a summary report to disk.
    """
    args = parse_args()

    if not args.technique and not args.all:
        print("Specify --technique <name> or --all")
        sys.exit(1)

    print(f"Loading dataset from {args.data} ({args.samples} balanced samples)...")
    df = load_dataset(args.data, args.samples, seed=args.seed)
    print(f"Loaded {len(df)} samples: {df['label'].value_counts().to_dict()}")

    techniques = list(TECHNIQUES) if args.all else [args.technique]

    all_results: dict[str, dict] = {}
    for tech in techniques:
        metrics = evaluate_technique(tech, df, args.url, args.output)
        all_results[tech] = metrics

    if len(all_results) > 1:
        save_summary(all_results, args.output)
    else:
        print(f"\nDone. Results saved to {args.output}/")


if __name__ == "__main__":
    main()
