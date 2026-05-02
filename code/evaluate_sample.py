import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, SUPPORT_TICKETS_DIR
from corpus_loader import load_corpus
from pipeline import triage_decision
from retriever import HybridRetriever


def _norm(value: str) -> str:
    return (value or "").strip().lower()


def evaluate(input_csv: Path, limit: int = 0):
    documents = load_corpus()
    retriever = HybridRetriever(documents)
    retriever.build()

    with open(input_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]

    counts = {
        "rows": 0,
        "status_correct": 0,
        "request_type_correct": 0,
        "product_area_correct": 0,
        "expected_escalated": 0,
        "predicted_escalated": 0,
        "true_escalated": 0,
        "replied_with_sources": 0,
        "validation_flags": 0,
    }
    latencies = []
    failures = []

    for row in rows:
        issue = row.get("Issue", "")
        subject = row.get("Subject", "")
        company = row.get("Company", "None")
        expected_status = _norm(row.get("Status"))
        expected_request_type = _norm(row.get("Request Type"))
        expected_product_area = _norm(row.get("Product Area"))

        started = time.time()
        decision = triage_decision(issue, subject, company, retriever)
        elapsed = time.time() - started
        latencies.append(elapsed)

        predicted_status = _norm(decision.status)
        predicted_request_type = _norm(decision.request_type)
        predicted_product_area = _norm(decision.product_area)

        counts["rows"] += 1
        counts["status_correct"] += predicted_status == expected_status
        counts["request_type_correct"] += predicted_request_type == expected_request_type
        counts["product_area_correct"] += predicted_product_area == expected_product_area
        counts["expected_escalated"] += expected_status == "escalated"
        counts["predicted_escalated"] += predicted_status == "escalated"
        counts["true_escalated"] += expected_status == "escalated" and predicted_status == "escalated"
        counts["replied_with_sources"] += decision.status == "replied" and bool(decision.sources)
        counts["validation_flags"] += bool(decision.risk_flags)

        if predicted_status != expected_status or predicted_request_type != expected_request_type:
            failures.append({
                "subject": subject,
                "expected_status": expected_status,
                "predicted_status": predicted_status,
                "expected_request_type": expected_request_type,
                "predicted_request_type": predicted_request_type,
                "confidence": round(decision.confidence, 3),
            })

    rows_count = max(counts["rows"], 1)
    precision_denominator = max(counts["predicted_escalated"], 1)
    recall_denominator = max(counts["expected_escalated"], 1)
    avg_latency = sum(latencies) / rows_count

    print("Evaluation summary")
    print(f"Rows: {counts['rows']}")
    print(f"Classification accuracy: {counts['request_type_correct'] / rows_count:.2%}")
    print(f"Product area accuracy: {counts['product_area_correct'] / rows_count:.2%}")
    print(f"Status accuracy: {counts['status_correct'] / rows_count:.2%}")
    print(f"Escalation precision: {counts['true_escalated'] / precision_denominator:.2%}")
    print(f"Escalation recall: {counts['true_escalated'] / recall_denominator:.2%}")
    print(f"Grounded reply proxy: {counts['replied_with_sources'] / rows_count:.2%}")
    print(f"Validation/risk flag rate: {counts['validation_flags'] / rows_count:.2%}")
    print(f"Average latency: {avg_latency:.2f}s")

    if failures:
        print("\nFailure cases")
        for failure in failures[:10]:
            print(f"- {failure}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sample support tickets.")
    parser.add_argument("--input", default=str(SUPPORT_TICKETS_DIR / "sample_support_tickets.csv"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    evaluate(Path(args.input), args.limit)
