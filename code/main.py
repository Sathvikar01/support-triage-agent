import sys
import csv
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import SUPPORT_TICKETS_DIR, DATA_DIR
from corpus_loader import load_corpus
from retriever import HybridRetriever
from pipeline import triage_ticket


OUTPUT_FIELDS = ["issue", "subject", "company", "response", "product_area", "status", "request_type", "justification"]


def run(input_csv: Path, output_csv: Path, sample: bool = False):
    print(f"Loading corpus from {DATA_DIR}...")
    documents = load_corpus()
    print(f"Loaded {len(documents)} chunks.")

    retriever = HybridRetriever(documents)
    retriever.build()

    print(f"Reading tickets from {input_csv}...")
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        tickets = list(reader)

    if sample:
        tickets = tickets[:10]
        print(f"Sample mode: processing {len(tickets)} tickets")
    else:
        print(f"Processing {len(tickets)} tickets")

    results = []
    for i, ticket in enumerate(tickets):
        issue = ticket.get("Issue", "").strip()
        subject = ticket.get("Subject", "").strip()
        company = ticket.get("Company", "None").strip()

        if not issue:
            continue

        print(f"  [{i+1}/{len(tickets)}] {subject[:50]}..." if subject else f"  [{i+1}/{len(tickets)}] {issue[:50]}...")
        start = time.time()
        result = triage_ticket(issue, subject, company, retriever)
        elapsed = time.time() - start
        print(f"    -> {result['status']} ({elapsed:.1f}s)")

        results.append({
            "issue": issue,
            "subject": subject,
            "company": company,
            "response": result["response"],
            "product_area": result["product_area"],
            "status": result["status"],
            "request_type": result["request_type"],
            "justification": result["justification"],
        })

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. Wrote {len(results)} rows to {output_csv}")
    replied = sum(1 for r in results if r["status"] == "replied")
    escalated = sum(1 for r in results if r["status"] == "escalated")
    print(f"  Replied: {replied}, Escalated: {escalated}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Support Triage Agent")
    parser.add_argument("--sample", action="store_true", help="Process only first 10 tickets")
    parser.add_argument("--input", type=str, default=str(SUPPORT_TICKETS_DIR / "support_tickets.csv"))
    parser.add_argument("--output", type=str, default=str(SUPPORT_TICKETS_DIR / "output.csv"))
    args = parser.parse_args()
    run(Path(args.input), Path(args.output), sample=args.sample)
