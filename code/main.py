import sys
import csv
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import SUPPORT_TICKETS_DIR, DATA_DIR
from corpus_loader import load_corpus
from retriever import HybridRetriever
from pipeline import triage_decision


OUTPUT_FIELDS = ["status", "product_area", "response", "justification", "request_type"]


def run(input_csv: Path, output_csv: Path, sample: bool = False, metadata_output: Path = None):
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
    metadata = []
    for i, ticket in enumerate(tickets):
        issue = (ticket.get("Issue") or ticket.get("issue") or ticket.get("Description") or ticket.get("description") or "").strip()
        subject = (ticket.get("Subject") or ticket.get("subject") or "").strip()
        company = (ticket.get("Company") or ticket.get("company") or "None").strip()

        if not issue:
            continue

        print(f"  [{i+1}/{len(tickets)}] {subject[:50]}..." if subject else f"  [{i+1}/{len(tickets)}] {issue[:50]}...")
        start = time.time()
        decision = triage_decision(issue, subject, company, retriever)
        result = decision.to_dict()
        elapsed = time.time() - start
        print(f"    -> {result['status']} / {result['resolution_status']} / confidence={result['confidence']} ({elapsed:.1f}s)")

        results.append(decision.to_submission_row(issue, subject, company))
        metadata.append({
            "issue": issue,
            "subject": subject,
            "input_company": company,
            **result,
        })

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone. Wrote {len(results)} rows to {output_csv}")
    replied = sum(1 for r in results if r["status"] == "replied")
    escalated = sum(1 for r in results if r["status"] == "escalated")
    print(f"  Replied: {replied}, Escalated: {escalated}")

    if metadata_output:
        with open(metadata_output, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Metadata: {metadata_output}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Support Triage Agent")
    parser.add_argument("--sample", action="store_true", help="Process only first 10 tickets")
    parser.add_argument("--input", type=str, default=str(SUPPORT_TICKETS_DIR / "support_tickets.csv"))
    parser.add_argument("--output", type=str, default=str(SUPPORT_TICKETS_DIR / "output.csv"))
    parser.add_argument("--metadata-output", type=str, default="", help="Optional JSONL-style debug metadata path")
    args = parser.parse_args()
    metadata_output = Path(args.metadata_output) if args.metadata_output else None
    run(Path(args.input), Path(args.output), sample=args.sample, metadata_output=metadata_output)
