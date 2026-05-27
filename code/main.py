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
        try:
            decision = triage_decision(issue, subject, company, retriever)
        except Exception as exc:
            print(f"    !! ERROR: {exc}")
            from decision import TriageDecision
            decision = TriageDecision(
                status="escalated",
                response="An error occurred while processing this ticket. A human agent will review it.",
                product_area="general",
                request_type="product_issue",
                justification=f"Pipeline error: {type(exc).__name__}: {str(exc)[:200]}",
                company=company if company not in ("", "None", None) else "Unknown",
                resolution_status="error",
                confidence=0.0,
                risk_flags=["pipeline_error"],
                sanitized_query=issue,
                sanitized_subject=subject,
                telemetry={"total_seconds": time.time() - start},
            )
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
    parser.add_argument("--voice", action="store_true", help="Enable voice output (TTS)")
    parser.add_argument("--voice-name", type=str, default="Mia", help="TTS voice: Mia/Chloe/Milo/Dean")
    parser.add_argument("--voice-input", type=str, default="", help="Path to audio file for STT input")
    parser.add_argument("--interactive", action="store_true", help="Interactive voice REPL mode")
    parser.add_argument("--audio-dir", type=str, default="", help="Output directory for audio files")
    args = parser.parse_args()

    # Voice agent mode
    if args.voice or args.voice_input or args.interactive:
        from voice_agent import VoiceAgent
        from config import AUDIO_OUTPUT_DIR
        agent = VoiceAgent(
            voice=args.voice_name,
            output_dir=args.audio_dir or str(AUDIO_OUTPUT_DIR),
        )
        if args.voice_input:
            result = agent.process_voice_input(args.voice_input)
            if result.get("error"):
                print(f"Error: {result['error']}")
            else:
                print(f"Transcription: {result['transcription']}")
                print(f"Status: {result['status']}")
                print(f"Response: {result['text_response']}")
                if result["audio_path"]:
                    print(f"Audio: {result['audio_path']}")
        elif args.interactive:
            agent.run_interactive(voice=args.voice_name)
        else:
            # Batch mode with voice output
            agent.run_batch(Path(args.input), voice=args.voice_name)
    else:
        # Standard text-only mode
        metadata_output = Path(args.metadata_output) if args.metadata_output else None
        run(Path(args.input), Path(args.output), sample=args.sample, metadata_output=metadata_output)
