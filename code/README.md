# Support Triage Agent Code

This directory contains the runnable Python agent for the HackerRank Orchestrate support-ticket challenge.

## Main Entry Points

```bash
python code/main.py
python code/main.py --sample
python code/evaluate_sample.py
python code/red_team.py
```

The main runner writes the required submission CSV. The optional metadata output exposes confidence, sources, risk flags, sanitized input, and timing:

```bash
python code/main.py --sample --metadata-output support_tickets/output_sample_metadata.json
```

## Modules

| Module | Purpose |
|---|---|
| `main.py` | CSV batch runner and output writer |
| `pipeline.py` | Shared decision flow for CLI and UI |
| `decision.py` | Unified `TriageDecision` and source metadata types |
| `corpus_loader.py` | Markdown-aware corpus chunking |
| `retriever.py` | TF-IDF, optional dense retrieval, RRF, reranking, cache manifest |
| `classifier.py` | Company, product area, and request-type routing |
| `escalation.py` | High-risk rules and confidence fallback routing |
| `sanitizer.py` | PII masking and prompt-injection detection |
| `responder.py` | LLM response generation plus deterministic template fallback |
| `model_client.py` | OpenAI-compatible model adapter with timeout/retry policy |
| `validator.py` | Response safety and groundedness validation |
| `multi_intent.py` | Conservative multi-intent splitting |
| `evaluate_sample.py` | Sample-label evaluation harness |
| `red_team.py` | Safety regression cases |

## Decision Flow

`sanitize -> classify -> hard-risk check -> company-scoped retrieval -> confidence check -> generate/template -> validate -> output`

The CSV still uses HackerRank's required `status` values: `replied` or `escalated`. The richer internal `resolution_status` is for debugging and UI telemetry.
