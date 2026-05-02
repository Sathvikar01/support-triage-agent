# HackerRank Orchestrate Support Triage Agent

Production-oriented prototype for the HackerRank Orchestrate May 2026 challenge. The agent classifies, redacts, retrieves, validates, and responds to support tickets across HackerRank, Claude, and Visa using only the provided support corpus.

The design favors safe, grounded answers over free-form generation. If the system cannot find enough trusted context, detects high-risk content, or fails response validation, it escalates instead of guessing.

## What It Does

- Reads tickets from `support_tickets/support_tickets.csv`.
- Produces the required output columns in `support_tickets/output.csv`.
- Redacts common PII before retrieval and model calls.
- Classifies company, product area, and request type.
- Uses company-scoped hybrid retrieval: TF-IDF, dense embeddings, RRF, and optional cross-encoder reranking.
- Applies hard escalation categories for fraud, score manipulation, platform outage, security disclosure, unauthorized action, refund disputes, and internal-disclosure attempts.
- Computes confidence and source metadata for debugging.
- Validates generated responses and falls back to deterministic source-grounded templates when no API key is available.

## Quick Setup

```bash
pip install -r code/requirements.txt
copy .env.example .env
```

Set the API values in `.env` if you want LLM-generated responses:

```env
XIAOMI_API_KEY=your_xiaomi_api_key_here
XIAOMI_BASE_URL=https://api.xiaomi.com/v1
XIAOMI_TIMEOUT_SECONDS=20
```

Without an API key, the agent still runs in deterministic offline mode and returns extractive, source-grounded template responses.

## Run

```bash
python code/main.py
```

Useful variants:

```bash
python code/main.py --sample
python code/main.py --input support_tickets/support_tickets.csv --output support_tickets/output.csv
python code/main.py --sample --metadata-output support_tickets/output_sample_metadata.json
```

Evaluation and red-team checks:

```bash
python code/evaluate_sample.py
python code/red_team.py
```

Run tests:

```bash
cd code && pytest test_agent.py -v
```

## Output Contract

The submission CSV keeps the required fields:

| Column | Values |
|---|---|
| `status` | `replied`, `escalated` |
| `product_area` | best support category |
| `response` | customer-facing answer or escalation message |
| `justification` | concise routing and grounding explanation |
| `request_type` | `product_issue`, `feature_request`, `bug`, `invalid` |

Internally, the pipeline also tracks `resolution_status`, `company`, `confidence`, `sources`, `risk_flags`, sanitized input, and per-stage timing. These are available in the optional metadata JSON, but the main CSV stays compatible with the challenge schema.

## Architecture

```mermaid
flowchart TB
    Input["CSV or Streamlit ticket"] --> Sanitize["Sanitize input\nPII masks + injection detection"]
    Sanitize --> Split["Conservative multi-intent split"]
    Split --> Classify["Route\ncompany + product_area + request_type"]
    Classify --> Risk["Hard-risk rules\nfraud, security, outage, score manipulation"]
    Risk --> Retrieve["Company-scoped hybrid retrieval\nTF-IDF + embeddings + RRF + rerank"]
    Retrieve --> Confidence["Confidence gate\nsource domain + score + context count"]
    Confidence --> Response["LLM adapter or offline template"]
    Response --> Validate["Response validator\nno raw docs, no internal disclosure"]
    Validate --> Decision["TriageDecision\nCSV fields + metadata"]
```

### Pipeline Sequence

```mermaid
sequenceDiagram
    participant Runner as CLI
    participant Pipe as pipeline.py
    participant Guard as sanitizer/escalation
    participant RAG as retriever.py
    participant Model as model_client.py
    participant Check as validator.py

    Runner->>Pipe: issue, subject, company
    Pipe->>Guard: redact and inspect
    Guard-->>Pipe: sanitized text, risk flags
    Pipe->>Pipe: classify ticket
    Pipe->>Guard: hard-risk check
    alt high risk
        Pipe-->>Runner: escalated decision
    else safe to retrieve
        Pipe->>RAG: retrieve(query, company)
        RAG-->>Pipe: chunks, scores, source metadata
        Pipe->>Guard: confidence and source-domain check
        alt low confidence
            Pipe-->>Runner: escalated decision
        else enough context
            Pipe->>Model: source-grounded prompt
            Model-->>Pipe: response or None
            Pipe->>Check: validate response
            Check-->>Pipe: pass/fail flags
            Pipe-->>Runner: replied or escalated decision
        end
    end
```

### Core Design Choices

- **Unified decision object:** `TriageDecision` carries the required CSV fields plus richer metadata: `resolution_status`, `company`, `confidence`, `sources`, `risk_flags`, sanitized input, and timings.
- **Company-scoped retrieval:** the retriever indexes all support documents but filters retrieval by the declared or inferred company before reranking. This prevents Claude documents from grounding a HackerRank answer.
- **Hybrid search:** TF-IDF protects exact product terms and error strings, dense retrieval handles semantic phrasing, RRF merges both, and the cross-encoder reranker improves final ordering when available.
- **Query expansion:** maps common support phrases to expanded terminology before retrieval (e.g., "stolen" -> "lost stolen card replacement").
- **Metadata boosting:** results from the expected company get a 15% score boost.
- **Deterministic safety first:** prompt injection, internal-disclosure requests, fraud, score manipulation, unauthorized action, platform outage, and refund disputes are routed before generation.
- **Offline deterministic mode:** if no API key is configured, the agent still produces a source-grounded template response instead of failing.
- **Response validation:** generated answers are checked for internal disclosure, raw markdown leakage, missing sources, and empty/overlong output. Unsafe generated responses are replaced with deterministic fallbacks or escalated.
- **Response cleaning:** LLM output is post-processed to strip markdown headers, image links, and URLs before validation and output.

### Escalation Categories

| Category | Example | Handling |
|---|---|---|
| `fraud` | stolen identity, fraudulent charge | escalate to human/security handling |
| `security` | vulnerability disclosure | escalate to security review |
| `score_manipulation` | change score, override hiring | refuse modification and route appropriately |
| `unauthorized_action` | destructive code or active abuse | refuse and escalate |
| `platform_outage` | all requests failing, site down | escalate to operations/human follow-up |
| `refund_demand` | immediate refund demand | escalate to billing/human review |
| `internal_disclosure` | reveal hidden rules or retrieved docs | refuse internal disclosure and escalate |
| `insufficient_context` | weak/no source match | escalate rather than guess |

Defensive security questions such as "How do I prevent SQL injection?" are not treated as abuse threats.

### Retrieval And Cache Invalidation

The vector cache lives under `vector_db/` and is ignored by git. `retriever.py` writes a `manifest.json` containing:

- cache version
- document count
- corpus hash
- embedding model
- reranker model

If any of these values changes, the cache is considered stale and rebuilt. This avoids silently pairing old vector IDs with new markdown chunks.

If optional ML dependencies are unavailable, the retriever degrades to TF-IDF-only mode instead of crashing. That keeps the submission runnable on constrained machines, though quality will be lower.

### Model Boundary

`model_client.py` wraps the Xiaomi/OpenAI-compatible endpoint behind a small adapter:

- API key and base URL are read only from `.env`/environment variables.
- Timeout is controlled by `XIAOMI_TIMEOUT_SECONDS`.
- The client uses low temperature and one retry.
- The client is cached for connection pooling across requests.
- The model receives sanitized ticket text and selected support chunks only.
- If the client is unavailable, the deterministic template fallback is used.

### Safety Model

The security layer is intentionally layered instead of regex-only:

- PII redaction runs before retrieval and model calls.
- The subject and body are both scanned for prompt-injection and internal-disclosure attempts.
- Hard-risk categories are deterministic and map to explicit response templates.
- Retrieval confidence and source-company agreement decide whether an answer is safe.
- Generated answers are validated before being returned.
- Raw retrieved chunks are shown only in the local telemetry logs, not required in the final CSV.

## Repository Layout

```
.
├── README.md                       # this file (architecture + setup)
├── .env.example                    # copy to .env; never commit .env
├── .gitignore
├── AGENTS.md                       # agent onboarding and logging protocol
├── evalutation_criteria.md         # official judging rubric
├── problem_statement.md            # challenge requirements
├── code/
│   ├── main.py                     # CLI batch runner
│   ├── pipeline.py                 # decision pipeline (CLI + UI)
│   ├── decision.py                 # TriageDecision and SourceRef types
│   ├── corpus_loader.py            # markdown-aware corpus chunking
│   ├── retriever.py                # hybrid retriever with cache
│   ├── classifier.py               # company, product area, request type
│   ├── escalation.py               # hard-risk rules and confidence gates
│   ├── sanitizer.py                # PII masking and injection detection
│   ├── responder.py                # LLM generation + template fallback
│   ├── model_client.py             # OpenAI-compatible model adapter
│   ├── validator.py                # response safety checks
│   ├── multi_intent.py             # multi-intent splitting
│   ├── evaluate_sample.py          # sample-label evaluation harness
│   ├── red_team.py                 # safety regression cases
│   ├── test_agent.py               # pytest test suite
│   ├── config.py                   # constants and keyword dictionaries
│   └── requirements.txt
├── data/
│   ├── hackerrank/                 # HackerRank support corpus
│   ├── claude/                     # Claude support corpus
│   └── visa/                       # Visa support corpus
├── support_tickets/
│   ├── sample_support_tickets.csv  # sample with expected outputs
│   ├── support_tickets.csv         # full input for submission
│   └── output.csv                  # generated output
└── vector_db/                      # cached retrieval index (gitignored)
```

## Current Trade-Offs

- Heuristic routing is fast and explainable, but misses subtle intent.
- Conservative confidence gates improve safety but can over-escalate.
- Template fallback is reliable offline, but less fluent than a model response.
- Source citations improve defensibility, but the final CSV schema has no dedicated source column, so source labels are included in justification and metadata.
- Streamlit telemetry was removed in favor of pure CLI evaluation entry points.
