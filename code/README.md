# Support Triage Agent — Code

Terminal-based multi-domain support triage agent for HackerRank, Claude, and Visa.

## Architecture

3-stage retrieval pipeline with PII safety guardrails and multi-intent decomposition:

1. **Corpus Loader** — markdown-aware semantic chunking (splits by `##` headers, prepends parent heading)
2. **Hybrid Retriever** — TF-IDF + `all-MiniLM-L6-v2` embeddings + RRF fusion → cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`)
3. **Pipeline** — sanitize → classify → retrieve → escalate → respond

## Setup

```bash
# From the repo root
pip install -r code/requirements.txt

# Set your Xiaomi API key in .env
cp .env.example .env
# Edit .env with XIAOMI_API_KEY
```

## Run

```bash
# Full run
python code/main.py

# Sample run (first 10 tickets)
python code/main.py --sample

# Custom input/output
python code/main.py --input path/to/input.csv --output path/to/output.csv
```

## Modules

| Module | Purpose |
|---|---|
| `config.py` | Constants, keyword maps, Xiaomi API config |
| `corpus_loader.py` | Markdown-aware chunking of `data/` corpus |
| `retriever.py` | TF-IDF + SentenceTransformer embeddings + FAISS + cross-encoder reranking |
| `classifier.py` | Company inference, request_type, product_area |
| `escalation.py` | Hard rules (fraud/security/outage) + relevance fallback |
| `responder.py` | Xiaomi `mimo-v2.5` LLM generation + template fallback |
| `sanitizer.py` | PII masking + prompt injection detection |
| `multi_intent.py` | Compound ticket splitting + merge logic |
| `pipeline.py` | Orchestrator: sanitize → classify → retrieve → escalate → respond |
| `main.py` | CLI entry point: CSV read → process → write output.csv |
