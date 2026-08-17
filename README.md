---
title: 3GPP Chatbot
emoji: 📡
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.24.0
app_file: gradio_app.py
pinned: false
---

# 3GPP RAG Chatbot

A RAG chatbot over 3GPP telecom standards (Rel-18/Rel-19, 15 specs),
built for near-zero hallucination: hybrid retrieval + reranking, query
rewriting grounded in an official 3GPP terminology glossary (TR 21.905),
evidence-aware context packing, a pre-generation confidence gate, and
explicit refusal on insufficient context. Every answer's sources are
tracked and shown in a separate panel in the UI, kept out of the answer
text itself for readability.

**Live demo:** https://huggingface.co/spaces/msiddharth/3gpp-chatbot

Hierarchical + recursive chunking → BGE-M3 dense + BM25 sparse hybrid
retrieval (RRF) → cross-encoder reranking → grounded generation via Groq
(`llama-3.1-8b-instant`). Served on HF Spaces (Gradio, ZeroGPU); indexed in
Qdrant Cloud.

## Architecture

Two separate pipelines, split by when they run: a **storage pipeline**
(offline, run once per corpus update) and a **query pipeline** (online,
per user request).

```
STORAGE PIPELINE (offline)
───────────────────────────
   3GPP specs (docx / YAML / CR)
              |
              v
     Ingestion & chunking
              |
              v
  Embedding (dense + sparse)
              |
              v
    Vector store (Qdrant)


QUERY PIPELINE (online, per request)
─────────────────────────────────────
     User query
        |
        v
  Query rewriting  <---- 3GPP glossary lookup (TR 21.905)
        |
        v
Hybrid retrieval + reranking
        |
        v
Evidence-aware context packing
        |
        v
    Sufficiency gate ----(low confidence)----> Refusal
        |
   (sufficient)
        |
        v
  Grounded generation
        |
        v
    Citation safety check
        |
        v
    Chat interface (sources shown separately, not inline)
```

Each stage of the query pipeline past retrieval was added to close a
specific failure found via live testing (fabricated citations,
retrieval-ranking misses, budget truncation silently dropping the right
chunk) — not designed upfront. The glossary lookup queries TR 21.905
(official 3GPP term/abbreviation definitions) separately from the main
corpus, purely to ground query rewriting — glossary content never enters
the answer context or the visible sources.

## Repo structure

```
.
├── storage_pipeline/
│   ├── build_manifest.py
│   ├── chunk_cr.py
│   ├── chunk_docx.py
│   ├── chunk_yaml.py
│   ├── chunk_glossary.py
│   └── build_qdrant_index.py
├── query_pipeline/
│   ├── embeddings.py
│   ├── query_rewrite.py
│   ├── glossary_lookup.py
│   ├── retrieval.py
│   ├── context_packing.py
│   ├── sufficiency_gate.py
│   ├── citation_check.py
│   ├── prompts.py
│   └── generation.py
├── data/
│   └── manifest.json
├── gradio_app.py
├── requirements.txt
└── README.md
```

## Setup (to run locally)

The fastest way to validate the system is the live demo above — no setup
needed. To run the query pipeline yourself (the corpus is already indexed
in Qdrant; you don't need to rebuild it):

**1. Prerequisites**
- Python 3.11
- The project's `.env` (`QDRANT_URL`, `QDRANT_API_KEY`, `GROQ_API_KEY`) —
  provided separately

**2. Install & run**
```
git clone https://github.com/msiddharthbansal/ChatWith3GPP.git
cd ChatWith3GPP
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# place the provided .env file in the repo root
python gradio_app.py
```

Note: `query_pipeline/embeddings.py` loads models onto `cuda` — this
matches how it's deployed (Hugging Face Spaces ZeroGPU) but means it will
raise `AssertionError: Torch not compiled with CUDA enabled` on a
CPU-only machine. Running it locally without a GPU requires changing the
`device`/`devices` arguments in that file to `"cpu"` first (functional,
just noticeably slower per query on CPU — confirmed directly during this
project's own testing).
