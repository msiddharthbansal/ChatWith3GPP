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
rewriting, evidence-aware context packing, a pre-generation confidence
gate, mandatory clause citation, post-generation citation verification,
and explicit refusal on insufficient context.

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
  Query rewriting
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
    Citation check
        |
        v
    Chat interface
```

Each stage of the query pipeline past retrieval was added to close a
specific failure found via live testing (fabricated citations,
retrieval-ranking misses, budget truncation silently dropping the right
chunk) — not designed upfront.

## Repo structure

```
.
├── storage_pipeline/
│   ├── build_manifest.py
│   ├── chunk_cr.py
│   ├── chunk_docx.py
│   ├── chunk_yaml.py
│   └── build_qdrant_index.py
├── query_pipeline/
│   ├── embeddings.py
│   ├── query_rewrite.py
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
