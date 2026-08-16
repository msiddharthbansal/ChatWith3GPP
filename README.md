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

# Chat3GPP-style RAG Chatbot

A RAG chatbot over 3GPP telecom standards (Rel-18/Rel-19, 15 specs),
built for near-zero hallucination: hybrid retrieval + reranking, mandatory
clause citation, explicit refusal on insufficient context.

**Live demo:** https://huggingface.co/spaces/msiddharth/3gpp-chatbot

Follows [Chat3GPP](https://arxiv.org/abs/2501.13954) (Huang et al., 2025)
as its architectural template: hierarchical + recursive chunking → BGE-M3
dense + BM25 sparse hybrid retrieval (RRF) → cross-encoder reranking →
grounded generation via Groq (`llama-3.1-8b-instant`). Served on HF Spaces
(Gradio, ZeroGPU); indexed in Qdrant Cloud.

## Architecture

```
   3GPP specs (docx / YAML)
              |
              v
     Ingestion & chunking
              |
              v
  Embedding (dense + sparse)
              |
              v
    Vector store (Qdrant)
              |
              v
Hybrid retrieval + reranking
              |
              v
    Grounded generation
              |
              v
      Chat interface
```

## Repo structure

```
.
├── app/
│   ├── embeddings.py
│   ├── generation.py
│   └── retrieval.py
├── ingest/
│   ├── build_manifest.py
│   ├── chunk_cr.py
│   ├── chunk_docx.py
│   └── chunk_yaml.py
├── kaggle/
│   └── build_qdrant_index.py
├── eval/
│   ├── questions.json
│   ├── run_eval.py
│   └── results.json
├── data/
│   └── manifest.json
├── gradio_app.py
├── requirements.txt
└── README.md
```
