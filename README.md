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

Full design notes, scope decisions, bugs found and fixed, and evaluation
results: **[WRITEUP.md](WRITEUP.md)**.

## Repo structure

```
ingest/       chunk_docx.py, chunk_yaml.py, chunk_cr.py, build_manifest.py
kaggle/       build_qdrant_index.py — bulk + incremental embedding/upsert
app/          embeddings.py, retrieval.py, generation.py — the serving pipeline
gradio_app.py HF Spaces entry point (chat UI + /answer_mcq API)
eval/         questions.json, run_eval.py, results.json — 94.7% MCQ accuracy
data/         manifest.json (provenance); raw/extracted/processed are
              gitignored — regenerable, not needed at runtime
```
