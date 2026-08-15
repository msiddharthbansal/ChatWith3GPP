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

A retrieval-augmented chatbot over 3GPP standards (TS 23.501, 23.502, 24.501,
33.501, 29.518; Rel-18 and Rel-19), following the architecture of
[Chat3GPP](https://arxiv.org/abs/2501.13954) (Huang et al., 2025):
hierarchical + recursive chunking → BGE-M3 dense + BM25 sparse hybrid
retrieval (RRF-fused) → BGE-M3 cross-encoder reranking → grounded generation
with mandatory clause citation and an explicit refusal path for
insufficient-context queries.
