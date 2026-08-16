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

A retrieval-augmented chatbot over 3GPP telecom standards, built for a
Graduate Engineer Trainee take-home assignment: **build a working RAG
chatbot over 3GPP standards documentation, explicitly optimized for
minimal-to-near-zero hallucination.**

**Live demo:** https://huggingface.co/spaces/msiddharth/3gpp-chatbot

---

## 1. Problem scope

The assignment is evaluated on three axes, and this document — along with
the code — is written to speak to all three:

1. **Quality and effectiveness of the submitted project.**
2. **Understanding of the solution design and code implementation.**
3. **Performance during the technical interview.**

Given a short build window, scope was narrowed deliberately rather than
attempted broadly and shallowly:

- **Releases: Rel-18 and Rel-19 only.** Both are frozen (no moving-target
  risk during development). Rel-16/17 and all-series coverage were
  explicitly considered and rejected as out of scope for the timeline —
  framed here as a scoping decision, not an oversight. Multi-release
  ingestion without release-aware metadata is a genuine **hallucination
  risk**, not just a scope problem: the same clause number can mean
  different things across releases, so release-awareness (see §3) was
  treated as a correctness requirement, not a nice-to-have.
- **Corpus: 15 full base specs**, spanning the 5G core architecture, NAS
  protocols, security, RAN, positioning, and service-based interfaces,
  plus a handful of **Change Request (CR)** documents layered on top (see
  §3.3). The initial 5-spec scope (23.501, 23.502, 24.501, 33.501, 29.518)
  was expanded mid-build after live testing surfaced real coverage gaps;
  see §5 for how that was diagnosed.

---

## 2. Architectural prior art

The design follows **[Chat3GPP](https://arxiv.org/abs/2501.13954)**
(Huang et al., Tsinghua, 2025) as its primary template: hierarchical +
recursive chunking → BGE-M3 dense + BM25 sparse hybrid retrieval (RRF
fusion) → BGE-M3-family cross-encoder reranking → prompted generation with
an explicit refusal instruction. Chat3GPP reports 78.7% accuracy on
TeleQnA (Rel-17/18) **zero-shot**, beating fine-tuned baselines including
**Telco-RAG** (Bornea et al., arXiv:2404.15939), which instead compensates
for weaker retrieval (dense-only FAISS, no reranking, 125-token chunks)
with a trained neural-network query router. Chat3GPP's hybrid
retrieval + reranking approach is architecturally simpler than Telco-RAG's
router — no training data, no closed-model dependency (Telco-RAG uses
OpenAI embeddings + GPT-3.5) — while matching or beating its accuracy, which
is why it was chosen as the template here.

---

## 3. Architecture

```
3GPP FTP archive (docx/YAML)
        |
        v
+----------------------------------------------------------+
| Ingestion (ingest/)                                       |
|  - chunk_docx.py  pandoc -> markdown -> heading-stack      |
|    parser. Hierarchical (title_path) + recursive           |
|    character split, ~1250 chars, 100-char overlap.         |
|  - chunk_yaml.py  OpenAPI specs, one chunk per              |
|    operation/schema, kept as raw YAML.                     |
|  - chunk_cr.py    Change Request cover-sheet + body         |
|    parser, tagged distinctly from ratified spec text.      |
+----------------------------------------------------------+
        |  JSONL chunks
        v
+----------------------------------------------------------+
| Embedding + Indexing (kaggle/build_qdrant_index.py)        |
|  - BGE-M3 dense (1024-dim) + BM25 sparse, on GPU            |
|  - Qdrant Cloud: one collection, dense+sparse named         |
|    vectors, keyword indexes on release/spec_id/             |
|    chunk_type/clause_number/cr_number                       |
+----------------------------------------------------------+
        |
        v
+----------------------------------------------------------+
| Retrieval (app/retrieval.py) - two-stage, per the paper's  |
| Algorithm 1                                                |
|  1. Pre-ranking: dense + BM25 prefetch (pool of 50),        |
|     fused server-side via Qdrant-native RRF                |
|  2. Reranking: BAAI/bge-reranker-v2-m3 cross-encoder        |
|     scores the pool, cut to top 8-10                       |
|  Release-aware: defaults to Rel-19; detects explicit        |
|  Rel-18 mentions or comparison intent and routes through    |
|  compare_search() - pairs the same clause across both       |
|  releases by exact filter, never blends a pooled search     |
+----------------------------------------------------------+
        |
        v
+----------------------------------------------------------+
| Generation (app/generation.py) - Groq, Llama-3.1-8B-Instant|
|  System prompt enforces: cite every claim, refuse on        |
|  insufficient context, compare-don't-blend, flag proposed   |
|  CR content as unratified, never fabricate a citation,      |
|  read ASN.1 range notation directly                        |
+----------------------------------------------------------+
        |
        v
+----------------------------------------------------------+
| Serving (gradio_app.py) - HF Spaces, Gradio 6, ZeroGPU      |
|  /respond    - chat UI                                     |
|  /answer_mcq - headless API for the eval harness            |
+----------------------------------------------------------+
```

### 3.1 Chunking

Matches the paper's two-stage design: a heading-stack walk assigns each
section a `clause_number`/`title`/`title_path`, then
`RecursiveCharacterTextSplitter` bounds oversized sections to ~1250
characters. One deliberate deviation: **100-character overlap** (the paper
uses 0) to avoid truncating facts at chunk boundaries. `title_path` is
prepended into a dedicated `embedding_text` field so the embedding vector
itself carries structural context, distinct from the clean `content` field
used for citation display.

### 3.2 Hybrid retrieval + reranking

Dense (BGE-M3) and sparse (BM25) candidates are pulled in parallel and
fused server-side via Qdrant's native RRF, then reranked by a
BGE-family cross-encoder. The paper's pre-ranking pool is Top-K₁=1000;
this system uses 50 — cross-encoder scoring runs on ZeroGPU at query time,
and 1000 pairs would make single-query latency impractical for an
interactive chat. That's a deliberate scoping tradeoff, documented rather
than hidden.

### 3.3 Change Requests: a differentiator, handled carefully

3GPP specs evolve through Change Requests (CRs) — proposed amendments that
may or may not yet be ratified. Four CR documents were added to the corpus
mid-build, and the ingestion (`chunk_cr.py`) parses their cover sheet
(target spec, CR number, category, release) separately from their proposed
clause text, tagging everything `chunk_type: change_request` /
`change_request_rationale`. Citations for this content read
`[PROPOSED CR 4696 — NOT ratified spec text — ...]`, and the system prompt
explicitly forbids presenting it as settled fact.

This surfaced a real correctness bug worth documenting: one CR's proposed
text turned out to be **byte-identical** to the ratified base spec once the
full spec was ingested — meaning the CR had already been incorporated, and
the "proposed, not ratified" framing was now factually wrong. This was
caught by diffing CR content against the base spec directly (not assumed),
and the stale CR was removed from the index. The other three CRs were
verified to still differ meaningfully from ratified text and remain
correctly flagged as pending.

### 3.4 Deployment reality: ZeroGPU

Hugging Face's free tier forces Gradio Spaces onto **ZeroGPU** hardware —
CPU Basic requires a PRO subscription, discovered only after the first
deploy attempt. This meant restructuring `app/embeddings.py`: per ZeroGPU's
documented requirement, BGE-M3 and the reranker load onto `cuda` at
**module level** (Space startup), not lazily, and only the actual GPU calls
are wrapped in `@spaces.GPU`. `requirements.txt` needed a standard
CUDA-capable `torch` build, not the CPU-only wheel originally planned.

---

## 4. Tools and software used

| Category | Tool | Why |
|---|---|---|
| Docx → text | `pandoc` 3.10.2 | Preserves heading levels + tables; the 2.9.2.1 system default OOM-killed on the largest specs and had to be upgraded |
| Chunking | `langchain-text-splitters` | `RecursiveCharacterTextSplitter` only |
| Dense embedding | `FlagEmbedding` (`BGEM3FlagModel`) | Real BGE-M3, matches the paper |
| Sparse embedding | `fastembed` (`Qdrant/bm25`) | Classic BM25, ONNX-based, no torch needed for this half |
| Reranking | `sentence-transformers` (`CrossEncoder`) | `BAAI/bge-reranker-v2-m3` — swapped from `FlagEmbedding.FlagReranker` after a real dependency conflict (see §5) |
| Vector store | Qdrant Cloud | Hybrid dense+sparse in one collection with native RRF fusion |
| Generation | Groq API, `llama-3.1-8b-instant` | Live successor to the paper's Llama-3-8B-Instruct |
| Serving | Gradio 6, HF Spaces (ZeroGPU) | Free-tier-only path for Gradio Spaces |
| Bulk embedding compute | Kaggle (free GPU) | One-time corpus embedding; incremental adds reuse the same script |
| Data source | 3GPP FTP archive | `ftp.3gpp.org` — master docx format, not PDF |

---

## 5. What actually went wrong (and why it matters for "understanding the design")

A few real engineering problems surfaced during the build, each with a
root cause traced before fixing — listed here because *how* they were
found and fixed is arguably better evidence of design understanding than
a clean success story would be:

- **Pandoc OOM on large specs** (24.501, then again on 38.331): the
  system pandoc build was outdated and later machine memory pressure
  required temporary swap. Fixed by upgrading pandoc and adding swap, not
  by silently skipping the documents.
- **`transformers`/`huggingface-hub` version conflict on the Space**:
  `FlagEmbedding`'s reranker calls a tokenizer method removed in
  `transformers` 5.x, but every `transformers` 4.x release requires
  `huggingface-hub<1.0`, which conflicts with `gradio` 6.x's
  `huggingface-hub>=1.0` — an unresolvable pin. Fixed by switching the
  reranker to `sentence-transformers`, not by fighting the pin.
- **A real fabricated citation**: a live query about EBI ranges got an
  answer citing a "proposed CR" that does not exist anywhere in the
  corpus. Traced precisely: the correct clause existed in the index but
  ranked 26th of 30 in BM25 pre-ranking — just outside the pool the
  reranker ever saw — and the model filled the resulting gap with an
  invented source instead of refusing. Fixed with two independent
  changes: a wider rerank pool, and an explicit system-prompt rule
  forbidding fabricated citations (verified via before/after A-B testing
  on the exact failing query).
- **Known remaining limitation**: two narrow numeric-fact queries still
  return "insufficient context" even though the answer exists in the
  corpus — confirmed by direct inspection of the stored chunks, not
  assumed. Both are retrieval-ranking misses (the answer-bearing chunk
  doesn't survive into the reranked top-k for that specific query
  phrasing), not coverage gaps, and both fail *safely* — refusal, not
  hallucination — which is the correct fallback behavior even when
  retrieval isn't perfect.

---

## 6. Evaluation

- **MCQ accuracy harness** (`eval/`): a 19-item TeleQnA-style multiple-choice
  set, hand-grounded in real retrieved chunk content (each question was
  checked against actual corpus text before being written, not generated
  from memory), spanning all 5 originally-ingested specs, both releases,
  and one explicit release-comparison item. Run via a dedicated
  `/answer_mcq` API endpoint (headless, separate from the chat UI).
  **Result: 94.7% (18/19).** The one miss is explainable: a comparison
  question routed through plain pooled retrieval rather than the paired
  `compare_search()` mechanism, confirming that mechanism is genuinely
  necessary rather than redundant engineering.

  This is *not* a like-for-like comparison with the papers' 78.7-80%
  TeleQnA figures — TeleQnA has 1,840 questions across a broader release
  set; this eval set has 19, built specifically against this corpus. It's
  reported as a scoped, honest internal accuracy check, not a benchmark
  claim.
- **Live qualitative testing**: beyond the fixed MCQ set, the deployed
  system was interrogated with real free-text questions during
  development, which is how the fabricated-citation bug (§5) was actually
  found — a benefit a static benchmark alone wouldn't have surfaced.
- **Not completed**: a RAGAS-style faithfulness/context-precision pass
  (LLM-judge scoring of free-text answers against retrieved context) was
  planned but not finished given the timeline — noted here as scoped-out,
  not hidden.

---

## 7. Repository structure

```
ingest/       chunk_docx.py, chunk_yaml.py, chunk_cr.py, build_manifest.py
kaggle/       build_qdrant_index.py — bulk + incremental embedding/upsert
app/          embeddings.py, retrieval.py, generation.py — the serving pipeline
gradio_app.py HF Spaces entry point (chat UI + /answer_mcq API)
eval/         questions.json, run_eval.py, results.json
data/         manifest.json (provenance); raw/extracted/processed are
              gitignored — regenerable, not needed at runtime
```
