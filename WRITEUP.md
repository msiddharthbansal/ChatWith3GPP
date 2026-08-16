# Design writeup

Full architecture notes, engineering decisions, bugs found and fixed, and
evaluation methodology — the detail behind the summary in [README.md](README.md).

---

## 1. Scope

- **Releases: Rel-18 and Rel-19 only.** Both are frozen, so no moving
  target during development. Rel-16/17 and all-series coverage were
  considered and cut — a time-budget call, not an oversight. Multi-release
  ingestion without release tagging is also a hallucination risk: the same
  clause number can mean different things across releases.
- **Corpus: 15 full base specs** — 5G core architecture, NAS, security,
  RAN, positioning, and service-based interfaces — plus a few Change
  Request (CR) documents layered on top (§3.3). Started at 5 specs, grew
  to 15 after live testing found real gaps (§5).

---

## 2. Prior art

Follows **[Chat3GPP](https://arxiv.org/abs/2501.13954)** (Huang et al.,
2025): hierarchical + recursive chunking → BGE-M3 dense + BM25 sparse
hybrid retrieval (RRF fusion) → cross-encoder reranking → prompted
generation with a refusal instruction. Chat3GPP reports 78.7% on TeleQnA
(Rel-17/18), zero-shot, beating fine-tuned baselines including **Telco-RAG**
(Bornea et al., arXiv:2404.15939). Telco-RAG compensates for weaker
retrieval — dense-only, no reranking, 125-token chunks — with a trained
router that predicts which of 18 3GPP series to search. Chat3GPP skips the
training entirely and just does hybrid retrieval + reranking properly.
Simpler, no closed-model dependency (Telco-RAG uses OpenAI embeddings +
GPT-3.5), and it wins zero-shot — that's why it's the template here.

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

A heading-stack walk assigns each section a `clause_number`, `title`, and
`title_path`. `RecursiveCharacterTextSplitter` then bounds oversized
sections to ~1250 characters. One change from the paper: 100-character
overlap instead of 0, to stop facts getting cut at chunk boundaries.
`title_path` is prepended into a separate `embedding_text` field, so the
embedding carries structural context while `content` stays clean for
citation display.

### 3.2 Hybrid retrieval + reranking

Dense (BGE-M3) and sparse (BM25) candidates are pulled in parallel and
fused server-side with Qdrant's native RRF, then reranked by a BGE-family
cross-encoder. The paper's pre-ranking pool is 1000 candidates; this uses
50 — reranking runs on ZeroGPU at query time, and 1000 pairs would make a
single query too slow for a chat interface. A scoping tradeoff, not an
oversight.

### 3.3 Change Requests

3GPP specs evolve through Change Requests (CRs) — proposed amendments that
may or may not be ratified yet. Four CR documents were added mid-build.
`chunk_cr.py` parses the cover sheet (target spec, CR number, category,
release) separately from the proposed clause text, and tags everything
`chunk_type: change_request` / `change_request_rationale`. Citations read
`[PROPOSED CR 4696 — NOT ratified spec text — ...]`, and the system prompt
is told never to present this as settled fact.

This caught a real bug: one CR's proposed text turned out to be
byte-identical to the ratified base spec once the full spec was ingested —
the CR had already been incorporated, so "proposed, not ratified" was now
wrong. Found by diffing the CR text against the base spec directly, not
assumed. That CR was removed from the index. The other three were checked
the same way and still differ from ratified text, so they stay flagged as
pending.

### 3.4 Deployment: ZeroGPU

HF's free tier forces Gradio Spaces onto ZeroGPU hardware — CPU Basic
needs a PRO subscription, found out after the first deploy failed. Fixed
by restructuring `app/embeddings.py`: BGE-M3 and the reranker load onto
`cuda` at module level (Space startup), per ZeroGPU's required pattern,
and only the actual GPU calls are wrapped in `@spaces.GPU`.
`requirements.txt` needed a normal CUDA-capable `torch` build, not the
CPU-only wheel originally planned.

---

## 4. Tools

| Category | Tool | Why |
|---|---|---|
| Docx → text | `pandoc` 3.10.2 | Keeps heading levels + tables; the system default (2.9.2.1) OOM-killed on the largest specs |
| Chunking | `langchain-text-splitters` | `RecursiveCharacterTextSplitter` only |
| Dense embedding | `FlagEmbedding` (`BGEM3FlagModel`) | Real BGE-M3, matches the paper |
| Sparse embedding | `fastembed` (`Qdrant/bm25`) | Classic BM25, ONNX-based |
| Reranking | `sentence-transformers` (`CrossEncoder`) | `BAAI/bge-reranker-v2-m3` — swapped in after a dependency conflict with `FlagEmbedding.FlagReranker` (§5) |
| Vector store | Qdrant Cloud | Hybrid dense+sparse in one collection, native RRF |
| Generation | Groq API, `llama-3.1-8b-instant` | Live successor to the paper's Llama-3-8B-Instruct |
| Serving | Gradio 6, HF Spaces (ZeroGPU) | Free-tier path for Gradio Spaces |
| Bulk embedding | Kaggle (free GPU) | One-time corpus embedding; incremental adds reuse the same script |
| Data source | 3GPP FTP archive | Master docx format, not PDF |

---

## 5. What broke, and how it was found

- **Pandoc OOM** on 24.501, then again on 38.331: old pandoc build, then
  memory pressure needing temporary swap. Fixed by upgrading pandoc and
  adding swap — not by skipping the documents.
- **`transformers`/`huggingface-hub` conflict on the Space**:
  `FlagEmbedding`'s reranker calls a tokenizer method removed in
  `transformers` 5.x. But every `transformers` 4.x release needs
  `huggingface-hub<1.0`, and `gradio` 6.x needs `huggingface-hub>=1.0` —
  no version of `transformers` satisfies both. Fixed by switching the
  reranker to `sentence-transformers` instead of fighting the pin.
- **A fabricated citation**: a live query about EBI ranges got an answer
  citing a "proposed CR" that doesn't exist anywhere in the corpus. Traced
  it: the correct clause was in the index but ranked 26th of 30 in BM25
  pre-ranking, just outside the pool the reranker saw. The model filled
  the gap by inventing a source instead of refusing. Fixed two ways: a
  wider rerank pool, and a system-prompt rule against fabricated
  citations. Verified by re-running the same failing query before and
  after.
- **Still open**: two narrow numeric questions still return "insufficient
  context" even though the answer is in the corpus — confirmed by
  checking the stored chunks directly. Both are retrieval-ranking misses,
  not missing data, and both fail safely (refusal, not hallucination).

---

## 6. Evaluation

- **MCQ accuracy** (`eval/`): 19 TeleQnA-style multiple-choice questions,
  each checked against real retrieved chunk text before being written —
  not generated from memory. Covers all 5 original specs, both releases,
  and one release-comparison question. Run through a dedicated
  `/answer_mcq` API endpoint, separate from the chat UI.
  **Result: 94.7% (18/19).** The one miss: a comparison question routed
  through plain pooled retrieval instead of the paired `compare_search()`
  path, which confirms that path is needed, not redundant.

  Not a direct comparison to the papers' 78.7-80% on TeleQnA — that
  benchmark has 1,840 questions across a wider release set; this one has
  19, built against this specific corpus. Reported as an honest internal
  check, not a benchmark claim.
- **Live testing**: real free-text questions during development is how
  the fabricated-citation bug (§5) actually got found — something a fixed
  MCQ set wouldn't have caught.
- **Not done**: a RAGAS-style faithfulness/context-precision pass was
  planned but not finished in the time available. Scoped out, not hidden.
