# PROJECT_MEMORY.md
**Project:** Chat3GPP-style RAG Chatbot — 3GPP Standards, Near-Zero Hallucination
**Context:** Take-home submission for a Graduate Engineer Trainee (GET) role.
**Deadline:** 17 August (submission), followed by technical interview.

---

## 1. The assignment

Build a working RAG chatbot over Telecom 3GPP standards documentation,
explicitly optimized for minimal-to-near-zero hallucination. Evaluation
criteria: quality/effectiveness of the project, understanding of the design
and implementation, and interview performance.

## 2. Reference research (cite these in the writeup)

- **Chat3GPP** (Huang et al., Tsinghua, arXiv:2501.13954, Jan 2025) — open-source
  RAG framework for 3GPP docs. This is the primary architectural template
  we're following. Key results: 78.7% accuracy on TeleQnA (Rel-17/18) without
  any fine-tuning, beating fine-tuned baselines (Telco-RAG, TelecomGPT,
  Llama-3-8B-Tele-it). Repo: github.com/huangl22/Chat3GPP.
  - Their pipeline: hierarchical (heading-based) chunking + recursive
    character splitting (~1250 chars, no overlap, LangChain's
    `RecursiveCharacterTextSplitter`) → BGE-M3 embeddings → hybrid retrieval
    (BM25 + dense, fused via Reciprocal Rank Fusion) → BGE-M3 cross-encoder
    rerank → prompted generation with an explicit "insufficient context"
    refusal instruction.
  - Their acknowledged limitations (our chance to differentiate): tables/
    figures excluded from ingestion; no faithfulness/entailment verification
    beyond the refusal prompt.
  - **Verified against the actual repo source (2026-08-15, not just the
    paper text) — `init_database.py`, `retrievers/VectorRetriever.py`,
    `text_splitter/TSdocx_splitter.py`, `server/llm.py`:**
    - `TSDocTextSplitter` confirms `RecursiveCharacterTextSplitter(chunk_size=1250,
      chunk_overlap=0)` — matches what we built. It also prepends the full
      ancestor-heading path to each chunk before embedding (their
      `concatenate_heading_content`) — exactly what our `embedding_text`
      field (`title_path + content`) does; independent confirmation we got
      that fix right. (Note: `configs/model_configs.py` has a
      `SPLITTER_CONFIG = {"chunk_size": 256}` that looks contradictory but
      is dead/unused — the real splitter hardcodes 1250.)
    - Single Elasticsearch index does double duty: a `dense_vector` field
      (1024-dim, cosine) for BGE-M3 embeddings + a `content` text field
      (custom analyzer: standard tokenizer, lowercase, stopword filter)
      for BM25 `match` queries. We're substituting Chroma/FAISS (dense) +
      `rank_bm25` (keyword) — logged scope tradeoff, see §8.
    - **RRF fusion, exact formula from `VectorRetriever.calculate_rrf`:**
      `score = 1/(k+bm25_rank) + 1/(k+vector_rank)`, **k=10** (not the more
      common k=60 — worth either matching for fidelity or explicitly
      flagging as a deliberate deviation). Retrieve top-1000 from each
      side (`top_k1=1000`), fuse, keep the top 10% of `top_k1` (~100
      candidates) by RRF score.
    - **Final rerank is NOT a separate cross-encoder** despite
      `RERANKER_CONFIG` naming `bge-reranker-large` — the actual
      `search_rrf()` reranks the ~100 RRF survivors using **BGE-M3's own
      ColBERT multi-vector late-interaction scoring**
      (`embed_model.encode(..., return_colbert_vecs=True)` +
      `embed_model.colbert_score(...)`), cutting to `top_k2=5` final docs.
      The `bge-reranker-large` config value appears to be unused/vestigial.
    - Their refusal prompt (`server/llm.py`, `generate_multiple_choice_prompt`):
      *"If the context does not provide sufficient information, respond
      with 'Insufficient context to answer.'"* — good phrasing template,
      but note the **public repo is MCQ-benchmark-only** (`chat.py`/
      `kb_chat.py` both load a fixed TeleQnA-style question JSON; there is
      no open-ended chat interface in the repo). Our generation prompt for
      free-form Q&A is our own design, only informed by their refusal
      phrasing — not adapted from an existing open-ended template, since
      none exists in their code.
- **Telco-RAG** (Bornea et al., arXiv:2404.15939) — earlier telecom RAG
  baseline, fine-tuned an NN router; Chat3GPP beats it without fine-tuning.

## 3. Scope decisions

**Releases:** Rel-18 and Rel-19 (both frozen — Rel-19 froze Dec 2025, so no
moving-target risk). Rel-16–19/all-series was considered and explicitly
rejected as out of scope for a 2-day build; framed as "production roadmap"
in the writeup instead. Reasoning logged: multi-release ingestion without
release-aware metadata is a *hallucination risk*, not just a scope/time
problem — same clause number can mean different things across releases.

**Specs (5 total, both releases = 10 sources):**

| Spec | Covers | Format |
|---|---|---|
| TS 23.501 | 5G System architecture | .docx |
| TS 23.502 | 5GS procedures | .docx |
| TS 24.501 | NAS protocol for 5GS | .docx |
| TS 33.501 | 5G security architecture | .docx |
| TS 29.518 | AMF services (Stage 3, service-based interfaces) | .docx **+ OpenAPI YAML** (Namf_Communication, Namf_EventExposure, Namf_Location, Namf_MBSBroadcast, Namf_MT, etc.) |

Optional stretch additions (only after the core 10 are working):
TR 21.918/21.919 (Rel-18/19 "what's new" summaries), TS 29.500 (base SBI
framework that 29.518 references).

## 4. Where the source files live / how to get more

3GPP FTP archive pattern:
```
https://www.3gpp.org/ftp/Specs/archive/<series>_series/<spec_number>/
```
e.g. `23_series/23.501/`, `29_series/29.518/`.

**Version filename decoding** (critical, and now automated in
`build_manifest.py`): filenames are `<specnum>-<versioncode>.zip`, e.g.
`23501-ic0.zip`. First character of the version code maps release number
via 0-9 then a-z = 10-35 (a=10, ... h=17=Rel-17, **i=18=Rel-18**,
**j=19=Rel-19**, k=20=Rel-20-in-progress). Always take the *highest*
letter-matching file in a folder for the final version of that release.

Confirmed real examples from this project:
- `23501-ic0.zip` → Rel-18, v18.12.0
- `23501-j80.zip` → Rel-19, v19.8.0
- `29518-ie0.zip` → Rel-18, v18.14.0 (has a `29518-ie0/` folder of YAMLs alongside the .docx)

## 5. Local folder structure

```
data/
├── raw/rel18/*.zip, raw/rel19/*.zip        (untouched originals, gitignored)
├── extracted/rel18/, extracted/rel19/       (.docx + YAML folders, gitignored)
├── processed/rel18/, processed/rel19/       (chunked .jsonl — safe to commit)
└── manifest.json                            (spec/release/version provenance)
```
`.docx` and YAML-folder names keep the zip's stem (e.g. `23501-ic0.docx`,
`29518-ie0/`) so the manifest can match everything up automatically.

## 6. Environment

- Python venv (`python3 -m venv venv`).
- `pandoc` required on PATH (system binary, not pip) — used to convert
  `.docx` → markdown while preserving heading levels and tables.
  **Upgraded system-wide to pandoc 3.10.2** (from Ubuntu jammy's stock
  2.9.2.1, via the official .deb from GitHub releases) — the old version
  OOM-killed (6.7GB RSS) converting `24501-id0.docx`/`24501-j70.docx`
  (TS 24.501, NAS protocol — heavy nested/merged-cell IE tables), a known
  pandoc docx-table perf bug fixed in later releases.
- `requirements.txt`: `pyyaml`, `langchain-text-splitters`, `langchain`.
  **Important LangChain packaging note:** as of LangChain v1.x, text
  splitters were split out of the core `langchain` package entirely —
  `RecursiveCharacterTextSplitter` only lives in the separate
  `langchain-text-splitters` package, even with full `langchain` installed.
  Full `langchain` is installed anyway, for the upcoming generation-phase
  prompt templates / output parsers (`langchain_core.prompts`, etc.).

## 7. Ingestion pipeline — status: BUILT, TESTED, AND RUN ACROSS ALL 10 SOURCES

Final chunk counts (`chunk_docx.py`, `--max-chars 1250 --overlap 10`):

| Spec | Rel-18 | Rel-19 |
|---|---|---|
| 23.501 | 3,581 | 4,004 |
| 23.502 | 3,763 | 4,050 |
| 24.501 | 9,486 | 10,848 |
| 29.518 | 4,017 | 4,472 |
| 33.501 | 997 | 1,073 |

**Two real bugs found and fixed during the pandoc 3.10.2 upgrade (both in
`parse_heading()`/`SKIP_HEADING_TITLES` in `chunk_docx.py`):**
1. Pandoc 3.x's markdown writer auto-appends `{#id .class}` attribute
   blocks to headings mapped from non-numbered/custom Word styles (e.g.
   `"Contents {#contents .TT}"`), which broke the exact-match boilerplate
   skip and let the Table of Contents leak into chunks. Fixed by stripping
   the trailing `{#...}` block in `parse_heading()` before title parsing.
2. Pandoc renders Annex-title headings (deep Word heading styles) as 7-8
   `#` characters, which the old `#{1,6}` regex didn't recognize as
   headings at all — so trailing Annex sections (critically, the
   **"Change history"** CR-tracking table every 3GPP spec ends with,
   often 1000+ rows) got silently absorbed as body text into whichever
   real clause preceded them, then shredded into thousands of
   near-meaningless chunks (this alone was 63% of TS 24.501's chunk count
   before the fix). Fixed by widening the regex to `#{1,}` (unlimited
   heading depth) and adding an `ANNEX_TITLE_RE`-aware skip so
   `"Annex F (informative): Change history"` is dropped like
   Foreword/Contents — CR bookkeeping isn't queryable spec content and
   was judged out of scope for the near-zero-hallucination Q&A goal.

**Known pre-existing gap (not yet fixed, lower severity):** `CLAUSE_NUM_RE`
only matches a single leading letter directly followed by digits (e.g.
`"A1"`), not 3GPP's letter-suffixed (`"9.11.3.18C"`) or Annex-dotted
(`"D.8.8"`) clause number formats. Chunks under these headings get
`clause_number: null` with the number folded into the (otherwise correct)
`title`/`content` — a metadata-completeness gap, not a content-loss bug.



Three scripts, each usable as a CLI *or* an importable function set (no
orchestrator script — every invocation is explicit and runs standalone):

- **`build_manifest.py`** — `parse_zip_filename()`, `build_manifest()`,
  `save_manifest()`. Decodes release/version from zip filenames.
- **`chunk_docx.py`** — `docx_to_markdown()` (pandoc), `split_into_sections()`
  (heading-stack walk → clause_number/title/title_path per section, skips
  boilerplate like Foreword/Contents), `recursive_split()` (LangChain
  `RecursiveCharacterTextSplitter`, applied *within* each clause section as
  a size-bounding second pass — two-stage design matching Chat3GPP),
  `chunk_document()` (ties it together, tags table vs text chunks via
  regex detection of pipe-tables and pandoc's ASCII grid-table dashes),
  `write_chunks_jsonl()`.
  - CLI requires explicit `--max-chars` (no silent default) and
    `--overlap` (default 0, matching the paper). Actual corpus run with
    `--max-chars 1250 --overlap 10` — a smaller overlap than originally
    floated (100), still a deliberate nonzero improvement over the paper's
    zero-overlap choice to avoid boundary-truncation of facts.
- **`chunk_yaml.py`** — `chunk_yaml_file()`, `chunk_yaml_directory()`,
  `write_chunks_jsonl()`. Chunks each `Namf_*.yaml` OpenAPI file into one
  chunk per path+HTTP-method operation and one per schema definition,
  keeping raw YAML structure (not flattened to prose) since exact field
  names/types matter for grounding on API questions. This is flagged as a
  genuine differentiator vs. the Chat3GPP paper, which ingested prose only.

**Chunk schema (docx):**
```json
{"spec_id": "23.501", "release": "Rel-18", "version": "18.12.0",
 "clause_number": "5.15.3", "title": "Network Slicing",
 "title_path": "Network functions > Network Slicing", "chunk_index": 0,
 "chunk_type": "text|table", "content": "...",
 "embedding_text": "Network functions > Network Slicing\n\n...",
 "source_file": "23501-ic0.docx"}
```
`embedding_text` (`title_path` + `"\n\n"` + `content`) is what should be fed
to the embedding model — `content` stays clean for citation display. This
was the previously-open item; now applied.
**Chunk schema (yaml):**
```json
{"spec_id": "29.518", "release": "Rel-18", "version": "18.14.0",
 "service": "Namf_Communication", "section_type": "operation|schema",
 "operation_path": "...", "http_method": "POST", "title": "...",
 "content": "<raw yaml>", "chunk_type": "openapi_yaml", "source_file": "..."}
```

**Known open item (metadata-completeness, not content-loss):**
`CLAUSE_NUM_RE` doesn't parse letter-suffixed (`"9.11.3.18C"`) or
Annex-dotted (`"D.8.8"`) clause numbers — those chunks get
`clause_number: null` with the number folded into `title`/`content`
instead. Fine to defer; revisit if per-clause citation lookups need exact
clause-number matches for these headings specifically.

## 8. Architecture plan — status: NOT YET BUILT

- **Embedding + indexing:** BGE-M3 (open-source, matches paper) → Chroma or
  FAISS (chosen over the paper's Elasticsearch specifically to avoid infra
  setup risk in the 2-day window — documented as a deliberate scoping
  tradeoff, not an oversight) + a keyword index (`rank_bm25`) for hybrid
  search, fused via Reciprocal Rank Fusion. Cross-encoder rerank (BGE-M3)
  on top-k before generation.
- **Release-aware retrieval:** default to Rel-19 (latest) unless the query
  references Rel-18 or asks for a cross-release comparison. Comparison
  queries ("what changed between Rel-18 and Rel-19") should retrieve the
  *same clause* from both releases and pass both to the generator with an
  explicit compare-don't-blend instruction — flagged as the strongest
  hallucination-mitigation demo query for the interview.
- **Generation:** low temperature (0–0.2), mandatory clause citation,
  explicit "not found in the provided specifications" refusal path (mirrors
  Chat3GPP's MCQ prompt's "Insufficient context to answer"), plus a planned
  faithfulness/entailment verification pass — this is the concrete
  improvement over the paper's refusal-prompt-only approach.
- **Evaluation:** build a small (15-20 item) TeleQnA-style MCQ test set from
  the ingested specs, report accuracy the same way the paper does, plus a
  RAGAS faithfulness/context-precision score the paper didn't report.

### Alternatives considered (for the writeup's scope-tradeoff framing)

| Decision point | Chosen | Alternatives considered | Why not chosen |
|---|---|---|---|
| Vector store | Chroma or FAISS | **Elasticsearch** (paper's actual choice — one index does double duty for dense + BM25 via a custom analyzer) | Needs a running ES cluster/daemon; standing one up (or Docker) is pure infra risk for a 2-day build with no payoff in retrieval quality. Framed in the writeup as "production roadmap," not an oversight — Chat3GPP's own results don't depend on ES specifically, just on hybrid retrieval existing. |
| | | **Qdrant / Weaviate / pgvector** | Same infra-standup cost as ES for no fidelity benefit — we're not trying to demo a specific vector DB, just hybrid retrieval. Chroma is embedded (no server), FAISS is a pure library — both run in-process with zero ops overhead. |
| BM25 implementation | `rank_bm25` (Python, in-process) | **Elasticsearch's built-in `match` query** (paper's choice, uses a custom analyzer: standard tokenizer + lowercase + stopword filter) | Tied to the ES decision above — dropping ES means dropping its BM25 too. `rank_bm25` reproduces the same ranking algorithm without the server dependency; the custom stopword-filter analyzer is a minor fidelity gap, noted but judged low-impact on a 10-document corpus. |
| RRF fusion constant | k=10 (matching paper exactly, confirmed from `VectorRetriever.calculate_rrf` source) | **k=60** (the more common default in RRF literature/most hybrid-search implementations, e.g. Elastic's own docs) | Deliberately kept at the paper's actual value rather than the "textbook" default — this is a reproduction of their specific method, and their reported TeleQnA accuracy was achieved with k=10, not k=60. Diverging here would weaken the "we followed the validated recipe" claim for the interview. |
| Rerank stage | BGE-M3 ColBERT late-interaction scoring (`encode(..., return_colbert_vecs=True)` + `colbert_score`) | **A separate cross-encoder model** (e.g. `bge-reranker-large`, which is what `RERANKER_CONFIG` in their repo names, and what the paper's prose implies) | The repo's actual executed code path (`search_rrf()`) never calls a cross-encoder — it reranks with BGE-M3's own multi-vector ColBERT scoring instead. `bge-reranker-large` looks vestigial/unused in their code. We're matching what the code *does*, not what the config *suggests*, since that's what produced their reported accuracy. Also avoids loading a second large model. |
| | | **No rerank stage (RRF output used directly)** | Paper's ablations (and general hybrid-RAG literature) show reranking meaningfully improves top-k precision; skipping it would be a real fidelity/quality regression, not just a scope simplification. |
| Embedding model | BGE-M3 (matches paper) | **OpenAI `text-embedding-3-*` / Cohere embed** | Paid API dependency + adds an external-service failure mode; BGE-M3 is open-source, runs locally, and is the exact model the paper's reported numbers are based on — using anything else would make our results non-comparable to their 78.7% TeleQnA baseline. |
| | | **`bge-large-en`** (also listed in the repo's own `MODEL_PATH` config, English-only) | 3GPP specs are English-only so this would technically work, but BGE-M3 additionally supports the dense+sparse+ColBERT multi-representation output needed for the ColBERT rerank stage above — switching models would mean re-deriving or dropping that rerank step. |

## 9. Next steps (in order)

1. ~~Run the ingestion CLI across all 10 docx sources (5 specs × 2 releases).~~
   DONE (2026-08-15) — see §7 for final chunk counts and the two pandoc
   bugs found/fixed along the way.
2. ~~Sanity-check output~~ DONE for docx output (clause numbers, table
   tagging, Rel-18/Rel-19 parity all checked).
3. ~~Apply the title_path-into-content fix~~ DONE — `embedding_text` field
   added to the docx chunk schema (§7).
4. ~~Run `chunk_yaml.py` over the 29.518 OpenAPI YAML folders~~ DONE
   (2026-08-15) — `data/processed/{rel18,rel19}/29.518_api.jsonl`, 238
   chunks (Rel-18: 29 operations, 209 schemas) and 265 chunks (Rel-19: same
   6 services plus a new `Namf_AIoT` service — 5G AIoT is new in Rel-19).
   Clean run, no parsing errors, no fixes needed.
5. Write embedding + hybrid indexing script (BGE-M3 + Chroma/FAISS + BM25 + RRF).
6. Write the grounded generation prompt + refusal + release-comparison logic.
7. Build the small evaluation set + RAGAS harness.
8. Write the architecture doc / submission writeup, explicitly citing
   Chat3GPP as prior art and framing scope decisions (release/series
   narrowing, Elasticsearch→Chroma swap, full Rel-16-19 as future work) as
   deliberate engineering tradeoffs.
