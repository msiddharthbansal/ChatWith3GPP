"""
Hybrid (dense + BM25) retrieval against the Qdrant index built in
kaggle/build_qdrant_index.py, fused server-side via RRF, with release-aware
filtering: default to Rel-19 unless the query names a release explicitly or
reads as a cross-release comparison, in which case both releases are pulled
and paired by clause rather than blended (so the generator can compare
instead of averaging over two possibly-different clause texts).
"""
import os
import re

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from app.embeddings import embed_query, rerank
from app.query_rewrite import rewrite_queries

load_dotenv()

COLLECTION = "chat3gpp"
DEFAULT_RELEASE = "Rel-19"

REL_RE = re.compile(r"\brel(?:ease)?[\s-]?(1[6-9]|2[0-9])\b", re.IGNORECASE)
COMPARE_RE = re.compile(r"\b(compar\w*|differ\w*|changed?|change[sd]?|between releases?)\b", re.IGNORECASE)

_client = None


def client():
    global _client
    if _client is None:
        _client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
    return _client


def detect_releases(query: str):
    """Return (releases_to_search: list[str], is_comparison: bool)."""
    mentioned = sorted(set(f"Rel-{m}" for m in REL_RE.findall(query)))
    is_comparison = bool(COMPARE_RE.search(query)) or len(mentioned) >= 2
    if is_comparison:
        return (mentioned or ["Rel-18", "Rel-19"]), True
    if mentioned:
        return mentioned, False
    return [DEFAULT_RELEASE], False


def _payload_to_result(payload, score):
    # clause_number/title_path only exist on docx-derived chunks; cr_number/
    # category/current_version/clauses_affected only exist on Change Request
    # chunks. Index defensively rather than assuming every payload shares
    # one schema — this collection has four now (docx, yaml/API, CR text,
    # CR rationale).
    return {
        "score": score,
        "spec_id": payload["spec_id"],
        "release": payload["release"],
        "clause_number": payload.get("clause_number"),
        "title": payload["title"],
        "title_path": payload.get("title_path"),
        "content": payload["content"],
        "chunk_type": payload["chunk_type"],
        "source_file": payload["source_file"],
        "cr_number": payload.get("cr_number"),
        "category": payload.get("category"),
        "current_version": payload.get("current_version"),
        "clauses_affected": payload.get("clauses_affected"),
    }


def hybrid_search(
    query: str,
    top_k: int = 15,
    releases: list[str] | None = None,
    rerank_pool: int = 50,
    max_per_clause: int = 3,
):
    """
    Dense+BM25 pre-ranking (RRF-fused, Top-K1=rerank_pool) followed by a BGE-M3
    cross-encoder rerank down to Top-K2=top_k, matching the paper's two-stage
    pipeline (Algorithm 1). The paper's Top-K1 is 1000; rerank_pool is far
    smaller because cross-encoder scoring runs on ZeroGPU in the deployed Space
    at query time — 1000 pairs would make single-query latency impractical for
    an interactive chat, so this is a deliberate scoping tradeoff, not an
    oversight. Bumped 30->50 after a real miss: a query's correct definitional
    clause ranked 26/30 in BM25 alone, right at the pool's edge — 50 gives
    borderline-relevant candidates more room to survive into reranking, where
    the cross-encoder can actually judge relevance properly.

    Also runs an optional multi-query rewrite pass (app.query_rewrite,
    toggled via QUERY_REWRITE_ENABLED, fanned out via QUERY_REWRITE_COUNT)
    to close vocabulary gaps between conversational questions and formal
    spec phrasing that a bigger pool alone doesn't fix, since the miss is
    semantic/lexical distance, not pool-edge truncation. Confirmed
    load-bearing by direct measurement, not just theory: the same answer
    chunk ranked 17th under a colloquial phrasing and 1st under a
    spec-vocabulary phrasing of the identical question (see
    .claude/memory.md section 9) — one rewrite only gets one guess at the
    right vocabulary, so several diverse reformulations are searched and
    merged (deduped by Qdrant point id) alongside the original, so a bad
    rewrite can only add candidates, never remove ones the original query
    would have found.
    """
    if releases is None:
        releases, _ = detect_releases(query)

    release_filter = models.Filter(
        must=[models.FieldCondition(key="release", match=models.MatchAny(any=releases))]
    )

    queries = [query] + rewrite_queries(query)

    candidates_by_id = {}
    for q in queries:
        dense_vec, sparse_vec = embed_query(q)
        hits = client().query_points(
            COLLECTION,
            prefetch=[
                models.Prefetch(query=dense_vec, using="dense", filter=release_filter, limit=rerank_pool),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist()
                    ),
                    using="bm25",
                    filter=release_filter,
                    limit=rerank_pool,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=rerank_pool,
            with_payload=True,
        ).points
        for h in hits:
            if h.id not in candidates_by_id:
                candidates_by_id[h.id] = _payload_to_result(h.payload, h.score)

    all_candidates = list(candidates_by_id.values())
    ranked = rerank(query, all_candidates, top_k=len(all_candidates))
    return _dedupe_by_clause(ranked, max_per_clause)[:top_k]


def _dedupe_by_clause(ranked_results, max_per_clause):
    """Keep at most `max_per_clause` highest-scored chunks per (spec_id,
    clause_number). Recursive-split chunking means a large clause can
    produce many sub-chunks (one real clause here has 20) covering
    genuinely different sub-topics, not just redundant restatements — a
    max_per_clause=1 cap was tried first and actively hid the correct
    answer, which lived in a lower-scored sub-chunk of the same clause as a
    higher-scored-but-off-topic one (see .claude/memory.md section 9).
    max_per_clause=3 is a compromise: still frees real budget from clauses
    where multiple sub-chunks ARE near-duplicates (the original motivating
    case — same clause, same content, different chunk boundary), without
    assuming a clause only has one relevant angle. Chunks without a
    clause_number (unnumbered headings, YAML/CR chunks) can't be identified
    as duplicates of each other, so always kept."""
    clause_counts = {}
    deduped = []
    for c in ranked_results:
        key = (c["spec_id"], c.get("clause_number"))
        if key[1] is not None:
            count = clause_counts.get(key, 0)
            if count >= max_per_clause:
                continue
            clause_counts[key] = count + 1
        deduped.append(c)
    return deduped


def compare_search(query: str, top_k: int = 3, chunks_per_release: int = 2):
    """
    Cross-release comparison: search across both releases to find the
    best-matching clauses, then fetch each matched clause from BOTH releases
    by exact payload filter (not vector search) so the generator gets an
    aligned pair per clause instead of two independently-ranked lists.

    top_k/chunks_per_release are kept small (vs. hybrid_search's default)
    because the generator model (Groq llama-3.1-8b-instant, free tier) caps
    at 6,000 tokens/minute — a comparison context that pulls too many clauses
    or too many sub-chunks per clause blows that budget (HTTP 413).
    """
    candidates = hybrid_search(query, top_k=top_k, releases=["Rel-18", "Rel-19"])

    seen_keys = []
    for c in candidates:
        key = (c["spec_id"], c["clause_number"])
        if c["clause_number"] and key not in seen_keys:
            seen_keys.append(key)

    paired = {}
    for spec_id, clause_number in seen_keys:
        by_release = {}
        for release in ("Rel-18", "Rel-19"):
            pts, _ = client().scroll(
                COLLECTION,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="spec_id", match=models.MatchValue(value=spec_id)),
                        models.FieldCondition(key="release", match=models.MatchValue(value=release)),
                        models.FieldCondition(
                            key="clause_number", match=models.MatchValue(value=clause_number)
                        ),
                    ]
                ),
                limit=chunks_per_release,
                with_payload=True,
            )
            by_release[release] = [p.payload for p in pts]
        paired[f"{spec_id} {clause_number}"] = by_release

    # Any candidates without a clause_number (unnumbered subheadings) can't be
    # paired by exact match — surface them unpaired rather than dropping them.
    unpaired = [c for c in candidates if not c["clause_number"]]

    return {"paired": paired, "unpaired": unpaired}
