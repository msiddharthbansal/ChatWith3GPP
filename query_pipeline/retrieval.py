import os
import re

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from query_pipeline.embeddings import embed_query, rerank
from query_pipeline.query_rewrite import rewrite_queries
from query_pipeline.context_packing import dedupe_by_clause

load_dotenv()

COLLECTION = "chat3gpp"
DEFAULT_RELEASE = "Rel-19"
GLOSSARY_SPEC_ID = "21.905"

REL_RE = re.compile(r"\brel(?:ease)?[\s-]?(1[6-9]|2[0-9])\b", re.IGNORECASE)
COMPARE_RE = re.compile(r"\b(compar\w*|differ\w*|changed?|change[sd]?|between releases?)\b", re.IGNORECASE)

_client = None


def client():
    global _client
    if _client is None:
        _client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
    return _client


def detect_releases(query: str):
    mentioned = sorted(set(f"Rel-{m}" for m in REL_RE.findall(query)))
    is_comparison = bool(COMPARE_RE.search(query)) or len(mentioned) >= 2
    if is_comparison:
        return (mentioned or ["Rel-18", "Rel-19"]), True
    if mentioned:
        return mentioned, False
    return [DEFAULT_RELEASE], False


def _payload_to_result(payload, score):
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
    if releases is None:
        releases, _ = detect_releases(query)

    release_filter = models.Filter(
        must=[models.FieldCondition(key="release", match=models.MatchAny(any=releases))],
        must_not=[models.FieldCondition(key="spec_id", match=models.MatchValue(value=GLOSSARY_SPEC_ID))],
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
    return dedupe_by_clause(ranked, max_per_clause)[:top_k]


def compare_search(query: str, top_k: int = 3, chunks_per_release: int = 2):
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

    unpaired = [c for c in candidates if not c["clause_number"]]

    return {"paired": paired, "unpaired": unpaired}
