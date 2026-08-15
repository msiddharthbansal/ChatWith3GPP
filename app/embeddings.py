"""
Query-time embedding — must match the models used to build the Qdrant index
in kaggle/build_qdrant_index.py exactly (dense: BAAI/bge-m3, sparse: Qdrant/bm25),
or cosine similarity against the indexed vectors is meaningless.

Runs on CPU: this is single-query inference at request time, not bulk corpus
embedding (that happened once, on GPU, in the Kaggle notebook).
"""
from functools import lru_cache


@lru_cache(maxsize=1)
def _dense_model():
    from FlagEmbedding import BGEM3FlagModel
    return BGEM3FlagModel("BAAI/bge-m3", use_fp16=False, devices="cpu")


@lru_cache(maxsize=1)
def _sparse_model():
    from fastembed import SparseTextEmbedding
    return SparseTextEmbedding(model_name="Qdrant/bm25")


@lru_cache(maxsize=1)
def _reranker_model():
    # The paper's "BGE-M3 cross-encoder" reranking stage is BAAI/bge-reranker-v2-m3
    # — BGE-M3's own encode() only produces embeddings, not pairwise scores; the
    # dedicated cross-encoder reranker built on the same backbone is this model.
    from FlagEmbedding import FlagReranker
    return FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False, devices=["cpu"])


def embed_query(text: str):
    """Return (dense_vector: list[float], sparse_vector: fastembed.SparseEmbedding)."""
    dense = _dense_model().encode(
        [text],
        max_length=1024,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )["dense_vecs"][0]
    sparse = next(_sparse_model().embed([text]))
    return dense.tolist(), sparse


def rerank(query: str, candidates: list[dict], top_k: int):
    """Cross-encoder rerank: jointly score each (query, chunk) pair and return
    the top_k candidates, highest-scoring first. Overwrites each dict's
    'score' with the reranker's score so callers see the final ranking basis."""
    if not candidates:
        return candidates
    pairs = [[query, c["content"]] for c in candidates]
    scores = _reranker_model().compute_score(pairs, normalize=True)
    if isinstance(scores, float):
        scores = [scores]
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    top = []
    for c, score in ranked[:top_k]:
        c = dict(c)
        c["score"] = score
        top.append(c)
    return top
