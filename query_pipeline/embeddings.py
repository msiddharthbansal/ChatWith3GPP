"""
Query-time embedding + reranking — must match the models used to build the
Qdrant index in kaggle/build_qdrant_index.py exactly (dense: BAAI/bge-m3,
sparse: Qdrant/bm25) and the paper's reranker (BAAI/bge-reranker-v2-m3), or
scores against the indexed vectors are meaningless.

Runs on HF Spaces ZeroGPU (the free tier's only option for Gradio Spaces —
CPU Basic requires a PRO subscription). Per ZeroGPU's documented requirement,
models are loaded onto 'cuda' at module level (Space startup) rather than
lazily inside the @spaces.GPU functions — CUDA placement at startup runs
under an emulation shim even with no physical GPU attached yet; the real GPU
is only attached for the duration of an @spaces.GPU-decorated call.
"""
import spaces  # must be imported before torch is imported anywhere in the process

from FlagEmbedding import BGEM3FlagModel
from fastembed import SparseTextEmbedding
from sentence_transformers import CrossEncoder

_dense_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, devices="cuda")
# Reranker loaded via sentence-transformers, not FlagEmbedding.FlagReranker —
# see the note in requirements.txt for why (FlagEmbedding's reranker breaks
# on the transformers version gradio 6.x requires).
_reranker_model = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda")
_sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")  # CPU-only, no GPU needed


@spaces.GPU
def _dense_encode(text: str) -> list[float]:
    return _dense_model.encode(
        [text],
        max_length=1024,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )["dense_vecs"][0].tolist()


@spaces.GPU
def _cross_encoder_scores(pairs: list[list[str]]) -> list[float]:
    return _reranker_model.predict(pairs).tolist()


def embed_query(text: str):
    """Return (dense_vector: list[float], sparse_vector: fastembed.SparseEmbedding)."""
    dense = _dense_encode(text)
    sparse = next(_sparse_model.embed([text]))
    return dense, sparse


def rerank(query: str, candidates: list[dict], top_k: int):
    """Cross-encoder rerank: jointly score each (query, chunk) pair and return
    the top_k candidates, highest-scoring first. Overwrites each dict's
    'score' with the reranker's score so callers see the final ranking basis."""
    if not candidates:
        return candidates
    pairs = [[query, c["content"]] for c in candidates]
    scores = _cross_encoder_scores(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    top = []
    for c, score in ranked[:top_k]:
        c = dict(c)
        c["score"] = score
        top.append(c)
    return top
