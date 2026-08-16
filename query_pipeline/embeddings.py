import spaces

from FlagEmbedding import BGEM3FlagModel
from fastembed import SparseTextEmbedding
from sentence_transformers import CrossEncoder

_dense_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, devices="cuda")
_reranker_model = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda")
_sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")


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
    dense = _dense_encode(text)
    sparse = next(_sparse_model.embed([text]))
    return dense, sparse


def rerank(query: str, candidates: list[dict], top_k: int):
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
