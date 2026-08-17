import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

from query_pipeline.embeddings import embed_query

load_dotenv()

COLLECTION = "chat3gpp"
GLOSSARY_SPEC_ID = "21.905"

_client = None


def client():
    global _client
    if _client is None:
        _client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
    return _client


def lookup_glossary(query: str, top_k: int = 5, pool: int = 20) -> list[str]:
    dense_vec, sparse_vec = embed_query(query)
    glossary_filter = models.Filter(
        must=[models.FieldCondition(key="spec_id", match=models.MatchValue(value=GLOSSARY_SPEC_ID))]
    )
    hits = client().query_points(
        COLLECTION,
        prefetch=[
            models.Prefetch(query=dense_vec, using="dense", filter=glossary_filter, limit=pool),
            models.Prefetch(
                query=models.SparseVector(
                    indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist()
                ),
                using="bm25",
                filter=glossary_filter,
                limit=pool,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    ).points
    return [h.payload["content"] for h in hits]
