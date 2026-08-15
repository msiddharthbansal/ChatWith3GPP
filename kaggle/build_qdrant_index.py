# Kaggle notebook: embed all chunks with real BGE-M3 (GPU) + BM25 sparse,
# upsert into the Qdrant Cloud collection. Paste each `# %%` block as a
# separate cell.
#
# Before running:
#   1. Upload data/processed/ (all 12 .jsonl files, both rel18 and rel19)
#      as a Kaggle Dataset, and attach it to this notebook.
#      Update CHUNKS_DIR below to match its mount path.
#   2. Add-ons > Secrets: add QDRANT_URL and QDRANT_API_KEY
#      (same values as the project's local .env).
#   3. Notebook settings: turn on GPU accelerator (T4 x1 is enough)
#      and turn on internet access.

# %%
!pip install -q FlagEmbedding qdrant-client fastembed

# %%
import json
import uuid
from pathlib import Path

from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
QDRANT_URL = secrets.get_secret("QDRANT_URL")
QDRANT_API_KEY = secrets.get_secret("QDRANT_API_KEY")

from qdrant_client import QdrantClient, models

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
print(client.get_collections())

# %%
CHUNKS_DIR = Path("/kaggle/input/chat3gpp-chunks")  # <-- update to your dataset's mount path

files = sorted(CHUNKS_DIR.rglob("*.jsonl"))
print(f"found {len(files)} chunk files")

records = []
for fp in files:
    with open(fp) as f:
        for line in f:
            records.append(json.loads(line))
print(f"total chunks: {len(records)}")

# %%
def text_for_embedding(r):
    # docx-derived chunks already carry title_path-prefixed text;
    # yaml/API chunks don't, so fall back to title + content.
    if r.get("embedding_text"):
        return r["embedding_text"]
    heading = r.get("title") or r.get("service") or ""
    return f"{heading}\n\n{r['content']}"


def point_id(r):
    key = "|".join(
        str(r.get(k, ""))
        for k in (
            "spec_id", "release", "source_file", "clause_number",
            "chunk_index", "operation_path", "http_method", "title",
        )
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


texts = [text_for_embedding(r) for r in records]
ids = [point_id(r) for r in records]
assert len(set(ids)) == len(ids), "duplicate point IDs — check the key fields above"

# %%
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, devices="cuda")

dense_vecs = model.encode(
    texts,
    batch_size=64,
    max_length=1024,       # chunks are capped at 1250 chars, this leaves headroom
    return_dense=True,
    return_sparse=False,
    return_colbert_vecs=False,
)["dense_vecs"]
print("dense shape:", dense_vecs.shape)

# %%
from fastembed import SparseTextEmbedding

# classic BM25 (matches the Chat3GPP paper's sparse side), CPU is fine for this
bm25 = SparseTextEmbedding(model_name="Qdrant/bm25")
sparse_vecs = list(bm25.embed(texts, batch_size=64))
print("sparse vectors:", len(sparse_vecs))

# %%
COLLECTION = "chat3gpp"

if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": models.VectorParams(
                size=dense_vecs.shape[1], distance=models.Distance.COSINE
            ),
        },
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
    )
    for field in ("release", "spec_id", "chunk_type", "clause_number"):
        client.create_payload_index(COLLECTION, field_name=field, field_schema="keyword")

print(client.get_collection(COLLECTION))

# %%
from tqdm import tqdm

BATCH = 128
for i in tqdm(range(0, len(records), BATCH)):
    batch_ids = ids[i : i + BATCH]
    batch_records = records[i : i + BATCH]
    batch_dense = dense_vecs[i : i + BATCH]
    batch_sparse = sparse_vecs[i : i + BATCH]

    points = [
        models.PointStruct(
            id=rid,
            vector={
                "dense": dvec.tolist(),
                "bm25": models.SparseVector(
                    indices=svec.indices.tolist(), values=svec.values.tolist()
                ),
            },
            payload=rec,
        )
        for rid, rec, dvec, svec in zip(batch_ids, batch_records, batch_dense, batch_sparse)
    ]
    client.upsert(collection_name=COLLECTION, points=points)

print("done:", client.count(COLLECTION))
