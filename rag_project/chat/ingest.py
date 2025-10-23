from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer


qdrant = QdrantClient(host="localhost", port=6333)
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Check if collection exists
COLLECTION_NAME = "docs"
if qdrant.collection_exists(COLLECTION_NAME):
    qdrant.delete_collection(collection_name=COLLECTION_NAME)

# Create collection
qdrant.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=384,  # Adjust to your embedding dimension
        distance=Distance.COSINE
    )
)
def ingest_texts(texts):
    points = []
    for idx, text in enumerate(texts):
        vector = embedder.encode(text).tolist()
        points.append(PointStruct(id=idx, vector=vector, payload={"text": text}))
    qdrant.upsert(collection_name="docs", points=points)

# Example
docs = ["Python is a programming language.", "Django is a web framework."]
ingest_texts(docs)

