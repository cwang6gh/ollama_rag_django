from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import requests
import json

# Config
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "docs"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Init
qdrant = QdrantClient(host="localhost", port=6333)
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def embed_text(text):
    return embedder.encode(text).tolist()

def query_qdrant(query, top_k=5):
    vec = embed_text(query)
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vec,
        limit=top_k
    )
    return [hit.payload['text'] for hit in results]


def generate_with_ollama(prompt, image=None):
    data = {"model": "llama3.2-vision", "prompt": prompt}
    if image:
        data["images"] = [image]

    response = requests.post("http://localhost:11434/api/generate", json=data, stream=True)

    if response.status_code != 200:
        raise Exception(f"Ollama error: {response.status_code} - {response.text}")

    full_response = ""
    for line in response.iter_lines():
        if line:
            try:
                chunk = line.decode("utf-8")
                json_chunk = json.loads(chunk)
                full_response += json_chunk.get("response", "")
            except Exception as e:
                print("⚠️ Error parsing chunk:", e)

    return full_response.strip()

