from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import requests
import json
import os
from PIL import Image

# --- Configuration ---
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "docs"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "clip-ViT-B-32"  # same as ingest.py
VECTOR_SIZE = 512

# --- Initialize Qdrant and embedder ---
qdrant = QdrantClient(host="localhost", port=6333)
embedder = SentenceTransformer(MODEL_NAME)

# --- Helper: Encode text or image into vector ---
def embed_input(input_data):
    """
    Encodes either a text string or an image path into a 512D vector
    using the CLIP model.
    """
    if os.path.isfile(input_data) and os.path.splitext(input_data)[1].lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
        image = Image.open(input_data).convert("RGB")
        return embedder.encode(image).tolist()
    else:
        return embedder.encode(input_data).tolist()

# --- Query Qdrant ---
def query_qdrant(query, top_k=5):
    """
    Searches Qdrant for documents or images similar to a given text or image query.
    """
    vec = embed_input(query)
    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vec,
        limit=top_k
    )

    return [
        {
            "text": hit.payload.get("text", ""),
            "path": hit.payload.get("path", ""),
            "score": hit.score
        }
        for hit in results
    ]

# --- Generate response with Ollama ---
def generate_with_ollama(prompt, image=None):
    """
    Uses Ollama's multimodal model (e.g., llama3.2-vision) to generate a response.
    Optionally passes an image (base64 or file) to the model.
    """
    data = {"model": "llama3.2-vision", "prompt": prompt}

    if image:
        data["images"] = [image]  # expects base64-encoded or file path handled by Ollama

    response = requests.post(OLLAMA_URL, json=data, stream=True)

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

# --- Main RAG function ---
def rag_query(user_query, top_k=5):
    """
    Retrieves context from Qdrant and feeds it into Ollama for final answer generation.
    """
    hits = query_qdrant(user_query, top_k=top_k)
    context = "\n\n".join([f"[{i+1}] {hit['text']}" for i, hit in enumerate(hits)])

    prompt = f"""
You are a helpful assistant with access to retrieved context.
User query: {user_query}

Relevant documents:
{context}

Provide a concise, helpful answer based on the context above.
    """.strip()

    return generate_with_ollama(prompt)

# --- Example usage ---
if __name__ == "__main__":
    query = "What is Django and how does it work?"
    print("🔍 Querying Qdrant...")
    answer = rag_query(query)
    print("\n💬 Ollama Response:\n")
    print(answer)
