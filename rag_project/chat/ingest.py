import os
from typing import List, Union
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import numpy as np

# --- Model configuration ---
MODEL_NAME = "clip-ViT-B-32"  # Multimodal: works for text + images (512D)
VECTOR_SIZE = 512
COLLECTION_NAME = "docs"

# --- Initialize model and client ---
embedder = SentenceTransformer(MODEL_NAME)
qdrant = QdrantClient(host="localhost", port=6333)

# --- Create collection if it doesn’t exist ---
if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"[INIT] Created collection '{COLLECTION_NAME}' with dim={VECTOR_SIZE}")

# --- Helpers for file reading ---
def extract_text_from_pdf(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        print(f"[WARN] Failed to read PDF {file_path}: {e}")
        return ""

def extract_text_from_docx(file_path: str) -> str:
    try:
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"[WARN] Failed to read DOCX {file_path}: {e}")
        return ""

def extract_vector_from_image(file_path: str) -> Union[np.ndarray, None]:
    try:
        image = Image.open(file_path).convert("RGB")
        return embedder.encode(image)
    except Exception as e:
        print(f"[WARN] Failed to process image {file_path}: {e}")
        return None

# --- Main ingestion function ---
def ingest_files(inputs: List[str]):
    """
    Ingests PDFs, DOCX, TXT, and images (JPG, PNG, etc.) into one multimodal Qdrant collection.
    """
    points = []
    idx = 0
    supported_exts = [".txt", ".pdf", ".doc", ".docx", ".png", ".jpg", ".jpeg", ".bmp"]

    # --- Collect all files ---
    all_files = []
    for path in inputs:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for f in files:
                    if os.path.splitext(f)[1].lower() in supported_exts:
                        all_files.append(os.path.join(root, f))
        elif os.path.isfile(path):
            all_files.append(path)
        else:
            print(f"[SKIP] Invalid path: {path}")

    # --- Process and encode files ---
    for file_path in all_files:
        ext = os.path.splitext(file_path)[1].lower()
        content = ""
        vector = None

        if ext == ".txt":
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    vector = embedder.encode(content)
            except Exception as e:
                print(f"[WARN] Failed to read TXT {file_path}: {e}")

        elif ext == ".pdf":
            content = extract_text_from_pdf(file_path)
            if content:
                vector = embedder.encode(content)

        elif ext in [".doc", ".docx"]:
            content = extract_text_from_docx(file_path)
            if content:
                vector = embedder.encode(content)

        elif ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            vector = extract_vector_from_image(file_path)
            content = f"Image file: {os.path.basename(file_path)}"

        else:
            print(f"[SKIP] Unsupported file: {file_path}")
            continue

        if vector is not None:
            points.append(
                PointStruct(
                    id=idx,
                    vector=vector.tolist(),
                    payload={"text": content or "", "path": file_path},
                )
            )
            idx += 1

    # --- Upload to Qdrant ---
    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"[INFO] Successfully ingested {len(points)} items into '{COLLECTION_NAME}'")
    else:
        print("[WARN] No valid files found for ingestion.")

# --- Query function ---
def query_qdrant(query: str, top_k: int = 3) -> List[str]:
    """
    Searches Qdrant collection for documents or images similar to the query (text or image path).
    """
    if not query:
        return []

    # Handle query type (text or image file path)
    if os.path.isfile(query) and os.path.splitext(query)[1].lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
        vector = extract_vector_from_image(query)
    else:
        vector = embedder.encode(query)

    if vector is None:
        print("[WARN] Could not encode query.")
        return []

    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector.tolist(),
        limit=top_k,
    )

    return [hit.payload.get("text", "") + " (" + hit.payload.get("path", "") + ")" for hit in search_result]

# --- Example usage ---
if __name__ == "__main__":
    sources = ["../../../data/"]  # directory containing PDFs, DOCX, TXT, and images
    ingest_files(sources)

    query = "Explain Django and its purpose."
    results = query_qdrant(query)
    print("\n🔍 Query Results:")
    for i, r in enumerate(results, start=1):
        print(f"{i}. {r[:300]}{'...' if len(r) > 300 else ''}")
