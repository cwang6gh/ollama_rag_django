import os
import io
import base64
from typing import List, Union
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# --- Document and image parsing imports ---
from PyPDF2 import PdfReader
from docx import Document
from PIL import Image
import numpy as np

# --- Models ---
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
IMAGE_MODEL = "clip-ViT-B-32"  # multimodal model for images

# --- Initialize Qdrant and models ---
qdrant = QdrantClient(host="localhost", port=6333)
text_embedder = SentenceTransformer(TEXT_MODEL)
image_embedder = SentenceTransformer(IMAGE_MODEL)

COLLECTION_NAME = "docs"

# --- Create collection if not exists ---
if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# --- File type handlers ---
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
        return image_embedder.encode(image)
    except Exception as e:
        print(f"[WARN] Failed to process image {file_path}: {e}")
        return None

# --- Main ingestion function ---
def ingest_files(inputs: List[str]):
    """
    Ingests a list of file paths or directories.
    Supports text, docx, pdf, and common image files.
    """
    points = []
    idx = 0

    supported_exts = [".txt", ".pdf", ".doc", ".docx", ".png", ".jpg", ".jpeg", ".bmp"]

    # --- Expand directories ---
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

    # --- Process files ---
    for file_path in all_files:
        ext = os.path.splitext(file_path)[1].lower()
        content = None
        vector = None

        if ext == ".txt":
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    vector = text_embedder.encode(content).tolist()
            except Exception as e:
                print(f"[WARN] Failed to read TXT {file_path}: {e}")

        elif ext == ".pdf":
            content = extract_text_from_pdf(file_path)
            if content:
                vector = text_embedder.encode(content).tolist()

        elif ext in [".doc", ".docx"]:
            content = extract_text_from_docx(file_path)
            if content:
                vector = text_embedder.encode(content).tolist()

        elif ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            vector = extract_vector_from_image(file_path)
            if vector is not None:
                vector = vector.tolist()
            content = f"Image file: {os.path.basename(file_path)}"

        else:
            print(f"[SKIP] Unsupported file: {file_path}")
            continue

        if vector is not None:
            points.append(PointStruct(id=idx, vector=vector, payload={"text": content or "", "path": file_path}))
            idx += 1

    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"[INFO] Successfully ingested {len(points)} items.")
    else:
        print("[WARN] No valid files found for ingestion.")

# --- Query function ---
def query_qdrant(query: str, top_k: int = 3) -> List[str]:
    """
    Searches the Qdrant collection for documents similar to the query.
    """
    if not query:
        return []

    vector = text_embedder.encode(query).tolist()
    search_result = qdrant.search(collection_name=COLLECTION_NAME, query_vector=vector, limit=top_k)
    return [hit.payload.get("text", "") for hit in search_result]

# --- Example usage ---
if __name__ == "__main__":
    # You can pass a mix of directories and files
    sources = [
        "../../../data/"   # a directory containing pdf/docx/txt/image files
        #docs/intro.pdf",    # a specific file
        #"images/"            # an image directory
    ]

    ingest_files(sources)

    query = "Explain Django and its purpose."
    results = query_qdrant(query)
    print("\n🔍 Query Results:")
    for i, r in enumerate(results, start=1):
        print(f"{i}. {r[:300]}{'...' if len(r) > 300 else ''}")


