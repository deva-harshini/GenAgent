"""
build_rag_index.py
- Reads text files from docs/
- Splits each file into small chunks
- Creates sentence-transformers embeddings for chunks
- Builds a FAISS index and saves index + metadata
"""

import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# 1. Settings
DOCS_DIR = Path("docs")
EMB_MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast embedding model
INDEX_PATH = "faiss_index.bin"
META_PATH = "faiss_meta.json"
CHUNK_SIZE = 250   # characters per chunk (naive, simple)
CHUNK_OVERLAP = 50 # characters overlap between chunks

def read_documents():
    texts = []
    metadatas = []
    for p in sorted(DOCS_DIR.glob("*.txt")):
        raw = p.read_text(encoding="utf-8")
        # naive chunking by characters (simple approach)
        i = 0
        while i < len(raw):
            chunk = raw[i:i+CHUNK_SIZE].strip()
            if chunk:
                texts.append(chunk)
                metadatas.append({"source": p.name, "start": i, "end": i+len(chunk)})
            i += CHUNK_SIZE - CHUNK_OVERLAP
    return texts, metadatas

def build_index(texts):
    # Load embedding model
    model = SentenceTransformer(EMB_MODEL_NAME)
    # Compute embeddings (numpy array)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    # Create FAISS index (L2 / inner-product could also be used if normalized)
    index = faiss.IndexFlatL2(dim)  # simple flat index for demo
    index.add(embeddings)
    return index, embeddings

def save_index(index, metadatas, texts):
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"texts": texts, "metadatas": metadatas}, f, ensure_ascii=False, indent=2)

def main():
    print("Reading documents...")
    texts, metadatas = read_documents()
    print(f"Found {len(texts)} chunks.")
    if len(texts) == 0:
        print("No docs found in docs/ . Add .txt files and retry.")
        return

    print("Building embeddings and FAISS index...")
    index, _ = build_index(texts)
    print("Saving index and metadata...")
    save_index(index, metadatas, texts)
    print("Indexing complete. Saved to:", INDEX_PATH, META_PATH)

if __name__ == "__main__":
    main()
