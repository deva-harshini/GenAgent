"""
rag_service.py — loads FAISS index + embeddings model, performs retrieval
"""

import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pathlib import Path

INDEX_PATH = "faiss_index.bin"
META_PATH = "faiss_meta.json"
EMB_MODEL_NAME = "all-MiniLM-L6-v2"

class RAGService:
    def __init__(self):
        self.index = None
        self.meta = None
        self.model = None
        self._load_resources()

    def _load_resources(self):
        if Path(INDEX_PATH).exists() and Path(META_PATH).exists():
            print("Loading embedding model...")
            self.model = SentenceTransformer(EMB_MODEL_NAME)
            print("Loading FAISS index...")
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
            print("RAG resources loaded.")
        else:
            print("No FAISS index or metadata found. Run scripts/build_rag_index.py first.")

    def search(self, query, top_k=3):
        """Return top_k chunks for the query"""
        if self.index is None or self.model is None:
            raise RuntimeError("RAG resources not loaded.")
        q_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx in I[0]:
            results.append({
                "text": self.meta["texts"][idx],
                "meta": self.meta["metadatas"][idx]
            })
        return results

# singleton
rag_service = RAGService()
