from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.services.rag_service import rag_service
import subprocess
import threading

router = APIRouter()

class RAGRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

@router.post("/rag")
def rag_query(req: RAGRequest):
    try:
        results = rag_service.search(req.query, top_k=req.top_k)
        # Naive answer: concatenate top results for now
        answer = " ".join([r["text"] for r in results])
        return {"query": req.query, "answer": answer, "hits": results}
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: endpoint to trigger re-index (runs the build script in background)
@router.post("/rag/reindex")
def reindex():
    def run_index():
        subprocess.run(["python", "scripts/build_rag_index.py"], check=True)
        # reload resources after indexing
        rag_service._load_resources()
    thread = threading.Thread(target=run_index)
    thread.start()
    return {"status": "reindex_started"}