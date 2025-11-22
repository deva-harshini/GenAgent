from fastapi import FastAPI
from app.api.v1.predict import router as predict_router
from app.api.v1.rag import router as rag_router

app = FastAPI(title="GenAgent API", version="1.0")

@app.get("/")
def home():
    return {"message": "GenAgent is running"}

app.include_router(predict_router, prefix="/api/v1")
app.include_router(rag_router, prefix="/api/v1")