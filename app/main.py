from fastapi import FastAPI
from app.api.v1.predict import router as predict_router

app = FastAPI(title="GenAgent API", version="1.0")

@app.get("/")
def home():
    return {"message": "GenAgent is running"}

app.include_router(predict_router, prefix="/api/v1")
