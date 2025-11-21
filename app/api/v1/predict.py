from fastapi import APIRouter, HTTPException
from app.schemas.predict_schema import PredictRequest, PredictResponse
from app.models.model_loader import model
import numpy as np

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
def predict(data: PredictRequest):
    try:
        X = np.array([data.features])
        prediction = model.predict(X)[0]
        return PredictResponse(prediction=int(prediction))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict_async", response_model=PredictResponse)
async def predict_async(data: PredictRequest):
    # same logic, just async
    X = np.array([data.features])
    prediction = model.predict(X)[0]
    return PredictResponse(prediction=int(prediction))