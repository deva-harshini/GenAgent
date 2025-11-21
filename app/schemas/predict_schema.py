from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    features: List[float] = Field(
        ..., 
        example=[5.1, 3.5, 1.4, 0.2]
    )

class PredictResponse(BaseModel):
    prediction: int
