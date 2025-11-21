from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict():
    payload = {"features": [5.1, 3.5, 1.4, 0.2]}
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
