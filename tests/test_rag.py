from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_rag():
    payload = {"query": "predictive maintenance", "top_k": 2}
    res = client.post("/api/v1/rag", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "hits" in data
    assert isinstance(data["hits"], list)
