from fastapi.testclient import TestClient
from app.main import app
from app.core.security import create_access_token


client = TestClient(app)


def get_auth_header():
    token = create_access_token({"sub": "test@example.com"})
    return {"Authorization": f"Bearer {token}"}


def test_analyze_sentiment():
    payload = {"text": "Me encanta este servicio"}
    resp = client.post("/api/v1/analytics/sentiment/analyze", json=payload, headers=get_auth_header())
    assert resp.status_code == 200
    data = resp.json()
    assert "sentiment" in data
