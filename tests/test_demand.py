from fastapi.testclient import TestClient
from app.main import app
from app.core.security import create_access_token


client = TestClient(app)


def get_auth_header():
    token = create_access_token({"sub": "test@example.com"})
    return {"Authorization": f"Bearer {token}"}


def test_predict_demand():
    payload = {"route_id": 1, "hours_ahead": 4}
    resp = client.post("/api/v1/analytics/demand/predict", json=payload, headers=get_auth_header())
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("route_id") == 1
    assert "predictions" in data
