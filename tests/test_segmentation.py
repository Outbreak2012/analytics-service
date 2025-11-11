from fastapi.testclient import TestClient
from app.main import app
from app.core.security import create_access_token


client = TestClient(app)


def get_auth_header():
    token = create_access_token({"sub": "test@example.com"})
    return {"Authorization": f"Bearer {token}"}


def test_segment_users():
    payload = {"eps": 0.5, "min_samples": 5}
    resp = client.post("/api/v1/analytics/users/segment", json=payload, headers=get_auth_header())
    assert resp.status_code == 200
    data = resp.json()
    assert "clusters" in data
