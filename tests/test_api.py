from fastapi.testclient import TestClient
from src.legalrag.api.app import app

client = TestClient(app)

def test_health():
    resp = client.get(\"/health\")
    assert resp.status_code == 200

def test_ask_smoke():
    resp = client.post(\"/ask\", json={\"question\": \"Was regelt § 433 Abs. 1 BGB?\", \"language\": \"ar\"})
    assert resp.status_code == 200
    body = resp.json()
    assert \"meta\" in body
