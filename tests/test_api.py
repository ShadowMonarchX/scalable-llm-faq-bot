import pytest
from fastapi.testclient import TestClient
from main import app  # Ensure main.py has `app = FastAPI()`

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "Server is running."

def test_query_endpoint():
    response = client.post("/api/query", json={"question": "What is 5G?"})
    assert response.status_code == 200
    assert "response" in response.json()
    assert "retrieved" in response.json()

def test_upload_documents():
    payload = {
        "documents": ["This is a test document."],
        "metadatas": [{"source": "unit_test"}],
        "ids": ["test-doc-1"]
    }
    response = client.post("/api/upload", json=payload)
    assert response.status_code == 200
    assert "status" in response.json()
