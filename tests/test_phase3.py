import pytest
from tests.conftest import client

def test_api_upload_invalid_file():
    # Only PDFs should be allowed
    response = client.post(
        "/api/upload",
        files={"file": ("test.txt", b"dummy content")}
    )
    assert response.status_code == 400
    assert "Only PDF files" in response.json()["detail"]

def test_list_sessions():
    response = client.get("/api/sessions")
    assert response.status_code == 200
    assert "sessions" in response.json()
