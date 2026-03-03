import pytest
import os
import shutil
from fastapi.testclient import TestClient
from main import app

# Create test client
client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Setup test directories and ensure isolation."""
    # Ensure test directories exist
    os.makedirs("./data/uploads", exist_ok=True)
    os.makedirs("./data/vectorstore", exist_ok=True)
    os.makedirs("./data/sessions", exist_ok=True)
    
    # Run tests
    yield
    
    # Teardown logic if needed (optional)
    # We could clean up the test vectorstore or test sessions here
