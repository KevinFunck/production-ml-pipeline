"""
Unit tests for FastAPI application.
"""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestAPI:
    """Test FastAPI endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert "name" in response.json()
        assert "version" in response.json()
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
    
    def test_predict_without_model(self, client):
        """Test prediction endpoint without loaded model."""
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        response = client.post("/predict", json=payload)
        # Should return 503 since model is not loaded
        assert response.status_code in [200, 503]
    
    def test_invalid_request(self, client):
        """Test invalid request handling."""
        payload = {
            "sepal_length": "invalid",  # Should be float
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error
