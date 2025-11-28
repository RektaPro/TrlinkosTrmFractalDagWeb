"""
Tests for the T-RLINKOS API endpoints.
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fastapi.testclient import TestClient
    from api import app, app_state, DEFAULT_X_DIM, DEFAULT_Y_DIM
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestAPIEndpoints:
    """Tests for API endpoints."""

    @pytest.fixture(autouse=True)
    def setup_app_state(self):
        """Initialize app state before each test."""
        app_state.initialize()
        yield
        # Cleanup after test
        app_state.model = None
        app_state.config = {}

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Health endpoint should return ok status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "trm_config" in data

    def test_reason_endpoint_valid_input(self, client):
        """Reason endpoint should process valid input."""
        features = np.random.randn(DEFAULT_X_DIM).tolist()
        response = client.post(
            "/reason",
            json={
                "features": features,
                "max_steps": 5,
                "inner_recursions": 2,
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "output" in data
        assert len(data["output"]) == DEFAULT_Y_DIM
        assert "dag_stats" in data
        assert "num_nodes" in data["dag_stats"]

    def test_reason_endpoint_wrong_dimension(self, client):
        """Reason endpoint should reject wrong input dimension."""
        features = [0.1, 0.2, 0.3]  # Wrong dimension
        response = client.post(
            "/reason",
            json={"features": features}
        )
        assert response.status_code == 400

    def test_reason_endpoint_with_backtrack(self, client):
        """Reason endpoint should work with backtracking."""
        features = np.random.randn(DEFAULT_X_DIM).tolist()
        response = client.post(
            "/reason",
            json={
                "features": features,
                "max_steps": 5,
                "backtrack": True,
            }
        )
        assert response.status_code == 200

    def test_reason_endpoint_dag_stats(self, client):
        """Reason endpoint should return valid DAG statistics."""
        features = np.random.randn(DEFAULT_X_DIM).tolist()
        response = client.post(
            "/reason",
            json={
                "features": features,
                "max_steps": 5,
            }
        )
        assert response.status_code == 200
        dag_stats = response.json()["dag_stats"]
        assert "num_nodes" in dag_stats
        assert dag_stats["num_nodes"] > 0
        assert "max_depth" in dag_stats
        assert "depth_statistics" in dag_stats

    def test_docs_endpoint(self, client):
        """Documentation endpoint should be accessible."""
        response = client.get("/docs")
        # FastAPI docs return 200 with HTML
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
