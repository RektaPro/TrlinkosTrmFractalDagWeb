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
        app_state.text_encoder = None
        app_state.last_dag = None

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


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestBatchReasoningEndpoint:
    """Tests for the batch reasoning endpoint."""

    @pytest.fixture(autouse=True)
    def setup_app_state(self):
        """Initialize app state before each test."""
        app_state.initialize()
        yield
        app_state.model = None
        app_state.config = {}
        app_state.text_encoder = None
        app_state.last_dag = None

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_batch_reason_valid_input(self, client):
        """Batch reason endpoint should process valid batch input."""
        batch = [np.random.randn(DEFAULT_X_DIM).tolist() for _ in range(3)]
        response = client.post(
            "/reason/batch",
            json={
                "batch": batch,
                "max_steps": 5,
                "inner_recursions": 2,
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "outputs" in data
        assert len(data["outputs"]) == 3
        assert len(data["outputs"][0]) == DEFAULT_Y_DIM
        assert data["batch_size"] == 3
        assert "dag_stats" in data

    def test_batch_reason_wrong_dimension(self, client):
        """Batch reason endpoint should reject wrong dimension in batch."""
        batch = [
            np.random.randn(DEFAULT_X_DIM).tolist(),
            [0.1, 0.2, 0.3],  # Wrong dimension
        ]
        response = client.post(
            "/reason/batch",
            json={"batch": batch}
        )
        assert response.status_code == 400
        assert "Sample 1" in response.json()["detail"]

    def test_batch_reason_single_item(self, client):
        """Batch reason endpoint should handle single item batch."""
        batch = [np.random.randn(DEFAULT_X_DIM).tolist()]
        response = client.post(
            "/reason/batch",
            json={"batch": batch, "max_steps": 3}
        )
        assert response.status_code == 200
        assert len(response.json()["outputs"]) == 1


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestTextReasoningEndpoint:
    """Tests for the text reasoning endpoint."""

    @pytest.fixture(autouse=True)
    def setup_app_state(self):
        """Initialize app state before each test."""
        app_state.initialize()
        yield
        app_state.model = None
        app_state.config = {}
        app_state.text_encoder = None
        app_state.last_dag = None

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_text_reason_valid_input(self, client):
        """Text reason endpoint should process valid text."""
        response = client.post(
            "/reason/text",
            json={
                "text": "What is machine learning?",
                "max_steps": 5,
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "output" in data
        assert len(data["output"]) == DEFAULT_Y_DIM
        assert "dag_stats" in data
        assert "text_embedding_dim" in data
        assert data["text_embedding_dim"] == DEFAULT_X_DIM

    def test_text_reason_short_text(self, client):
        """Text reason endpoint should handle short text."""
        response = client.post(
            "/reason/text",
            json={"text": "AI", "max_steps": 3}
        )
        assert response.status_code == 200

    def test_text_reason_long_text(self, client):
        """Text reason endpoint should handle long text."""
        long_text = "This is a test. " * 100
        response = client.post(
            "/reason/text",
            json={"text": long_text, "max_steps": 3}
        )
        assert response.status_code == 200


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestDAGVisualizationEndpoint:
    """Tests for the DAG visualization endpoint."""

    @pytest.fixture(autouse=True)
    def setup_app_state(self):
        """Initialize app state before each test."""
        app_state.initialize()
        yield
        app_state.model = None
        app_state.config = {}
        app_state.text_encoder = None
        app_state.last_dag = None

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_dag_visualize_no_dag(self, client):
        """DAG visualize should return 404 when no DAG available."""
        response = client.get("/dag/visualize")
        assert response.status_code == 404

    def test_dag_visualize_json_format(self, client):
        """DAG visualize should return JSON format."""
        # First run reasoning to create a DAG
        features = np.random.randn(DEFAULT_X_DIM).tolist()
        client.post("/reason", json={"features": features, "max_steps": 3})

        response = client.get("/dag/visualize?format=json")
        assert response.status_code == 200
        data = response.json()
        assert "metadata" in data
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) > 0

    def test_dag_visualize_html_format(self, client):
        """DAG visualize should return HTML format."""
        # First run reasoning to create a DAG
        features = np.random.randn(DEFAULT_X_DIM).tolist()
        client.post("/reason", json={"features": features, "max_steps": 3})

        response = client.get("/dag/visualize?format=html")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "T-RLINKOS" in response.text

    def test_dag_visualize_dot_format(self, client):
        """DAG visualize should return DOT format."""
        # First run reasoning to create a DAG
        features = np.random.randn(DEFAULT_X_DIM).tolist()
        client.post("/reason", json={"features": features, "max_steps": 3})

        response = client.get("/dag/visualize?format=dot")
        assert response.status_code == 200
        assert "digraph" in response.text


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestModelInfoEndpoint:
    """Tests for the model info endpoint."""

    @pytest.fixture(autouse=True)
    def setup_app_state(self):
        """Initialize app state before each test."""
        app_state.initialize()
        yield
        app_state.model = None
        app_state.config = {}
        app_state.text_encoder = None
        app_state.last_dag = None

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_model_info(self, client):
        """Model info endpoint should return model details."""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "config" in data
        assert "total_parameters" in data
        assert "components" in data
        assert data["total_parameters"] > 0
        assert data["config"]["x_dim"] == DEFAULT_X_DIM
        assert "experts" in data["components"]
        assert "text_encoder" in data["components"]


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestBenchmarkEndpoint:
    """Tests for the benchmark endpoint."""

    @pytest.fixture(autouse=True)
    def setup_app_state(self):
        """Initialize app state before each test."""
        app_state.initialize()
        yield
        app_state.model = None
        app_state.config = {}
        app_state.text_encoder = None
        app_state.last_dag = None

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_benchmark_default(self, client):
        """Benchmark endpoint should run with defaults."""
        response = client.get("/benchmark")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "throughput_samples_per_sec" in data
        assert "time_per_step_ms" in data
        assert "memory_estimate_mb" in data
        assert data["throughput_samples_per_sec"] > 0

    def test_benchmark_custom_params(self, client):
        """Benchmark endpoint should accept custom parameters."""
        response = client.get("/benchmark?batch_size=4&max_steps=4&num_runs=2")
        assert response.status_code == 200
        data = response.json()
        assert data["throughput_samples_per_sec"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
