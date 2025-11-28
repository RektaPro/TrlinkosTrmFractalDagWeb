"""Tests for MCP server and tools."""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server import TRLinkosMCPServer


class TestMCPServer:
    """Tests for TRLinkosMCPServer."""

    @pytest.fixture
    def server(self):
        """Create a test server instance."""
        return TRLinkosMCPServer(
            x_dim=16,
            y_dim=8,
            z_dim=16,
            hidden_dim=32,
            num_experts=2,
        )

    def test_server_initialization(self, server):
        """Server should initialize with correct config."""
        assert server.config["x_dim"] == 16
        assert server.config["y_dim"] == 8
        assert server.config["z_dim"] == 16
        assert server.model is not None

    def test_reason_step(self, server):
        """reason_step should return valid output."""
        x = np.random.randn(16).tolist()
        result = server.reason_step(x)

        assert "y_next" in result
        assert "z_next" in result
        assert len(result["y_next"]) == 8
        assert len(result["z_next"]) == 16

    def test_reason_step_with_state(self, server):
        """reason_step should work with provided state."""
        x = np.random.randn(16).tolist()
        y = np.random.randn(8).tolist()
        z = np.random.randn(16).tolist()

        result = server.reason_step(x, y=y, z=z, inner_recursions=2)

        assert "y_next" in result
        assert "z_next" in result
        assert result["inner_recursions"] == 2

    def test_run_trm_recursive(self, server):
        """run_trm_recursive should complete reasoning loop."""
        x = np.random.randn(16).tolist()
        result = server.run_trm_recursive(x, max_steps=5, inner_recursions=2)

        assert "y_pred" in result
        assert "dag_id" in result
        assert "dag_stats" in result
        assert len(result["y_pred"]) == 8
        assert result["dag_stats"]["num_nodes"] > 0

    def test_run_trm_recursive_with_fractal(self, server):
        """run_trm_recursive should work with fractal branching."""
        x = np.random.randn(16).tolist()
        result = server.run_trm_recursive(
            x,
            max_steps=5,
            fractal_branching=True,
            backtrack=True,
        )

        assert "y_pred" in result
        assert result["config"]["fractal_branching"] is True

    def test_dag_add_node(self, server):
        """dag_add_node should create DAG and add node."""
        y = np.random.randn(8).tolist()
        z = np.random.randn(16).tolist()

        result = server.dag_add_node(
            dag_id="new_dag",
            step=0,
            y=y,
            z=z,
            score=-0.5,
        )

        assert "dag_id" in result
        assert "node_id" in result
        assert result["step"] == 0

    def test_dag_get_state(self, server):
        """dag_get_state should return DAG statistics."""
        # First create a DAG via reasoning
        x = np.random.randn(16).tolist()
        result = server.run_trm_recursive(x, max_steps=3)
        dag_id = result["dag_id"]

        state = server.dag_get_state(dag_id)

        assert state["dag_id"] == dag_id
        assert "num_nodes" in state
        assert "depth_statistics" in state

    def test_dag_best_path(self, server):
        """dag_best_path should return path to best node."""
        # Create DAG with nodes that have scores
        y = np.random.randn(8).tolist()
        z = np.random.randn(16).tolist()

        # Add nodes with scores
        result1 = server.dag_add_node(
            dag_id="test_dag",
            step=0,
            y=y,
            z=z,
            score=-1.0,
        )
        dag_id = result1["dag_id"]
        node1_id = result1["node_id"]

        result2 = server.dag_add_node(
            dag_id=dag_id,
            step=1,
            y=[v * 0.9 for v in y],
            z=[v * 0.9 for v in z],
            parents=[node1_id],
            score=-0.5,  # Better score
        )

        path_result = server.dag_best_path(dag_id)

        assert path_result["dag_id"] == dag_id
        assert "path" in path_result
        assert len(path_result["path"]) > 0
        assert path_result["best_score"] == -0.5

    def test_torque_route(self, server):
        """torque_route should return routing weights."""
        x = np.random.randn(16).tolist()
        y = np.random.randn(8).tolist()
        z = np.random.randn(16).tolist()

        result = server.torque_route(x, y, z)

        assert "weights" in result
        assert len(result["weights"]) == 2  # num_experts
        assert "top_expert" in result
        assert 0 <= result["top_expert"] < 2

    def test_dcaap_forward(self, server):
        """dcaap_forward should return z_next state."""
        x = np.random.randn(16).tolist()
        y = np.random.randn(8).tolist()
        z = np.random.randn(16).tolist()

        result = server.dcaap_forward(x, y, z)

        assert "z_next" in result
        assert len(result["z_next"]) == 16

    def test_fractal_branch(self, server):
        """fractal_branch should create branch in DAG."""
        # Create DAG via reasoning
        x = np.random.randn(16).tolist()
        result = server.run_trm_recursive(x, max_steps=3)
        dag_id = result["dag_id"]

        # Get a node to branch from
        dag = server.dags[dag_id]
        parent_id = list(dag.nodes.keys())[0]

        y = np.random.randn(8).tolist()
        z = np.random.randn(16).tolist()

        branch_result = server.fractal_branch(
            dag_id=dag_id,
            parent_node_id=parent_id,
            y=y,
            z=z,
            score=-0.3,
        )

        assert "branch_node_id" in branch_result or "error" in branch_result

    def test_evaluate_score_mse(self, server):
        """evaluate_score should compute MSE."""
        y_pred = [0.1, 0.2, 0.3]
        y_target = [0.15, 0.25, 0.35]

        result = server.evaluate_score(y_pred, y_target, metric="mse")

        assert "score" in result
        assert result["metric"] == "mse"
        assert result["score"] >= 0

    def test_evaluate_score_cosine(self, server):
        """evaluate_score should compute cosine similarity."""
        y_pred = [1.0, 0.0, 0.0]
        y_target = [0.0, 1.0, 0.0]

        result = server.evaluate_score(y_pred, y_target, metric="cosine")

        assert "score" in result
        assert result["metric"] == "cosine"

    def test_get_repo_state_file(self, server):
        """get_repo_state should read files."""
        result = server.get_repo_state("README.md")

        assert result["type"] == "file"
        assert "content" in result
        assert len(result["content"]) > 0

    def test_get_repo_state_directory(self, server):
        """get_repo_state should list directories."""
        result = server.get_repo_state("tests")

        assert result["type"] == "directory"
        assert "entries" in result
        assert len(result["entries"]) > 0

    def test_get_repo_state_not_found(self, server):
        """get_repo_state should handle missing paths."""
        result = server.get_repo_state("nonexistent_file.txt")

        assert "error" in result

    def test_handle_tool_call(self, server):
        """handle_tool_call should dispatch to correct tool."""
        x = np.random.randn(16).tolist()

        result = server.handle_tool_call("reason_step", {"x": x})

        assert "y_next" in result

    def test_handle_tool_call_unknown(self, server):
        """handle_tool_call should handle unknown tools."""
        result = server.handle_tool_call("unknown_tool", {})

        assert "error" in result

    def test_handle_resource_read_config(self, server):
        """handle_resource_read should return model config."""
        result = server.handle_resource_read("trlinkos://model/config")

        assert "content" in result
        assert result["mimeType"] == "application/json"


class TestMCPTools:
    """Tests for individual MCP tools."""

    def test_reasoning_tools_import(self):
        """Reasoning tools should import correctly."""
        from mcp.tools.reasoning import (
            reason_step,
            run_trm_recursive,
            torque_route,
            dcaap_forward,
            evaluate_score,
        )
        assert callable(reason_step)
        assert callable(run_trm_recursive)

    def test_dag_tools_import(self):
        """DAG tools should import correctly."""
        from mcp.tools.dag import (
            dag_add_node,
            dag_best_path,
            dag_get_state,
            fractal_branch,
        )
        assert callable(dag_add_node)
        assert callable(dag_best_path)

    def test_model_tools_import(self):
        """Model tools should import correctly."""
        from mcp.tools.model import (
            load_model,
            save_model,
            get_model_config,
        )
        assert callable(load_model)
        assert callable(save_model)

    def test_repo_tools_import(self):
        """Repo tools should import correctly."""
        from mcp.tools.repo import (
            get_repo_state,
            write_repo_state,
        )
        assert callable(get_repo_state)
        assert callable(write_repo_state)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
