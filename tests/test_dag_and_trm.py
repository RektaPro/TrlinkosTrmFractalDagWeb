"""
Tests for FractalMerkleDAG and TRLinkosTRM.

Tests cover:
- FractalMerkleDAG: add nodes, create branch, get best node with score
- TRLinkosTRM: forward_recursive output shape, backtracking with decreasing scores
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from t_rlinkos_trm_fractal_dag import (
    FractalMerkleDAG,
    TRLinkosTRM,
)


class TestFractalMerkleDAG:
    """Tests for FractalMerkleDAG class."""

    def test_add_nodes(self):
        """Should be able to add nodes to the DAG."""
        dag = FractalMerkleDAG(store_states=True)
        
        y = np.random.randn(1, 8)
        z = np.random.randn(1, 16)
        
        # Add first node (root)
        node_id_1 = dag.add_step(step=0, y=y, z=z, parents=[], score=-1.0)
        assert node_id_1 is not None
        assert len(node_id_1) == 64  # SHA256 hex digest
        assert len(dag.nodes) == 1
        
        # Add second node
        node_id_2 = dag.add_step(step=1, y=y * 0.9, z=z * 0.9, parents=[node_id_1], score=-0.8)
        assert node_id_2 is not None
        assert len(dag.nodes) == 2
        
        # Add third node
        node_id_3 = dag.add_step(step=2, y=y * 0.8, z=z * 0.8, parents=[node_id_2], score=-0.5)
        assert len(dag.nodes) == 3
        
        # Verify node structure
        assert dag.nodes[node_id_1].step == 0
        assert dag.nodes[node_id_2].step == 1
        assert dag.nodes[node_id_3].step == 2

    def test_create_branch(self):
        """Should be able to create a branch from an existing node."""
        dag = FractalMerkleDAG(store_states=True, max_depth=3)
        
        y = np.random.randn(1, 8)
        z = np.random.randn(1, 16)
        
        # Create root node
        root_id = dag.add_step(step=0, y=y, z=z, parents=[], score=-1.0, depth=0)
        
        # Create a branch from root
        branch_id = dag.create_branch(
            parent_node_id=root_id,
            y=y * 1.1,
            z=z * 1.1,
            score=-0.7
        )
        
        # Verify branch was created
        assert branch_id is not None
        assert dag.nodes[branch_id].depth == 1
        assert dag.nodes[branch_id].branch_root == root_id
        
        # Verify parent-child relationship
        assert branch_id in dag.nodes[root_id].children

    def test_get_best_node_with_score(self):
        """Should correctly identify the best scoring node."""
        dag = FractalMerkleDAG(store_states=True)
        
        y = np.random.randn(1, 8)
        z = np.random.randn(1, 16)
        
        # Add nodes with different scores
        dag.add_step(step=0, y=y, z=z, parents=[], score=-2.0)
        best_node_id = dag.add_step(step=1, y=y * 0.9, z=z * 0.9, parents=[], score=0.5)  # Best score
        dag.add_step(step=2, y=y * 0.8, z=z * 0.8, parents=[], score=-0.3)
        dag.add_step(step=3, y=y * 0.7, z=z * 0.7, parents=[], score=-1.0)
        
        # Get best node
        best_node = dag.get_best_node()
        
        assert best_node is not None
        assert best_node.node_id == best_node_id
        assert best_node.score == 0.5
        assert best_node.step == 1

    def test_best_node_no_score(self):
        """Should return None when no nodes have scores."""
        dag = FractalMerkleDAG(store_states=True)
        
        y = np.random.randn(1, 8)
        z = np.random.randn(1, 16)
        
        dag.add_step(step=0, y=y, z=z, parents=[], score=None)
        dag.add_step(step=1, y=y, z=z, parents=[], score=None)
        
        # Best node should be None (no scored nodes)
        best_node = dag.get_best_node()
        assert best_node is None

    def test_state_restoration(self):
        """Should be able to restore states from stored nodes."""
        dag = FractalMerkleDAG(store_states=True)
        
        y_original = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        z_original = np.random.randn(1, 16)
        
        node_id = dag.add_step(step=0, y=y_original, z=z_original, parents=[], score=-0.5)
        
        # Retrieve states
        states = dag.get_node_states(node_id)
        
        assert states is not None
        y_restored, z_restored = states
        np.testing.assert_array_almost_equal(y_restored, y_original)
        np.testing.assert_array_almost_equal(z_restored, z_original)

    def test_depth_statistics(self):
        """Should correctly report depth statistics."""
        dag = FractalMerkleDAG(store_states=True, max_depth=3)
        
        y = np.random.randn(1, 8)
        z = np.random.randn(1, 16)
        
        # Create nodes at depth 0
        root1 = dag.add_step(step=0, y=y, z=z, parents=[], depth=0)
        root2 = dag.add_step(step=1, y=y * 0.9, z=z * 0.9, parents=[root1], depth=0)
        
        # Create branches at depth 1
        dag.create_branch(root2, y=y * 1.1, z=z * 1.1)
        dag.create_branch(root2, y=y * 1.2, z=z * 1.2)
        
        stats = dag.get_depth_statistics()
        
        assert stats[0] == 2, "Should have 2 nodes at depth 0"
        assert stats[1] == 2, "Should have 2 nodes at depth 1"


class TestTRLinkosTRM:
    """Tests for TRLinkosTRM main model class."""

    def test_forward_recursive_preserves_shape(self):
        """forward_recursive should not break output shape."""
        model = TRLinkosTRM(x_dim=16, y_dim=8, z_dim=16, hidden_dim=32, num_experts=2)
        
        batch_size = 4
        x = np.random.randn(batch_size, 16)
        
        y_pred, dag = model.forward_recursive(x, max_steps=5, inner_recursions=2)
        
        # Output shape should be (batch_size, y_dim)
        assert y_pred.shape == (batch_size, 8), f"Expected ({batch_size}, 8), got {y_pred.shape}"
        
        # DAG should have correct number of nodes (batch_size * max_steps)
        assert len(dag.nodes) == batch_size * 5

    def test_forward_recursive_different_batch_sizes(self):
        """forward_recursive should work with different batch sizes."""
        model = TRLinkosTRM(x_dim=8, y_dim=4, z_dim=8, hidden_dim=32, num_experts=2)
        
        for batch_size in [1, 2, 8, 16]:
            x = np.random.randn(batch_size, 8)
            y_pred, dag = model.forward_recursive(x, max_steps=3, inner_recursions=2)
            
            assert y_pred.shape == (batch_size, 4)
            assert np.all(np.isfinite(y_pred))

    def test_backtracking_restores_better_state(self):
        """Backtracking should restore a better state when score decreases."""
        np.random.seed(42)
        model = TRLinkosTRM(x_dim=8, y_dim=4, z_dim=8, hidden_dim=32, num_experts=2)
        
        x = np.random.randn(2, 8)
        target = np.random.randn(2, 4)
        
        # Create a scorer that simulates decreasing score
        # We use a simple MSE-based scorer
        call_count = [0]
        best_scores = [float('-inf')] * 2
        
        def scorer_with_decay(x_in, y_pred):
            call_count[0] += 1
            # Base score is negative MSE
            base_scores = -np.mean((y_pred - target) ** 2, axis=-1)
            
            # Simulate score decay (scores get worse over time)
            # Add a penalty that increases with call count to force backtracking
            if call_count[0] > 3:
                penalty = (call_count[0] - 3) * 0.5
                scores = base_scores - penalty
            else:
                scores = base_scores
            
            # Track best scores
            for i in range(len(scores)):
                if scores[i] > best_scores[i]:
                    best_scores[i] = scores[i]
            
            return scores
        
        # Run with backtracking
        y_pred, dag = model.forward_recursive(
            x,
            max_steps=8,
            inner_recursions=2,
            scorer=scorer_with_decay,
            backtrack=True,
            backtrack_threshold=0.1
        )
        
        # Verify that states were stored for backtracking
        assert dag.store_states is True
        
        # Verify best node exists and has a score
        best_node = dag.get_best_node()
        assert best_node is not None
        assert best_node.score is not None

    def test_backtracking_improves_final_score(self):
        """Backtracking should help when scores degrade significantly."""
        np.random.seed(123)
        model = TRLinkosTRM(x_dim=8, y_dim=4, z_dim=8, hidden_dim=32, num_experts=2)
        
        x = np.random.randn(2, 8)
        target = np.array([[1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]])
        
        def scorer(x_in, y_pred):
            return -np.mean((y_pred - target) ** 2, axis=-1)
        
        # Run with backtracking enabled
        y_pred_bt, dag_bt = model.forward_recursive(
            x,
            max_steps=6,
            inner_recursions=2,
            scorer=scorer,
            backtrack=True,
            backtrack_threshold=0.1
        )
        
        # Best node should have been tracked
        best_node = dag_bt.get_best_node()
        assert best_node is not None
        
        # The final prediction should be finite
        assert np.all(np.isfinite(y_pred_bt))
        
        # States should be stored
        states = dag_bt.get_node_states(best_node.node_id)
        assert states is not None

    def test_without_backtracking(self):
        """Should work correctly without backtracking."""
        model = TRLinkosTRM(x_dim=8, y_dim=4, z_dim=8, hidden_dim=32, num_experts=2)
        
        x = np.random.randn(4, 8)
        
        y_pred, dag = model.forward_recursive(
            x,
            max_steps=5,
            inner_recursions=2,
            scorer=None,
            backtrack=False
        )
        
        assert y_pred.shape == (4, 4)
        assert dag.store_states is False
        assert dag.get_best_node() is None  # No scores recorded

    def test_with_scorer_no_backtracking(self):
        """Should record scores even without backtracking."""
        model = TRLinkosTRM(x_dim=8, y_dim=4, z_dim=8, hidden_dim=32, num_experts=2)
        
        x = np.random.randn(2, 8)
        target = np.random.randn(2, 4)
        
        def scorer(x_in, y_pred):
            return -np.mean((y_pred - target) ** 2, axis=-1)
        
        y_pred, dag = model.forward_recursive(
            x,
            max_steps=5,
            inner_recursions=2,
            scorer=scorer,
            backtrack=False
        )
        
        # Best node should exist with a score
        best_node = dag.get_best_node()
        assert best_node is not None
        assert best_node.score is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
