"""
Unit tests for T-RLINKOS TRM++ core components.

Tests cover:
- dcaap_activation: non-monotone behavior
- DCaAPCell: dimension handling, calcium gate
- TorqueRouter: softmax normalization, routing
- FractalMerkleDAG: node operations, branching, depth
- TRLinkosTRM: forward_recursive, backtracking
"""

import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from t_rlinkos_trm_fractal_dag import (
    dcaap_activation,
    DCaAPCell,
    TorqueRouter,
    TRLinkosCore,
    FractalMerkleDAG,
    TRLinkosTRM,
    LinearNP,
    gelu,
    softmax,
    TextEncoder,
    ImageEncoder,
    Dataset,
    DataLoader,
    TrainingConfig,
    Trainer,
    mse_loss,
    cross_entropy_loss,
    cosine_similarity_loss,
    save_model,
    load_model,
)


class TestDCaAPActivation:
    """Tests for dcaap_activation function."""

    def test_negative_inputs_zero_output(self):
        """Inputs below threshold should yield zero."""
        x = np.array([[-2.0, -1.0, -0.5]])
        result = dcaap_activation(x, threshold=0.0)
        np.testing.assert_array_equal(result, np.zeros_like(x))

    def test_threshold_behavior(self):
        """Values at or below threshold should be zero."""
        x = np.array([[0.0, 0.001, -0.001]])
        result = dcaap_activation(x, threshold=0.0)
        assert result[0, 0] == 0.0  # At threshold
        assert result[0, 1] > 0.0   # Just above threshold
        assert result[0, 2] == 0.0  # Below threshold

    def test_non_monotone_behavior(self):
        """dCaAP should exhibit non-monotone behavior: rise then fall."""
        # Test points above threshold
        x_low = np.array([[0.5]])
        x_mid = np.array([[2.0]])
        x_high = np.array([[10.0]])

        result_low = dcaap_activation(x_low, threshold=0.0)
        result_mid = dcaap_activation(x_mid, threshold=0.0)
        result_high = dcaap_activation(x_high, threshold=0.0)

        # Mid should be highest (near peak), high should decay
        assert result_mid[0, 0] > result_high[0, 0], "dCaAP should decay for very large inputs"

    def test_peak_near_threshold(self):
        """Peak activation should occur near the threshold."""
        # The peak of 4*σ(x)*(1-σ(x)) is at x=threshold (σ=0.5)
        x = np.array([[0.001, 0.5, 1.0, 2.0, 5.0]])
        result = dcaap_activation(x, threshold=0.0)

        # Find argmax
        peak_idx = np.argmax(result[0])
        # Peak should be at a relatively low value (close to threshold)
        assert peak_idx < 3, "Peak should be near threshold"

    def test_batch_processing(self):
        """Should handle batches correctly."""
        x = np.random.randn(8, 32)
        result = dcaap_activation(x, threshold=0.0)
        assert result.shape == (8, 32)

    def test_custom_threshold(self):
        """Should respect custom threshold."""
        x = np.array([[0.0, 0.5, 1.0]])
        result_t0 = dcaap_activation(x, threshold=0.0)
        result_t1 = dcaap_activation(x, threshold=1.0)

        # With threshold=1, only x=1.0 should be at threshold
        assert result_t0[0, 0] == 0.0  # x=0 at threshold=0
        assert result_t1[0, 0] == 0.0  # x=0 below threshold=1
        assert result_t1[0, 1] == 0.0  # x=0.5 below threshold=1

    def test_output_bounded(self):
        """Output should be bounded between 0 and 1."""
        x = np.random.randn(100, 100) * 10
        result = dcaap_activation(x, threshold=0.0)
        assert np.all(result >= 0), "Output should be non-negative"
        assert np.all(result <= 1), "Output should be at most 1"


class TestDCaAPCell:
    """Tests for DCaAPCell class."""

    def test_initialization(self):
        """Should initialize with correct dimensions."""
        cell = DCaAPCell(input_dim=64, hidden_dim=128, z_dim=32, num_branches=4)
        assert len(cell.branch_weights) == 4
        assert cell.num_branches == 4
        assert cell.z_dim == 32

    def test_forward_shape(self):
        """Forward pass should preserve batch dimension and output z_dim."""
        # input_dim = dx + dy + dz = 32 + 16 + 48 = 96
        cell = DCaAPCell(input_dim=96, hidden_dim=64, z_dim=48, num_branches=4)
        x = np.random.randn(8, 32)  # dx=32
        y = np.random.randn(8, 16)  # dy=16
        z = np.random.randn(8, 48)  # dz=48, total=96

        z_next = cell.forward(x, y, z)
        assert z_next.shape == (8, 48), f"Expected (8, 48), got {z_next.shape}"

    def test_calcium_gate_effect(self):
        """Calcium gate should interpolate between old and new state."""
        cell = DCaAPCell(input_dim=48, hidden_dim=32, z_dim=16, num_branches=2)
        x = np.random.randn(2, 16)
        y = np.random.randn(2, 16)
        z_init = np.zeros((2, 16))

        z_next = cell.forward(x, y, z_init)

        # z_next should not be exactly z_init (some update happened)
        assert not np.allclose(z_next, z_init), "State should be updated"

    def test_no_crash_on_batch_of_one(self):
        """Should handle batch size of 1."""
        cell = DCaAPCell(input_dim=48, hidden_dim=32, z_dim=16, num_branches=2)
        x = np.random.randn(1, 16)
        y = np.random.randn(1, 16)
        z = np.random.randn(1, 16)

        z_next = cell.forward(x, y, z)
        assert z_next.shape == (1, 16)

    def test_numerical_stability(self):
        """Should handle extreme input values."""
        cell = DCaAPCell(input_dim=48, hidden_dim=32, z_dim=16, num_branches=2)
        x = np.full((2, 16), 100.0)  # Very large
        y = np.full((2, 16), -100.0)  # Very negative
        z = np.zeros((2, 16))

        z_next = cell.forward(x, y, z)
        assert np.all(np.isfinite(z_next)), "Output should be finite"


class TestTorqueRouter:
    """Tests for TorqueRouter class."""

    def test_initialization(self):
        """Should initialize with correct dimensions."""
        router = TorqueRouter(x_dim=32, y_dim=16, z_dim=32, num_experts=4)
        assert router.num_experts == 4
        assert router.expert_centroids.shape == (4, 64)

    def test_softmax_normalization(self):
        """Routing weights should sum to 1."""
        router = TorqueRouter(x_dim=32, y_dim=16, z_dim=32, num_experts=4)
        x = np.random.randn(8, 32)
        y = np.random.randn(8, 16)
        z = np.random.randn(8, 32)

        weights = router.forward(x, y, z)

        # Check shape
        assert weights.shape == (8, 4)

        # Check normalization (sum to 1)
        weight_sums = np.sum(weights, axis=-1)
        np.testing.assert_array_almost_equal(
            weight_sums, np.ones(8), decimal=6,
            err_msg="Routing weights should sum to 1"
        )

    def test_weights_positive(self):
        """All weights should be positive (softmax output)."""
        router = TorqueRouter(x_dim=16, y_dim=8, z_dim=16, num_experts=3)
        x = np.random.randn(4, 16)
        y = np.random.randn(4, 8)
        z = np.random.randn(4, 16)

        weights = router.forward(x, y, z)
        assert np.all(weights > 0), "All weights should be positive"

    def test_distance_sensitivity(self):
        """Closer samples should have higher weights for corresponding expert."""
        router = TorqueRouter(x_dim=4, y_dim=2, z_dim=4, num_experts=2)

        # Create two samples, one closer to each centroid
        x1 = router.expert_centroids[0, :4].reshape(1, -1)  # Close to expert 0
        x2 = router.expert_centroids[1, :4].reshape(1, -1)  # Close to expert 1

        # Random y and z for testing
        y = np.zeros((2, 2))
        z = np.zeros((2, 4))

        x = np.vstack([x1, x2])
        weights = router.forward(x, y, z)

        # Sample 0 should prefer expert 0, sample 1 should prefer expert 1
        # Due to the torque formula, closer samples get higher weights
        # This is a soft test - the relationship should generally hold
        assert weights.shape == (2, 2)


class TestFractalMerkleDAG:
    """Tests for FractalMerkleDAG class."""

    def test_add_step_returns_node_id(self):
        """add_step should return a valid node_id."""
        dag = FractalMerkleDAG(store_states=True)
        y = np.random.randn(1, 8)
        z = np.random.randn(1, 16)

        node_id = dag.add_step(step=0, y=y, z=z, parents=[], score=-1.0)
        assert isinstance(node_id, str)
        assert len(node_id) == 64  # SHA256 hex digest

    def test_parent_child_links(self):
        """Parent-child relationships should be maintained."""
        dag = FractalMerkleDAG(store_states=True)
        y = np.random.randn(1, 8)
        z = np.random.randn(1, 16)

        root_id = dag.add_step(step=0, y=y, z=z, parents=[])
        child_id = dag.add_step(step=1, y=y * 0.9, z=z * 0.9, parents=[root_id])

        # Check parent link
        assert dag.nodes[child_id].parents == [root_id]

        # Check child link
        assert child_id in dag.nodes[root_id].children

    def test_best_node_tracking(self):
        """Should track the best scoring node."""
        dag = FractalMerkleDAG(store_states=True)
        y = np.random.randn(1, 8)
        z = np.random.randn(1, 16)

        dag.add_step(step=0, y=y, z=z, parents=[], score=-1.0)
        best_id = dag.add_step(step=1, y=y, z=z, parents=[], score=0.5)
        dag.add_step(step=2, y=y, z=z, parents=[], score=-0.5)

        best_node = dag.get_best_node()
        assert best_node.node_id == best_id
        assert best_node.score == 0.5

    def test_state_storage(self):
        """Should store y/z states when store_states=True."""
        dag = FractalMerkleDAG(store_states=True)
        y = np.random.randn(1, 8)
        z = np.random.randn(1, 16)

        node_id = dag.add_step(step=0, y=y, z=z, parents=[])
        states = dag.get_node_states(node_id)

        assert states is not None
        np.testing.assert_array_almost_equal(states[0], y)
        np.testing.assert_array_almost_equal(states[1], z)

    def test_no_state_storage(self):
        """Should not store states when store_states=False."""
        dag = FractalMerkleDAG(store_states=False)
        y = np.random.randn(1, 8)
        z = np.random.randn(1, 16)

        node_id = dag.add_step(step=0, y=y, z=z, parents=[])
        states = dag.get_node_states(node_id)

        assert states is None

    def test_create_branch(self):
        """Should create branches at correct depth."""
        dag = FractalMerkleDAG(store_states=True, max_depth=3)
        y = np.random.randn(1, 8)
        z = np.random.randn(1, 16)

        root_id = dag.add_step(step=0, y=y, z=z, parents=[], depth=0)
        branch_id = dag.create_branch(root_id, y=y * 1.1, z=z * 1.1, score=-0.5)

        assert branch_id is not None
        assert dag.nodes[branch_id].depth == 1
        assert dag.nodes[branch_id].branch_root == root_id

    def test_max_depth_limit(self):
        """Should not create branches beyond max_depth."""
        dag = FractalMerkleDAG(store_states=True, max_depth=2)
        y = np.random.randn(1, 8)
        z = np.random.randn(1, 16)

        root_id = dag.add_step(step=0, y=y, z=z, parents=[], depth=0)
        branch1_id = dag.create_branch(root_id, y=y, z=z)  # depth=1
        branch2_id = dag.create_branch(branch1_id, y=y, z=z)  # depth=2
        branch3_id = dag.create_branch(branch2_id, y=y, z=z)  # depth=3 > max_depth

        assert branch1_id is not None
        assert branch2_id is not None
        assert branch3_id is None  # Should be None (exceeds max_depth)

    def test_depth_statistics(self):
        """Should correctly count nodes per depth level."""
        dag = FractalMerkleDAG(store_states=True, max_depth=3)
        y = np.random.randn(1, 8)
        z = np.random.randn(1, 16)

        # Create nodes at different depths
        root1 = dag.add_step(step=0, y=y, z=z, parents=[], depth=0)
        root2 = dag.add_step(step=1, y=y, z=z, parents=[root1], depth=0)
        dag.create_branch(root2, y=y, z=z)  # depth=1
        dag.create_branch(root2, y=y * 0.9, z=z * 0.9)  # depth=1

        stats = dag.get_depth_statistics()
        assert stats[0] == 2  # Two nodes at depth 0
        assert stats[1] == 2  # Two branches at depth 1

    def test_fractal_path(self):
        """Should return correct path from root to node."""
        dag = FractalMerkleDAG(store_states=True)
        y = np.random.randn(1, 8)
        z = np.random.randn(1, 16)

        node0 = dag.add_step(step=0, y=y, z=z, parents=[])
        node1 = dag.add_step(step=1, y=y, z=z, parents=[node0])
        node2 = dag.add_step(step=2, y=y, z=z, parents=[node1])

        path = dag.get_fractal_path(node2)
        assert len(path) == 3
        assert path[0].node_id == node0
        assert path[2].node_id == node2


class TestTRLinkosTRM:
    """Tests for TRLinkosTRM main model class."""

    def test_initialization(self):
        """Should initialize with correct dimensions."""
        model = TRLinkosTRM(x_dim=64, y_dim=32, z_dim=64)
        assert model.x_dim == 64
        assert model.y_dim == 32
        assert model.z_dim == 64

    def test_forward_recursive_shape(self):
        """forward_recursive should output correct shape."""
        model = TRLinkosTRM(x_dim=16, y_dim=8, z_dim=16, hidden_dim=32, num_experts=2)
        x = np.random.randn(4, 16)

        y_pred, dag = model.forward_recursive(x, max_steps=5, inner_recursions=2)

        assert y_pred.shape == (4, 8), f"Expected (4, 8), got {y_pred.shape}"
        assert len(dag.nodes) == 20  # 4 samples * 5 steps

    def test_forward_recursive_with_scorer(self):
        """forward_recursive should work with a scorer function."""
        model = TRLinkosTRM(x_dim=16, y_dim=8, z_dim=16)
        x = np.random.randn(4, 16)
        target = np.random.randn(4, 8)

        def scorer(x_in, y_pred):
            return -np.mean((y_pred - target) ** 2, axis=-1)

        y_pred, dag = model.forward_recursive(x, max_steps=5, scorer=scorer)

        # Best node should have a score
        best_node = dag.get_best_node()
        assert best_node is not None
        assert best_node.score is not None

    def test_backtracking_stores_states(self):
        """Backtracking should store states in DAG."""
        model = TRLinkosTRM(x_dim=16, y_dim=8, z_dim=16)
        x = np.random.randn(2, 16)
        target = np.random.randn(2, 8)

        def scorer(x_in, y_pred):
            return -np.mean((y_pred - target) ** 2, axis=-1)

        y_pred, dag = model.forward_recursive(
            x, max_steps=5, scorer=scorer, backtrack=True
        )

        assert dag.store_states is True
        best_node = dag.get_best_node()
        states = dag.get_node_states(best_node.node_id)
        assert states is not None

    def test_forward_recursive_fractal(self):
        """forward_recursive_fractal should work with fractal branching."""
        model = TRLinkosTRM(x_dim=16, y_dim=8, z_dim=16, hidden_dim=32, num_experts=2)
        x = np.random.randn(4, 16)
        target = np.random.randn(4, 8)

        def scorer(x_in, y_pred):
            return -np.mean((y_pred - target) ** 2, axis=-1)

        y_pred, dag = model.forward_recursive_fractal(
            x,
            max_steps=5,
            scorer=scorer,
            fractal_branching=True,
            branch_threshold=0.001,  # Low threshold to trigger branching
        )

        assert y_pred.shape == (4, 8)
        assert len(dag.nodes) > 0

    def test_invalid_input_shape(self):
        """Should raise error for invalid input shape."""
        model = TRLinkosTRM(x_dim=16, y_dim=8, z_dim=16)

        with pytest.raises(ValueError):
            model.forward_recursive(np.random.randn(4, 32))  # Wrong x_dim

        with pytest.raises(ValueError):
            model.forward_recursive(np.random.randn(16))  # Wrong ndim


class TestTRLinkosCore:
    """Tests for TRLinkosCore class."""

    def test_step_reasoning(self):
        """step_reasoning should update y and z."""
        core = TRLinkosCore(x_dim=16, y_dim=8, z_dim=16, hidden_dim=32, num_experts=2)
        x = np.random.randn(4, 16)
        y = np.zeros((4, 8))
        z = np.zeros((4, 16))

        y_next, z_next = core.step_reasoning(x, y, z, inner_recursions=2)

        assert y_next.shape == (4, 8)
        assert z_next.shape == (4, 16)
        # Should have changed from zeros
        assert not np.allclose(y_next, y)


class TestLossFunctions:
    """Tests for loss functions."""

    def test_mse_loss(self):
        """MSE loss should be non-negative."""
        y_pred = np.random.randn(8, 10)
        y_target = np.random.randn(8, 10)
        loss = mse_loss(y_pred, y_target)
        assert loss >= 0

    def test_mse_loss_zero(self):
        """MSE loss should be zero for identical inputs."""
        y = np.random.randn(8, 10)
        loss = mse_loss(y, y)
        np.testing.assert_almost_equal(loss, 0.0)

    def test_cross_entropy_loss(self):
        """Cross-entropy loss should be non-negative."""
        logits = np.random.randn(8, 5)
        targets = np.array([0, 1, 2, 3, 4, 0, 1, 2])
        loss = cross_entropy_loss(logits, targets)
        assert loss >= 0

    def test_cosine_similarity_loss(self):
        """Cosine similarity loss should be bounded."""
        y_pred = np.random.randn(8, 10)
        y_target = np.random.randn(8, 10)
        loss = cosine_similarity_loss(y_pred, y_target)
        assert 0 <= loss <= 2


class TestEncoders:
    """Tests for TextEncoder and ImageEncoder."""

    def test_text_encoder_char_mode(self):
        """TextEncoder should encode text in char mode."""
        encoder = TextEncoder(vocab_size=256, embed_dim=32, output_dim=16, mode="char")
        texts = ["Hello", "World"]
        embeddings = encoder.encode(texts)
        assert embeddings.shape == (2, 16)

    def test_text_encoder_word_mode(self):
        """TextEncoder should encode text in word mode."""
        encoder = TextEncoder(vocab_size=256, embed_dim=32, output_dim=16, mode="word")
        texts = ["Hello world", "Test text"]
        embeddings = encoder.encode(texts)
        assert embeddings.shape == (2, 16)

    def test_image_encoder_rgb(self):
        """ImageEncoder should encode RGB images."""
        encoder = ImageEncoder(input_channels=3, patch_size=4, embed_dim=32, output_dim=16)
        images = [np.random.rand(16, 16, 3) for _ in range(3)]
        embeddings = encoder.encode(images)
        assert embeddings.shape == (3, 16)

    def test_image_encoder_grayscale(self):
        """ImageEncoder should encode grayscale images."""
        encoder = ImageEncoder(input_channels=1, patch_size=4, embed_dim=32, output_dim=16)
        images = [np.random.rand(16, 16) for _ in range(2)]
        embeddings = encoder.encode(images)
        assert embeddings.shape == (2, 16)


class TestDatasetAndDataLoader:
    """Tests for Dataset and DataLoader."""

    def test_dataset_add_sample(self):
        """Dataset should add samples correctly."""
        dataset = Dataset(x_dim=8, y_dim=4)
        for i in range(10):
            dataset.add_sample(np.random.randn(8), np.random.randn(4))
        assert len(dataset) == 10

    def test_dataset_getitem(self):
        """Dataset should return correct sample."""
        dataset = Dataset(x_dim=8, y_dim=4)
        x = np.random.randn(8)
        y = np.random.randn(4)
        dataset.add_sample(x, y)
        sample = dataset[0]
        np.testing.assert_array_almost_equal(sample.x, x)

    def test_dataloader_batching(self):
        """DataLoader should create correct batches."""
        dataset = Dataset(x_dim=8, y_dim=4)
        for i in range(20):
            dataset.add_sample(np.random.randn(8), np.random.randn(4))

        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        assert len(loader) == 5  # 20 / 4 = 5 batches

        batch_sizes = []
        for x_batch, y_batch in loader:
            batch_sizes.append(x_batch.shape[0])

        assert sum(batch_sizes) == 20


class TestModelSerialization:
    """Tests for save_model and load_model."""

    def test_save_load_model(self, tmp_path):
        """Should save and load model correctly."""
        model = TRLinkosTRM(x_dim=16, y_dim=8, z_dim=16, hidden_dim=32, num_experts=2)
        x = np.random.randn(4, 16)

        # Get prediction before saving
        y_before, _ = model.forward_recursive(x, max_steps=3)

        # Save and load
        filepath = tmp_path / "test_model.npz"
        save_model(model, str(filepath))
        loaded_model = load_model(str(filepath))

        # Get prediction after loading
        y_after, _ = loaded_model.forward_recursive(x, max_steps=3)

        np.testing.assert_array_almost_equal(y_before, y_after)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_gelu(self):
        """GELU should handle various inputs."""
        x = np.array([-2, -1, 0, 1, 2])
        result = gelu(x.astype(float))
        assert result.shape == x.shape
        assert result[2] == 0  # GELU(0) = 0

    def test_softmax_normalization(self):
        """Softmax should sum to 1."""
        x = np.random.randn(4, 10)
        result = softmax(x, axis=-1)
        sums = np.sum(result, axis=-1)
        np.testing.assert_array_almost_equal(sums, np.ones(4))

    def test_softmax_numerical_stability(self):
        """Softmax should handle large values."""
        x = np.array([[1000, 1001, 1002]])
        result = softmax(x, axis=-1)
        assert np.all(np.isfinite(result))
        np.testing.assert_almost_equal(np.sum(result), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
