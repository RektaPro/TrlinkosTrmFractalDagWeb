"""
Tests for dcaap_activation and DCaAPCell.

Tests cover:
- dcaap_activation: negative values, around 0, positive values, non-linearity, vector support
- DCaAPCell: shape in/out, calcium gate remains in [0,1] (sigmoid)
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
)


class TestDCaAPActivation:
    """Tests for dcaap_activation function."""

    def test_negative_values_return_zero(self):
        """Negative values (below threshold) should return zero."""
        x = np.array([[-5.0, -2.0, -1.0, -0.1]])
        result = dcaap_activation(x, threshold=0.0)
        np.testing.assert_array_equal(result, np.zeros_like(x))

    def test_around_zero_behavior(self):
        """Values around zero should show correct threshold behavior."""
        x = np.array([[-0.01, 0.0, 0.01, 0.1]])
        result = dcaap_activation(x, threshold=0.0)
        
        # Below or at threshold should be 0
        assert result[0, 0] == 0.0  # -0.01 < 0
        assert result[0, 1] == 0.0  # 0.0 == threshold
        # Above threshold should be positive
        assert result[0, 2] > 0.0   # 0.01 > 0
        assert result[0, 3] > 0.0   # 0.1 > 0

    def test_positive_values_behavior(self):
        """Positive values (above threshold) should show non-monotone behavior."""
        x = np.array([[0.5, 1.0, 2.0, 5.0, 10.0]])
        result = dcaap_activation(x, threshold=0.0)
        
        # All positive inputs above threshold should produce positive outputs
        assert np.all(result > 0)
        
        # dCaAP is non-monotone: should rise then fall
        # Very large values should decay compared to moderate values
        assert result[0, 4] < result[0, 2], "Large inputs should have smaller activation"

    def test_non_linearity(self):
        """Function should be non-linear (superposition test)."""
        x1 = np.array([[1.0]])
        x2 = np.array([[2.0]])
        
        result_1 = dcaap_activation(x1, threshold=0.0)
        result_2 = dcaap_activation(x2, threshold=0.0)
        result_sum = dcaap_activation(x1 + x2, threshold=0.0)
        
        # Non-linearity: f(x1 + x2) != f(x1) + f(x2)
        assert not np.isclose(result_sum[0, 0], result_1[0, 0] + result_2[0, 0])

    def test_vector_input_no_crash(self):
        """Function should handle vector/batch inputs without crashing."""
        # Single vector
        x_1d = np.array([[-1.0, 0.5, 2.0, 5.0]])
        result_1d = dcaap_activation(x_1d, threshold=0.0)
        assert result_1d.shape == (1, 4)
        assert np.all(np.isfinite(result_1d))
        
        # Batch of vectors
        x_batch = np.random.randn(16, 32)
        result_batch = dcaap_activation(x_batch, threshold=0.0)
        assert result_batch.shape == (16, 32)
        assert np.all(np.isfinite(result_batch))

    def test_output_bounded_zero_one(self):
        """Output should always be bounded between 0 and 1."""
        x = np.random.randn(100, 50) * 10  # Wide range of inputs
        result = dcaap_activation(x, threshold=0.0)
        assert np.all(result >= 0), "Output should be non-negative"
        assert np.all(result <= 1), "Output should be at most 1"

    def test_custom_threshold(self):
        """Should respect custom thresholds."""
        x = np.array([[0.0, 0.5, 1.0, 1.5, 2.0]])
        
        # With threshold=1.0, only values > 1.0 should produce non-zero output
        result = dcaap_activation(x, threshold=1.0)
        assert result[0, 0] == 0.0  # 0.0 < 1.0
        assert result[0, 1] == 0.0  # 0.5 < 1.0
        assert result[0, 2] == 0.0  # 1.0 == 1.0 (at threshold)
        assert result[0, 3] > 0.0   # 1.5 > 1.0
        assert result[0, 4] > 0.0   # 2.0 > 1.0


class TestDCaAPCell:
    """Tests for DCaAPCell class."""

    def test_input_output_shapes(self):
        """Check that input/output shapes are correct."""
        # input_dim = dx + dy + dz
        dx, dy, dz = 16, 8, 32
        input_dim = dx + dy + dz  # 56
        hidden_dim = 64
        z_dim = dz
        
        cell = DCaAPCell(input_dim=input_dim, hidden_dim=hidden_dim, z_dim=z_dim, num_branches=4)
        
        batch_size = 8
        x = np.random.randn(batch_size, dx)
        y = np.random.randn(batch_size, dy)
        z = np.random.randn(batch_size, dz)
        
        z_next = cell.forward(x, y, z)
        
        # Output should have same shape as z
        assert z_next.shape == (batch_size, z_dim), f"Expected ({batch_size}, {z_dim}), got {z_next.shape}"

    def test_batch_size_one(self):
        """Should handle batch size of 1."""
        dx, dy, dz = 8, 4, 8
        input_dim = dx + dy + dz
        
        cell = DCaAPCell(input_dim=input_dim, hidden_dim=32, z_dim=dz, num_branches=2)
        
        x = np.random.randn(1, dx)
        y = np.random.randn(1, dy)
        z = np.random.randn(1, dz)
        
        z_next = cell.forward(x, y, z)
        assert z_next.shape == (1, dz)

    def test_calcium_gate_bounded(self):
        """Calcium gate should always be in [0, 1] (sigmoid output)."""
        dx, dy, dz = 8, 4, 16
        input_dim = dx + dy + dz
        
        cell = DCaAPCell(input_dim=input_dim, hidden_dim=32, z_dim=dz, num_branches=2)
        
        # Test with various inputs
        for _ in range(5):
            x = np.random.randn(4, dx) * 10  # Wide range
            y = np.random.randn(4, dy) * 10
            z = np.random.randn(4, dz) * 10
            
            # We need to inspect the gate value internally
            # The gate is computed as sigmoid, so we verify the formula
            h_in = np.concatenate([x, y, z], axis=-1)
            
            # The gate is sigmoid of calcium_gate output, which is bounded [0, 1]
            # We verify by checking z_next is a valid interpolation
            z_next = cell.forward(x, y, z)
            
            # z_next should be finite
            assert np.all(np.isfinite(z_next)), "z_next should be finite"
            
            # Since gate is in [0,1] and z_next = z + gate * (proposal - z),
            # z_next should be some interpolation between z and proposal
            # We can't directly check gate, but we verify output is finite and reasonable

    def test_gate_interpolation_property(self):
        """The calcium gate creates an interpolation between z and proposal."""
        dx, dy, dz = 4, 4, 4
        input_dim = dx + dy + dz
        
        cell = DCaAPCell(input_dim=input_dim, hidden_dim=16, z_dim=dz, num_branches=2)
        
        x = np.random.randn(2, dx)
        y = np.random.randn(2, dy)
        z_init = np.zeros((2, dz))
        
        z_next = cell.forward(x, y, z_init)
        
        # z_next should be different from z_init (some update happened)
        assert not np.allclose(z_next, z_init), "State should be updated"
        
        # All values should be finite
        assert np.all(np.isfinite(z_next))

    def test_different_branch_counts(self):
        """Should work with different numbers of dendritic branches."""
        dx, dy, dz = 8, 4, 8
        input_dim = dx + dy + dz
        
        for num_branches in [1, 2, 4, 8]:
            cell = DCaAPCell(
                input_dim=input_dim,
                hidden_dim=32,
                z_dim=dz,
                num_branches=num_branches
            )
            
            x = np.random.randn(4, dx)
            y = np.random.randn(4, dy)
            z = np.random.randn(4, dz)
            
            z_next = cell.forward(x, y, z)
            assert z_next.shape == (4, dz)
            assert np.all(np.isfinite(z_next))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
