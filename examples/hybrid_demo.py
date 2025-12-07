#!/usr/bin/env python3
"""
Hybrid Demo: TRLinkosTRM + THRML Integration

This example demonstrates how to combine TRLinkosTRM's deterministic
recursive reasoning with THRML's probabilistic sampling for hybrid inference.
"""

import sys
import os

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp
from t_rlinkos_trm_fractal_dag import TRLinkosTRM
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init


def main():
    """Demonstrate hybrid TRLinkosTRM + THRML inference."""
    print("=" * 70)
    print("Hybrid Demo: TRLinkosTRM + THRML")
    print("=" * 70)
    print()

    # =========================================================================
    # Part 1: TRLinkosTRM Deterministic Reasoning
    # =========================================================================
    print("PART 1: TRLinkosTRM Deterministic Recursive Reasoning")
    print("-" * 70)

    # Create TRLinkosTRM model
    print("1. Creating TRLinkosTRM model...")
    trm_model = TRLinkosTRM(x_dim=64, y_dim=32, z_dim=64, hidden_dim=128, num_experts=4)
    print(f"   - Input dim: {trm_model.x_dim}")
    print(f"   - Output dim: {trm_model.y_dim}")
    print(f"   - Hidden dim: {trm_model.z_dim}")
    print(f"   - Experts: {trm_model.core.num_experts}")
    print()

    # Generate random input
    print("2. Running deterministic forward pass...")
    np.random.seed(42)
    x_input = np.random.randn(4, 64)
    y_output, dag = trm_model.forward_recursive(x_input, max_steps=5)

    print(f"   - Input shape: {x_input.shape}")
    print(f"   - Output shape: {y_output.shape}")
    print(f"   - DAG nodes: {len(dag.nodes)}")
    print(f"   - Reasoning steps: 5")
    print()

    # Show output statistics
    print("3. Output statistics:")
    print(f"   - Mean: {np.mean(y_output):.4f}")
    print(f"   - Std: {np.std(y_output):.4f}")
    print(f"   - Min: {np.min(y_output):.4f}")
    print(f"   - Max: {np.max(y_output):.4f}")
    print()

    # =========================================================================
    # Part 2: THRML Probabilistic Sampling
    # =========================================================================
    print("PART 2: THRML Probabilistic Sampling")
    print("-" * 70)

    # Create Ising model
    print("1. Creating Ising model for probabilistic inference...")
    n_nodes = 8
    nodes = [SpinNode() for _ in range(n_nodes)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(n_nodes - 1)]

    # Use TRLinkosTRM output to influence Ising model parameters
    # (In a real application, you might use TRM output to set biases/weights)
    biases = jnp.zeros((n_nodes,))
    weights = jnp.ones((n_nodes - 1,)) * 0.3
    beta = jnp.array(1.5)

    model = IsingEBM(nodes, edges, biases, weights, beta)
    print(f"   - Nodes: {n_nodes}")
    print(f"   - Edges: {n_nodes - 1}")
    print(f"   - Beta: {beta}")
    print()

    # Set up sampling
    print("2. Setting up block Gibbs sampling...")
    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    print(f"   - Block 1: nodes {list(range(0, n_nodes, 2))}")
    print(f"   - Block 2: nodes {list(range(1, n_nodes, 2))}")
    print()

    # Sample
    print("3. Running probabilistic sampling...")
    key = jax.random.key(42)
    k_init, k_samp = jax.random.split(key, 2)
    init_state = hinton_init(k_init, model, free_blocks, ())
    schedule = SamplingSchedule(n_warmup=50, n_samples=200, steps_per_sample=2)

    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
    spins = jnp.where(samples[0], 1, -1)

    print(f"   - Samples collected: {samples[0].shape[0]}")
    print(f"   - Sample shape: {samples[0].shape}")
    print()

    # Analyze samples
    print("4. Sample statistics:")
    magnetization = jnp.mean(spins, axis=1)
    print(f"   - Mean magnetization: {jnp.mean(magnetization):.4f}")
    print(f"   - Magnetization std: {jnp.std(magnetization):.4f}")
    print()

    # =========================================================================
    # Part 3: Hybrid Analysis
    # =========================================================================
    print("PART 3: Hybrid Analysis")
    print("-" * 70)

    print("Combining deterministic and probabilistic reasoning:")
    print()

    # Show how outputs could be combined
    print("1. TRLinkosTRM provides deterministic reasoning:")
    print("   - Fast forward passes")
    print("   - Explicit reasoning traces in DAG")
    print("   - Hierarchical expert routing")
    print()

    print("2. THRML provides probabilistic inference:")
    print("   - Uncertainty quantification")
    print("   - Alternative hypotheses exploration")
    print("   - Energy-based model optimization")
    print()

    print("3. Hybrid approaches enable:")
    print("   - Using TRM output to parameterize THRML models")
    print("   - THRML sampling to explore TRM reasoning paths")
    print("   - Energy-based expert selection in TRM")
    print("   - Probabilistic backtracking in reasoning DAG")
    print()

    # Example: Use magnetization variance as uncertainty estimate
    uncertainty = float(jnp.std(magnetization))
    print(f"4. Example uncertainty metric: {uncertainty:.4f}")
    print("   (Higher variance = more uncertain predictions)")
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 70)
    print("Hybrid Demo Complete!")
    print("=" * 70)
    print()
    print("Key Insights:")
    print("  • TRLinkosTRM: Deterministic reasoning with expert mixture")
    print("  • THRML: Probabilistic sampling with energy-based models")
    print("  • Hybrid: Combines best of both approaches")
    print()
    print("Next Steps:")
    print("  • Explore THRML_INTEGRATION.md for advanced patterns")
    print("  • Implement custom hybrid architectures")
    print("  • Use THRML EBMs for TRM expert routing")
    print("  • Apply probabilistic sampling to reasoning traces")
    print()


if __name__ == "__main__":
    main()
