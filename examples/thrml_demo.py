#!/usr/bin/env python3
"""
THRML Demo: Simple Ising Model Sampling

This example demonstrates basic usage of the THRML library
for sampling from an Ising model using block Gibbs sampling.
"""

import sys
import os

# Add parent directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init


def main():
    """Run a simple THRML Ising model sampling demo."""
    print("=" * 70)
    print("THRML Demo: Ising Model Sampling")
    print("=" * 70)
    print()

    # Create a simple Ising chain with 5 nodes
    print("1. Creating Ising chain with 5 nodes...")
    nodes = [SpinNode() for _ in range(5)]
    edges = [(nodes[i], nodes[i + 1]) for i in range(4)]

    # Define model parameters
    biases = jnp.zeros((5,))  # No bias on individual spins
    weights = jnp.ones((4,)) * 0.5  # Coupling strength between neighbors
    beta = jnp.array(1.0)  # Inverse temperature

    # Create the Ising model
    model = IsingEBM(nodes, edges, biases, weights, beta)
    print(f"   - Nodes: {len(nodes)}")
    print(f"   - Edges: {len(edges)}")
    print(f"   - Weights: {weights}")
    print(f"   - Beta (inverse temperature): {beta}")
    print()

    # Define free blocks for two-color Gibbs sampling
    # Blocks alternate: [0, 2, 4] and [1, 3]
    print("2. Setting up two-color block Gibbs sampling...")
    free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
    print(f"   - Block 1 (red): nodes {[0, 2, 4]}")
    print(f"   - Block 2 (black): nodes {[1, 3]}")
    print()

    # Create sampling program
    program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
    print("3. Initializing sampling program...")
    print()

    # Initialize the chain
    print("4. Initializing chain state...")
    key = jax.random.key(42)  # Fixed seed for reproducibility
    k_init, k_samp = jax.random.split(key, 2)
    init_state = hinton_init(k_init, model, free_blocks, ())
    print("   - Initial state created")
    print()

    # Define sampling schedule
    print("5. Setting up sampling schedule...")
    schedule = SamplingSchedule(
        n_warmup=100,  # Burn-in samples
        n_samples=1000,  # Number of samples to collect
        steps_per_sample=2,  # MCMC steps between samples
    )
    print(f"   - Warmup steps: {schedule.n_warmup}")
    print(f"   - Samples to collect: {schedule.n_samples}")
    print(f"   - Steps per sample: {schedule.steps_per_sample}")
    print()

    # Run sampling
    print("6. Running Gibbs sampling...")
    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
    print(f"   ✓ Sampling complete!")
    print(f"   - Collected {samples[0].shape[0]} samples")
    print(f"   - Sample shape: {samples[0].shape}")
    print()

    # Analyze results
    print("7. Analyzing samples...")
    # Convert boolean samples to -1/+1 spins
    spins = jnp.where(samples[0], 1, -1)

    # Compute magnetization (average spin)
    magnetization = jnp.mean(spins, axis=1)
    print(f"   - Average magnetization: {jnp.mean(magnetization):.4f}")
    print(f"   - Magnetization std: {jnp.std(magnetization):.4f}")

    # Compute spin correlations
    correlations = jnp.corrcoef(spins.T)
    print(f"   - Correlation matrix shape: {correlations.shape}")
    print("   - Nearest-neighbor correlations:")
    for i in range(len(nodes) - 1):
        print(f"     Node {i} ↔ Node {i+1}: {correlations[i, i+1]:.4f}")
    print()

    # Show sample spin configurations
    print("8. Example spin configurations:")
    print("   (showing first 10 samples)")
    print("   Format: -1 = spin down, +1 = spin up")
    for i in range(min(10, len(spins))):
        spin_str = " ".join([f"{s:+2d}" for s in spins[i]])
        print(f"   Sample {i:2d}: [{spin_str}]")
    print()

    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  - Explore examples/01_all_of_thrml.ipynb for advanced features")
    print("  - Try examples/02_spin_models.ipynb for more complex models")
    print("  - Read THRML_INTEGRATION.md for integration with TRLinkosTRM")
    print()


if __name__ == "__main__":
    main()
