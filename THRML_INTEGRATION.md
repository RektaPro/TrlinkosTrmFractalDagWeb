# THRML Integration with TRLinkosTRM

## Overview

This repository now integrates the **THRML (Thermodynamic Hypergraphical Model Library)** from Extropic AI, providing a powerful JAX-based framework for building and sampling probabilistic graphical models alongside the existing TRLinkosTRM (Tiny Recursive Linkos Model) implementation.

## What is THRML?

THRML is a JAX library developed by Extropic AI for building and sampling probabilistic graphical models (PGMs), with a particular focus on:

- **Blocked Gibbs Sampling** for efficient inference on PGMs
- **Energy-Based Models (EBMs)** including Ising models and Restricted Boltzmann Machines
- **Thermodynamic Computing** - designed to prototype future hardware that performs probabilistic computations more energy efficiently
- **GPU-Accelerated Block Sampling** using JAX for maximum parallelism
- **Heterogeneous Graphical Models** with arbitrary PyTree node states

## Integration Architecture

### Directory Structure

```
TrlinkosTrmFractalDagWeb/
├── thrml/                          # THRML library (new)
│   ├── __init__.py
│   ├── block_management.py         # Block organization and state management
│   ├── block_sampling.py           # Core Gibbs sampling algorithms
│   ├── conditional_samplers.py     # Sampling strategies (Bernoulli, Softmax)
│   ├── factor.py                   # Factor-based PGM representations
│   ├── interaction.py              # Node interactions in PGMs
│   ├── observers.py                # Sampling observation and metrics
│   ├── pgm.py                      # Base PGM node classes
│   ├── py.typed                    # Type checking marker
│   └── models/                     # Pre-built models
│       ├── __init__.py
│       ├── discrete_ebm.py         # Discrete energy-based models
│       ├── ebm.py                  # General EBM interface
│       └── ising.py                # Ising model implementations
├── examples/                        # Examples (existing + THRML)
│   ├── 00_probabilistic_computing.ipynb  # THRML: Intro to probabilistic computing
│   ├── 01_all_of_thrml.ipynb            # THRML: Complete library tour
│   ├── 02_spin_models.ipynb             # THRML: Ising/spin systems
│   └── blueprints_demo.py               # TRLinkosTRM: Blueprint patterns
├── tests/                          # Tests (existing + THRML)
│   ├── test_thrml_*.py             # THRML test suite
│   └── test_*.py                   # TRLinkosTRM tests
├── t_rlinkos_trm_fractal_dag.py   # Main TRLinkosTRM implementation
├── THRML_README.md                 # THRML documentation
├── THRML_LICENSE                   # THRML Apache 2.0 license
└── THRML_INTEGRATION.md           # This file
```

## Complementary Technologies

### TRLinkosTRM (Existing)
- **Framework**: Pure NumPy with optional PyTorch
- **Focus**: Recursive reasoning with dendritic neurons (dCaAP)
- **Architecture**: Mixture of Experts with Torque Clustering
- **Features**: Fractal DAG reasoning traces, neuromorphic computing
- **Use Cases**: Hierarchical reasoning, multi-step decision making

### THRML (New)
- **Framework**: JAX (functional, composable)
- **Focus**: Probabilistic graphical models and energy-based inference
- **Architecture**: Block Gibbs sampling on heterogeneous graphs
- **Features**: Thermodynamic sampling, discrete EBMs
- **Use Cases**: Probabilistic inference, sampling, optimization

## Combined Use Cases

The integration of THRML with TRLinkosTRM enables powerful hybrid architectures:

1. **Probabilistic Reasoning Layers**: Use THRML's EBMs as uncertainty-aware components within TRLinkosTRM's recursive reasoning
2. **Energy-Based Expert Selection**: Replace or augment Torque Clustering with THRML's energy-based routing
3. **Thermodynamic Backtracking**: Use THRML's sampling for exploring alternative reasoning paths in the Fractal DAG
4. **Hybrid Optimization**: Combine deterministic TRLinkosTRM forward passes with probabilistic THRML sampling for better exploration

## Getting Started

### Installation

Install all dependencies including THRML requirements:

```bash
pip install -r requirements.txt
```

This installs:
- Core dependencies (NumPy)
- THRML dependencies (JAX, Equinox, JaxTyping)
- Optional: Numba, PyTorch, Transformers, ONNX

### Quick Start: THRML

```python
import jax
import jax.numpy as jnp
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

# Create an Ising chain
nodes = [SpinNode() for _ in range(5)]
edges = [(nodes[i], nodes[i+1]) for i in range(4)]
biases = jnp.zeros((5,))
weights = jnp.ones((4,)) * 0.5
beta = jnp.array(1.0)
model = IsingEBM(nodes, edges, biases, weights, beta)

# Define sampling blocks (two-color)
free_blocks = [Block(nodes[::2]), Block(nodes[1::2])]
program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

# Sample from the model
key = jax.random.key(0)
k_init, k_samp = jax.random.split(key, 2)
init_state = hinton_init(k_init, model, free_blocks, ())
schedule = SamplingSchedule(n_warmup=100, n_samples=1000, steps_per_sample=2)

samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
print(f"Sampled {samples[0].shape[0]} states")
```

### Quick Start: TRLinkosTRM

```python
from t_rlinkos_trm_fractal_dag import TRLinkosTRM
import numpy as np

# Create TRLinkosTRM model
model = TRLinkosTRM(input_dim=64, hidden_dim=32, output_dim=64)

# Forward pass with recursive reasoning
x = np.random.randn(8, 64)
output, reasoning_trace = model.forward(x, max_steps=10, return_trace=True)
print(f"Output shape: {output.shape}")
print(f"Reasoning steps: {len(reasoning_trace)}")
```

## Example Notebooks

Explore the THRML examples to understand probabilistic computing:

1. **00_probabilistic_computing.ipynb**: Introduction to probabilistic computing principles
2. **01_all_of_thrml.ipynb**: Comprehensive tour of THRML's capabilities
3. **02_spin_models.ipynb**: Working with Ising models and spin systems

## Testing

Run THRML tests:

```bash
# Run specific THRML tests
pytest tests/test_thrml_ising.py
pytest tests/test_thrml_block_sampling.py

# Run all THRML tests
pytest tests/test_thrml_*.py
```

Run TRLinkosTRM tests:

```bash
python run_all_tests.py
```

## Advanced Integration Patterns

### 1. Probabilistic Expert Routing

Use THRML's energy-based models to implement probabilistic routing in TRLinkosTRM's MoE architecture:

```python
from t_rlinkos_trm_fractal_dag import TRLinkosTRM
from thrml.models import IsingEBM
import jax.numpy as jnp

# Define energy-based router
def create_energy_router(n_experts, n_features):
    nodes = [SpinNode() for _ in range(n_experts)]
    # Energy function based on input features
    biases = jnp.zeros(n_experts)
    model = IsingEBM(nodes, [], biases, [], beta=1.0)
    return model

# Integrate with TRLinkosTRM
trm_model = TRLinkosTRM(64, 32, 64)
energy_router = create_energy_router(n_experts=trm_model.n_experts, n_features=64)
```

### 2. Thermodynamic Reasoning Traces

Enhance TRLinkosTRM's Fractal DAG with probabilistic sampling:

```python
from thrml import sample_states, SamplingSchedule

# After TRLinkosTRM reasoning step
reasoning_state = trm_model.forward(x, max_steps=5)

# Sample alternative reasoning paths using THRML
# Convert reasoning state to THRML-compatible format
# Sample alternative paths for exploration
```

### 3. Hybrid Training

Combine deterministic backpropagation (TRLinkosTRM) with probabilistic sampling (THRML):

```python
# Deterministic forward pass
output = trm_model.forward(x)

# Probabilistic exploration with THRML
# Use samples to augment training data or guide search
```

## Performance Considerations

- **THRML**: GPU-accelerated via JAX, optimal for parallel block sampling
- **TRLinkosTRM**: CPU-optimized with Numba, or GPU via PyTorch
- **Hybrid**: Consider computational device placement for optimal performance

## Architecture Patterns

Both libraries support enterprise-ready patterns:

### TRLinkosTRM Blueprints
- Safety Guardrails Pattern
- AI Observability Pattern  
- Resilient Workflow Pattern
- Goal Monitoring Pattern

### THRML Design Patterns
- Block Gibbs Sampling
- Factor Graph Representations
- Observer Pattern for Metrics
- Heterogeneous Node Systems

## References

### THRML
- **Repository**: https://github.com/extropic-ai/thrml
- **Documentation**: https://docs.thrml.ai/
- **Paper**: Jelinčič et al., "An efficient probabilistic hardware architecture for diffusion-like models", arXiv:2510.23972, 2025

### TRLinkosTRM
- **Gidon et al.**: "Dendritic action potentials and computation in human layer 2/3 cortical neurons", Science, 2020
- **Hashemi & Tetzlaff**: "Cortical dendrites enhance learning through dendritic democracy", bioRxiv, 2025
- **Yang & Lin**: "Torque Clustering: A Novel Method for Clustering Data", TPAMI, 2025

## License

- **THRML**: Apache License 2.0 (see THRML_LICENSE)
- **TRLinkosTRM**: BSD-3-Clause (see LICENSE)

Both libraries are open-source and can be used together in compatible projects.

## Contributing

Contributions to either library are welcome:

- **THRML**: See THRML_CONTRIBUTING.md
- **TRLinkosTRM**: See CONTRIBUTING.md

## Future Directions

1. **Hardware Acceleration**: Leverage Extropic's thermodynamic sampling units when available
2. **Unified API**: Develop higher-level abstractions combining both libraries
3. **Benchmarks**: Comparative studies on reasoning + probabilistic inference tasks
4. **Hybrid Architectures**: New model architectures combining recursive reasoning and energy-based sampling

## Support

- **Issues**: Report issues specific to THRML integration via GitHub issues
- **Discussions**: Join discussions about hybrid architectures and use cases
- **Documentation**: Refer to THRML_README.md for THRML-specific documentation

---

*Last Updated: December 7, 2025*
