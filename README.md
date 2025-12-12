# T-RLINKOS TRM++ Fractal DAG

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-blue.svg)](https://numpy.org/)
[![CI](https://github.com/RektaPro/TrlinkosTrmFractalDagWeb/workflows/CI/badge.svg)](https://github.com/RektaPro/TrlinkosTrmFractalDagWeb/actions)
[![codecov](https://codecov.io/gh/RektaPro/TrlinkosTrmFractalDagWeb/branch/main/graph/badge.svg)](https://codecov.io/gh/RektaPro/TrlinkosTrmFractalDagWeb)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**T-RLINKOS TRM++ (Tiny Recursive Linkos Model ++)** is a pure NumPy implementation of a recursive reasoning architecture inspired by neuroscience research and modern clustering techniques. Now enhanced with **AI Architecture Blueprints** for production-ready deployments and **integrated with THRML** for probabilistic graphical models.

## 🧠 Overview

This project implements a recursive reasoning model that combines:

- **dCaAP-inspired neurons** (Dendritic Calcium Action Potential) based on [Gidon et al., Science 2020](https://www.science.org/doi/10.1126/science.aax6239) and [Hashemi & Tetzlaff, bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.06.10.658823v1)
- **Torque Clustering router** for expert selection based on [Yang & Lin, TPAMI 2025](https://github.com/JieYangBruce/TorqueClustering)
- **Fractal Merkle-DAG** for reasoning trace auditing and backtracking
- **Mixture of Experts (MoE)** architecture with biologically-inspired activation functions
- **🆕 AI Architecture Blueprints** - Enterprise-grade safety, observability, and resilience
- **🆕 THRML Integration** - JAX-based probabilistic graphical models and energy-based inference

The model is designed to be **framework-agnostic**, using only NumPy for computation, making it portable and easy to understand.

## ⚡ Advanced Features Now Activated!

All 5 advanced features are now **activated** in the repository:

1. ✅ **Numba/JIT Optimization** - Automatic 2-5x acceleration (already enabled!)
2. ✅ **Multi-GPU Support** - Training scalability on multiple GPUs
3. ✅ **HuggingFace Integration** - Pre-trained encoders (BERT, GPT-2, ViT, etc.)
4. ✅ **ONNX Export** - Production deployment for multiplatform use
5. ✅ **Neuromorphic Computing** - Experimental spike-based implementation

**Quick Installation:**
```bash
pip install -r requirements.txt  # Core + Numba + ONNX
pip install torch transformers   # Optional: GPU + HuggingFace models
```

📖 **See [ACTIVATION_GUIDE.md](ACTIVATION_GUIDE.md) for complete usage examples and documentation.**
## 🎯 AI Architecture Blueprints Integration

T-RLINKOS TRM++ now integrates **4 key patterns** from [THE-BLUEPRINTS.md](THE-BLUEPRINTS.md) to provide production-ready AI:

1. **Safety Guardrails Pattern** - Input/output validation and sanitization
2. **AI Observability Pattern** - Real-time metrics and health monitoring
3. **Resilient Workflow Pattern** - Automatic retry and circuit breakers
4. **Goal Monitoring Pattern** - Progress tracking toward objectives

See [BLUEPRINTS_INTEGRATION.md](BLUEPRINTS_INTEGRATION.md) for complete documentation.

### Quick Start with Enhanced TRM

```python
from t_rlinkos_trm_fractal_dag import TRLinkosTRM
from blueprints import EnhancedTRLinkosTRM, EnhancedTRMConfig
import numpy as np

# Create base model
base_model = TRLinkosTRM(64, 32, 64)

# Wrap with enterprise features
config = EnhancedTRMConfig(
    enable_safety_guardrails=True,
    enable_observability=True,
    enable_resilient_workflow=True,
)
model = EnhancedTRLinkosTRM(base_model, config)

# Safe inference with automatic validation and monitoring
x = np.random.randn(8, 64)
result = model.forward_safe(x, max_steps=10)

if result["success"]:
    print(f"✓ Predictions: {result['predictions'].shape}")
    print(f"✓ Latency: {result['metrics']['latency_ms']:.2f}ms")
    print(f"✓ Validation: {result['validation_reports']['input']['result']}")
else:
    print(f"✗ Error: {result['error']}")

# Get metrics dashboard
dashboard = model.get_dashboard()
print(f"Health: {dashboard['health']['is_healthy']}")
```

### Enhanced API Server

```bash
# Start enhanced API with all blueprint features
python api_enhanced.py

# Access endpoints
curl http://localhost:8000/health/detailed   # Detailed health check
curl http://localhost:8000/metrics           # Observability metrics
curl http://localhost:8000/dashboard         # Complete dashboard
```

## 🔌 MCP Integration (Model Context Protocol)

T-RLINKOS now supports the **Model Context Protocol (MCP)**, enabling seamless integration with LLMs and AI agents. The MCP server exposes all reasoning capabilities as tools that can be called by any MCP-compatible client.

### Quick Start with MCP

```bash
# Start the MCP server (stdio mode for LLM integration)
python mcp/server.py --stdio

# Or start HTTP mode for REST API access
python mcp/server.py --http --port 8080
```

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `reason_step` | Execute a single reasoning step with TRLinkosTRM |
| `run_trm_recursive` | Run complete recursive reasoning loop |
| `dag_add_node` | Add a node to the Fractal Merkle-DAG |
| `dag_best_path` | Get the best reasoning path |
| `dag_get_state` | Get DAG statistics and state |
| `torque_route` | Compute expert routing weights |
| `dcaap_forward` | Execute dCaAP cell forward pass |
| `fractal_branch` | Create a fractal branch for exploration |
| `evaluate_score` | Evaluate prediction scores (MSE, cosine, MAE) |
| `load_model` / `save_model` | Model persistence |
| `get_repo_state` / `write_repo_state` | File operations |
| `execute_command` | Execute system commands |
| `get_system_info` | Get system information (OS, Python version, etc.) |
| `list_directory` | List directory contents |
| `get_environment_variable` | Get environment variable values |
| `check_command_exists` | Check if a command exists in PATH |

### MCP Configuration

The MCP manifest (`mcp.json`) defines all available tools and their schemas. Example usage with an MCP client:

```python
# Example: Using MCP tools programmatically
from mcp.server import TRLinkosMCPServer

server = TRLinkosMCPServer(x_dim=64, y_dim=32, z_dim=64)

# Run recursive reasoning
result = server.run_trm_recursive(
    x=[0.1] * 64,
    max_steps=10,
    backtrack=True,
)
print(f"Prediction: {result['y_pred']}")
print(f"DAG nodes: {result['dag_stats']['num_nodes']}")

# Execute system commands
result = server.execute_command("python --version")
print(f"Python version: {result['stdout']}")

# Get system information
result = server.get_system_info()
print(f"OS: {result['system']['os']}")
print(f"Python: {result['system']['python_version']}")
```

#### System Tools

The MCP server now includes system interaction tools that enable:

- **Command Execution**: Run system commands securely with timeout and environment control
- **System Information**: Query OS details, Python version, and environment variables
- **File System Operations**: List directories and check command availability
- **Environment Access**: Read environment variables and check system state

Example system tool usage:

```python
# Check if a command exists
result = server.check_command_exists("python")
if result["exists"]:
    print(f"Python found at: {result['path']}")

# Execute a command with timeout
result = server.execute_command("ls -la", timeout=10)
print(result["stdout"])

# List directory contents
result = server.list_directory("/home/user/project")
for entry in result["entries"]:
    print(f"{entry['name']} - {entry['type']}")
```

### MCP Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP Client (LLM/Agent)                   │
│  (Claude, GPT, Mistral, or any MCP-compatible client)       │
└─────────────────────────┬───────────────────────────────────┘
                          │ JSON-RPC over stdio/HTTP
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    TRLinkosMCPServer                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │ Reasoning Tools │  │   DAG Tools     │  │ Model Tools │  │
│  │ - reason_step   │  │ - dag_add_node  │  │ - load_model│  │
│  │ - run_recursive │  │ - dag_best_path │  │ - save_model│  │
│  │ - torque_route  │  │ - fractal_branch│  └─────────────┘  │
│  │ - dcaap_forward │  └─────────────────┘                   │
│  └─────────────────┘  ┌─────────────────┐  ┌─────────────┐  │
│                       │  System Tools   │  │  File Tools │  │
│                       │ - execute_cmd   │  │ - get_repo  │  │
│                       │ - system_info   │  │ - write_repo│  │
│                       │ - list_dir      │  └─────────────┘  │
│                       └─────────────────┘                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      TRLinkosTRM Core                        │
│  (dCaAP Experts, Torque Router, Fractal Merkle-DAG)         │
└─────────────────────────────────────────────────────────────┘
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRLinkosTRM                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    TRLinkosCore                           │  │
│  │  ┌─────────────┐   ┌─────────────────────────────────┐   │  │
│  │  │TorqueRouter │──▶│     DCaAPCell Experts (MoE)     │   │  │
│  │  │(Mass × R²)  │   │  ┌────────┐ ┌────────┐          │   │  │
│  │  └─────────────┘   │  │Expert 1│ │Expert N│ ...      │   │  │
│  │                    │  │(dCaAP) │ │(dCaAP) │          │   │  │
│  │                    │  └────────┘ └────────┘          │   │  │
│  │                    └─────────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  FractalMerkleDAG                         │  │
│  │  (Reasoning Trace with Backtracking & Auto-Similarity)    │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🔬 Key Components

### 1. dCaAP Activation (`dcaap_activation`)

The **dendritic Calcium Action Potential (dCaAP)** activation is a non-monotonic function that enables anti-coincidence detection and intrinsic XOR capability:

```
dCaAP(x) = 4 × σ(x-θ) × (1 - σ(x-θ)) × (x > θ)
```

Unlike ReLU, a single neuron with dCaAP activation can solve the XOR problem.

### 2. DCaAPCell

A biologically-inspired neuron implementing:
- **Multiple dendritic branches** with local integration
- **Adaptive thresholds** per branch (dendritic heterogeneity)
- **Calcium gate** for temporal accumulation
- **Somatic integration**: dendrites → soma → output

### 3. TorqueRouter

Expert router based on **Torque Clustering** (τ = Mass × R²):
- Computes distance² (R²) to expert centroids
- Calculates local mass (density) for each sample
- Routes inputs to experts via affinity scores: `score = mass / (R² + ε)`

### 4. FractalMerkleDAG

A fractal data structure for reasoning audit:
- **Merkle**: SHA256 hashing of states for integrity
- **DAG**: Directed Acyclic Graph with parent/child links
- **Fractal**: Self-similar structure with recursive branches
- **Backtracking**: State restoration to best-scoring nodes

## 🔥 THRML Integration (NEW!)

This repository now includes **THRML (Thermodynamic Hypergraphical Model Library)**, a JAX-based library from Extropic AI for probabilistic graphical models and energy-based inference.

### What is THRML?

THRML provides:
- **Block Gibbs Sampling** for efficient inference on probabilistic graphical models (PGMs)
- **Energy-Based Models (EBMs)** including Ising models and Restricted Boltzmann Machines
- **GPU-Accelerated Sampling** using JAX for maximum parallelism
- **Heterogeneous Graphical Models** with arbitrary node states
- **Thermodynamic Computing** prototypes for future energy-efficient hardware

### Quick Start with THRML

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

# Define sampling blocks
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

### THRML + TRLinkosTRM Integration

Combine deterministic recursive reasoning (TRLinkosTRM) with probabilistic sampling (THRML):

1. **Probabilistic Expert Routing**: Use THRML's EBMs for uncertainty-aware expert selection
2. **Energy-Based Backtracking**: Sample alternative reasoning paths in the Fractal DAG
3. **Hybrid Optimization**: Combine deterministic forward passes with probabilistic exploration
4. **Thermodynamic Inference**: Leverage energy-based models for robust decision-making

See [THRML_INTEGRATION.md](THRML_INTEGRATION.md) for complete documentation and advanced examples.

### THRML Examples

- **examples/thrml_demo.py**: Simple Ising model sampling demonstration
- **examples/00_probabilistic_computing.ipynb**: Introduction to probabilistic computing
- **examples/01_all_of_thrml.ipynb**: Comprehensive THRML library tour
- **examples/02_spin_models.ipynb**: Working with Ising models and spin systems

## 📦 Installation

### Prerequisites

- Python 3.8+
- NumPy 1.20+

### Install

```bash
# Clone the repository
git clone https://github.com/RektaPro/TrlinkosTrmFractalDag.git
cd TrlinkosTrmFractalDag

# Install core dependencies + activated features (Numba, ONNX)
pip install -r requirements.txt

# Optional: For GPU support and HuggingFace models
pip install torch>=2.0.0
pip install transformers>=4.30.0

# Optional: For web scraping utilities
pip install requests beautifulsoup4
```

**What's included with `pip install -r requirements.txt`:**
- ✅ NumPy (core dependency)
- ✅ JAX + Equinox + JaxTyping (THRML dependencies)
- ✅ Numba (2-5x speedup via JIT compilation)
- ✅ ONNX + ONNX Runtime (model export for production)
- ✅ pytest (testing)
- ✅ FastAPI + Uvicorn (web API)

**Optional packages** (install separately if needed):
- PyTorch: GPU acceleration and multi-GPU training
- Transformers: Pre-trained encoders from HuggingFace (BERT, GPT-2, ViT, etc.)

## 📁 Project Structure

```
TrlinkosTrmFractalDagWeb/
├── t_rlinkos_trm_fractal_dag.py   # Core NumPy implementation
├── trlinkos_trm_torch.py          # PyTorch GPU implementation
├── trlinkos_llm_layer.py          # LLM reasoning layer integration
├── empirical_validation.py        # Empirical validation suite
├── api.py                         # FastAPI web API
├── api_enhanced.py                # 🆕 Enhanced API with blueprints
├── THE-BLUEPRINTS.md              # AI Architecture Blueprints documentation
├── BLUEPRINTS_INTEGRATION.md      # 🆕 Blueprint integration guide
├── blueprints/                    # 🆕 AI Architecture Blueprints package
│   ├── __init__.py
│   ├── safety_guardrails.py       # Input/output validation
│   ├── observability.py           # Metrics and health monitoring
│   ├── resilient_workflow.py      # Retry and circuit breakers
│   ├── goal_monitoring.py         # Progress tracking
│   └── enhanced_trm.py            # Enhanced TRM wrapper
├── thrml/                         # 🆕 THRML library (JAX-based PGMs)
│   ├── __init__.py
│   ├── block_management.py        # Block organization and state management
│   ├── block_sampling.py          # Core Gibbs sampling algorithms
│   ├── conditional_samplers.py    # Sampling strategies
│   ├── factor.py                  # Factor-based PGM representations
│   ├── interaction.py             # Node interactions
│   ├── observers.py               # Sampling observation and metrics
│   ├── pgm.py                     # Base PGM node classes
│   └── models/                    # Pre-built THRML models
│       ├── discrete_ebm.py        # Discrete energy-based models
│       ├── ebm.py                 # General EBM interface
│       └── ising.py               # Ising model implementations
├── examples/                      # 🆕 Example scripts
│   ├── __init__.py
│   ├── blueprints_demo.py         # Complete blueprint demonstration
│   ├── thrml_demo.py              # 🆕 THRML Ising model demo
│   ├── 00_probabilistic_computing.ipynb  # 🆕 THRML intro
│   ├── 01_all_of_thrml.ipynb      # 🆕 THRML complete tour
│   └── 02_spin_models.ipynb       # 🆕 THRML spin systems
├── mcp.json                       # MCP manifest (tool definitions)
├── mcp/                           # MCP Server Package
│   ├── __init__.py
│   ├── server.py                  # Main MCP server
│   └── tools/                     # MCP tool implementations
│       ├── __init__.py
│       ├── reasoning.py           # Reasoning tools
│       ├── dag.py                 # DAG manipulation tools
│       ├── model.py               # Model persistence tools
│       └── repo.py                # Repository file tools
├── tests/                         # Test suite
│   ├── test_api.py
│   ├── test_mcp.py                # MCP server tests
│   ├── test_dag_and_trm.py
│   └── ...
├── config.py                      # Training configuration
├── encoders.py                    # Text/Image encoders (PyTorch)
├── datasets.py                    # Dataset utilities (PyTorch)
├── training.py                    # Training pipeline (PyTorch)
├── run_all_tests.py               # Complete system test runner
├── requirements.txt               # Dependencies
├── README.md                      # This documentation
└── LICENSE                        # BSD 3-Clause License
```

### Core Files

| File | Description | Dependencies |
|------|-------------|--------------|
| `t_rlinkos_trm_fractal_dag.py` | Pure NumPy recursive reasoning model | NumPy |
| `trlinkos_trm_torch.py` | PyTorch version with GPU support | PyTorch |
| `trlinkos_llm_layer.py` | LLM integration layer | NumPy, t_rlinkos_trm_fractal_dag |
| `empirical_validation.py` | Comprehensive empirical validation suite | NumPy |
| `api.py` | FastAPI web API | FastAPI, Uvicorn |
| `mcp/server.py` | MCP server for LLM integration | NumPy |
| `mcp/tools/*.py` | MCP tool implementations | NumPy |
| `run_all_tests.py` | Complete system test runner | NumPy, (optional) PyTorch |

### Utility Files

| File | Description | Dependencies |
|------|-------------|--------------|
| `download_data.py` | HTTP/HTTPS file downloader | requests |
| `google_scraper.py` | Google search result scraper | requests, beautifulsoup4 |

## 🚀 Quick Start

> **✅ All examples in this section are fully implemented and executable.**

### Minimal Example

```python
import numpy as np
from t_rlinkos_trm_fractal_dag import TRLinkosTRM

# Initialize model
x_dim, y_dim, z_dim = 64, 32, 64
model = TRLinkosTRM(x_dim, y_dim, z_dim)

# Create input batch
x_batch = np.random.randn(8, x_dim)

# Run recursive reasoning
y_pred, dag = model.forward_recursive(
    x_batch,
    max_steps=10,
    inner_recursions=3
)

print(f"Output shape: {y_pred.shape}")  # (8, 32)
print(f"DAG nodes: {len(dag.nodes)}")   # 80
```

### Verify Installation

```bash
# Install dependencies
pip install numpy

# Run tests to verify installation
python run_all_tests.py

# Expected output: 🎉 ALL TESTS PASSED! 🎉
```

## 📖 Usage Examples

> **✅ All examples below are fully implemented and can be run directly.** Copy any example into a Python file and execute it.

### Basic Inference

```python
import numpy as np
from t_rlinkos_trm_fractal_dag import TRLinkosTRM

# Create model
model = TRLinkosTRM(
    x_dim=64,         # Input dimension
    y_dim=32,         # Output dimension
    z_dim=64,         # Internal state dimension
    hidden_dim=256,   # Hidden layer size
    num_experts=4     # Number of dCaAP experts
)

# Forward pass
x = np.random.randn(4, 64)
y_pred, dag = model.forward_recursive(x, max_steps=16)
```

### With Scoring Function

```python
import numpy as np
from t_rlinkos_trm_fractal_dag import TRLinkosTRM

model = TRLinkosTRM(64, 32, 64)
x_batch = np.random.randn(8, 64)
target = np.random.randn(8, 32)

# Define a scorer (higher = better)
def scorer(x, y_pred):
    return -np.mean((y_pred - target) ** 2, axis=-1)

y_pred, dag = model.forward_recursive(
    x_batch,
    max_steps=10,
    scorer=scorer
)

# Get best reasoning node
best_node = dag.get_best_node()
print(f"Best step: {best_node.step}, Score: {best_node.score}")
```

### With Backtracking

```python
import numpy as np
from t_rlinkos_trm_fractal_dag import TRLinkosTRM

model = TRLinkosTRM(64, 32, 64)
x_batch = np.random.randn(8, 64)
target = np.random.randn(8, 32)

def scorer(x, y_pred):
    return -np.mean((y_pred - target) ** 2, axis=-1)

# Enable backtracking to restore best states
y_pred, dag = model.forward_recursive(
    x_batch,
    max_steps=10,
    scorer=scorer,
    backtrack=True,              # Enable backtracking
    backtrack_threshold=0.1      # Trigger threshold (10% degradation)
)
```

### Exploring the Fractal DAG

```python
from t_rlinkos_trm_fractal_dag import FractalMerkleDAG
import numpy as np

# Create fractal DAG
dag = FractalMerkleDAG(store_states=True, max_depth=3)

# Add nodes
y, z = np.random.randn(1, 8), np.random.randn(1, 16)
root_id = dag.add_step(step=0, y=y, z=z, parents=[], score=-1.0)
node1_id = dag.add_step(step=1, y=y*0.9, z=z*0.9, parents=[root_id], score=-0.8)

# Create fractal branch
branch_id = dag.create_branch(node1_id, y=y*1.1, z=z*1.1, score=-0.7)

# Get statistics
depth_stats = dag.get_depth_statistics()
print(f"Depth statistics: {depth_stats}")

# Traverse fractal path
path = dag.get_fractal_path(branch_id)
print(f"Path depths: {[n.depth for n in path]}")
```

### Using Fractal Branching in Forward Recursive

```python
from t_rlinkos_trm_fractal_dag import TRLinkosTRM
import numpy as np

model = TRLinkosTRM(64, 32, 64)
x_batch = np.random.randn(8, 64)
target = np.random.randn(8, 32)

def scorer(x, y_pred):
    return -np.mean((y_pred - target) ** 2, axis=-1)

# Enable fractal branching for exploration
y_pred, dag = model.forward_recursive_fractal(
    x_batch,
    max_steps=10,
    scorer=scorer,
    backtrack=True,
    fractal_branching=True,      # Enable fractal exploration
    branch_threshold=0.05,       # Variance threshold for branching
    max_branches_per_node=2      # Max branches per node
)

# Inspect fractal structure
depth_stats = dag.get_depth_statistics()
print(f"Fractal depth statistics: {depth_stats}")
```

### Training with Text Data

```python
from t_rlinkos_trm_fractal_dag import (
    TRLinkosTRM, TextEncoder, Dataset, Trainer, TrainingConfig
)
import numpy as np

# Create text encoder
text_encoder = TextEncoder(vocab_size=256, embed_dim=64, output_dim=64, mode="char")

# Create dataset with text data
dataset = Dataset(x_dim=64, y_dim=32, encoder_type="text", text_encoder=text_encoder)

# Add training samples
dataset.add_sample("Hello world", np.random.randn(32))
dataset.add_sample("AI reasoning", np.random.randn(32))

# Create model and trainer
model = TRLinkosTRM(64, 32, 64)
config = TrainingConfig(
    learning_rate=0.01,
    num_epochs=10,
    batch_size=4,
    max_steps=8,
    loss_fn="mse"
)

trainer = Trainer(model, config)
history = trainer.train(dataset)
```

### Training with Image Data

```python
from t_rlinkos_trm_fractal_dag import (
    TRLinkosTRM, ImageEncoder, Dataset, Trainer, TrainingConfig
)
import numpy as np

# Create image encoder
image_encoder = ImageEncoder(input_channels=3, patch_size=8, embed_dim=64, output_dim=64)

# Create dataset with image data
dataset = Dataset(x_dim=64, y_dim=32, encoder_type="image", image_encoder=image_encoder)

# Add training samples (images as numpy arrays)
for _ in range(10):
    image = np.random.rand(64, 64, 3)  # RGB image
    target = np.random.randn(32)
    dataset.add_sample(image, target)

# Create model and trainer
model = TRLinkosTRM(64, 32, 64)
config = TrainingConfig(
    learning_rate=0.01,
    num_epochs=10,
    batch_size=4
)

trainer = Trainer(model, config)
history = trainer.train(dataset)

# Evaluate
loss, predictions = trainer.evaluate(dataset)
print(f"Evaluation loss: {loss}")
```

## 📚 API Reference

### TRLinkosTRM

Main model class for recursive reasoning.

```python
TRLinkosTRM(
    x_dim: int,           # Input dimension
    y_dim: int,           # Output dimension
    z_dim: int,           # Internal state dimension
    hidden_dim: int = 256,  # Hidden layer size
    num_experts: int = 4    # Number of dCaAP experts
)
```

#### Methods

- **`forward_recursive(x, max_steps, inner_recursions, scorer, backtrack, backtrack_threshold)`**
  - `x`: Input tensor `[B, x_dim]`
  - `max_steps`: Maximum reasoning steps (default: 16)
  - `inner_recursions`: Internal recursions per step (default: 3)
  - `scorer`: Optional scoring function `(x, y) -> scores`
  - `backtrack`: Enable state restoration (default: False)
  - `backtrack_threshold`: Degradation threshold for backtracking (default: 0.1)
  - Returns: `(y_pred, FractalMerkleDAG)`

- **`forward_recursive_fractal(x, max_steps, inner_recursions, scorer, backtrack, backtrack_threshold, fractal_branching, branch_threshold, max_branches_per_node, perturbation_scale)`**
  - All parameters from `forward_recursive`, plus:
  - `fractal_branching`: Enable fractal branch exploration (default: True)
  - `branch_threshold`: Score variance threshold for creating branches (default: 0.05)
  - `max_branches_per_node`: Maximum branches per node (default: 2)
  - `perturbation_scale`: Scale of perturbation for branch exploration (default: 0.1)
  - Returns: `(y_pred, FractalMerkleDAG)` with fractal structure

### FractalMerkleDAG

Fractal data structure for reasoning trace.

```python
FractalMerkleDAG(
    store_states: bool = False,  # Store y/z states for backtracking
    max_depth: int = 3           # Maximum fractal depth
)
```

#### Methods

- **`add_step(step, y, z, parents, score, depth, branch_root)`**: Add reasoning step
- **`create_branch(parent_node_id, y, z, score)`**: Create fractal branch
- **`get_best_node()`**: Get node with highest score
- **`get_node_states(node_id)`**: Retrieve states for backtracking
- **`get_fractal_path(node_id)`**: Get path from root to node
- **`get_depth_statistics()`**: Get node count per depth level

### DCaAPCell

Biologically-inspired neuron with dCaAP activation.

```python
DCaAPCell(
    input_dim: int,    # Input dimension (x + y + z)
    hidden_dim: int,   # Hidden dimension
    z_dim: int,        # State dimension
    num_branches: int = 4  # Number of dendritic branches
)
```

### TorqueRouter

Expert routing based on Torque Clustering.

```python
TorqueRouter(
    x_dim: int,
    y_dim: int,
    z_dim: int,
    num_experts: int
)
```

### TextEncoder

Simple text encoder for textual data.

```python
TextEncoder(
    vocab_size: int = 256,    # Vocabulary size (256 for ASCII chars)
    embed_dim: int = 64,      # Embedding dimension
    output_dim: int = 64,     # Output dimension
    mode: str = "char"        # "char" or "word" tokenization
)
```

#### Methods

- **`encode(texts, max_length)`**: Encode list of texts to vectors `[B, output_dim]`

### ImageEncoder

Simple image encoder for visual data.

```python
ImageEncoder(
    input_channels: int = 3,   # Number of input channels (3 for RGB, 1 for grayscale)
    patch_size: int = 8,       # Patch size for "convolution"
    embed_dim: int = 64,       # Patch embedding dimension
    output_dim: int = 64       # Output dimension
)
```

#### Methods

- **`encode(images)`**: Encode list of images to vectors `[B, output_dim]`

### Dataset

Dataset class for training data.

```python
Dataset(
    x_dim: int,                              # Expected input dimension
    y_dim: int,                              # Expected output dimension
    encoder_type: str = "vector",            # "vector", "text", or "image"
    text_encoder: Optional[TextEncoder],     # TextEncoder instance (for text)
    image_encoder: Optional[ImageEncoder]    # ImageEncoder instance (for images)
)
```

#### Methods

- **`add_sample(x, y_target, metadata)`**: Add a sample to the dataset
- **`__len__()`**: Get dataset size
- **`__getitem__(idx)`**: Get sample by index

### DataLoader

Simple DataLoader for batched training.

```python
DataLoader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True
)
```

### TrainingConfig

Training configuration dataclass.

```python
TrainingConfig(
    learning_rate: float = 0.01,         # Learning rate
    num_epochs: int = 100,               # Number of epochs
    batch_size: int = 32,                # Batch size
    max_steps: int = 8,                  # Max reasoning steps
    inner_recursions: int = 3,           # Inner recursions per step
    use_fractal_branching: bool = False, # Enable fractal exploration
    loss_fn: str = "mse",                # Loss function ("mse", "cross_entropy", "cosine")
    log_interval: int = 10,              # Logging interval
    gradient_clip: float = 1.0           # Gradient clipping value
)
```

### Trainer

Training pipeline for T-RLINKOS.

```python
Trainer(
    model: TRLinkosTRM,
    config: TrainingConfig
)
```

#### Methods

- **`train(train_dataset, val_dataset)`**: Full training loop, returns history
- **`evaluate(dataset)`**: Evaluate model, returns `(loss, predictions)`
- **`train_epoch(dataloader)`**: Train one epoch, returns average loss

### Loss Functions

- **`mse_loss(y_pred, y_target)`**: Mean Squared Error loss
- **`cross_entropy_loss(logits, targets)`**: Cross-entropy loss for classification
- **`cosine_similarity_loss(y_pred, y_target)`**: Cosine similarity loss (1 - cos_sim)

### Model Serialization Functions

- **`save_model(model, filepath)`**: Save a TRLinkosTRM model to disk (.npz format)
- **`load_model(filepath)`**: Load a saved TRLinkosTRM model from disk

### Benchmark Functions

- **`benchmark_forward_recursive(model, batch_size, max_steps, ...)`**: Benchmark inference performance
- **`benchmark_forward_recursive_fractal(model, batch_size, max_steps, ...)`**: Benchmark fractal inference
- **`run_benchmark_suite(configs, batch_sizes, ...)`**: Run comprehensive benchmark suite
- **`print_benchmark_results(results)`**: Print formatted benchmark results

### BenchmarkResult

Result of a benchmark run (dataclass).

```python
@dataclass
class BenchmarkResult:
    name: str                    # Benchmark name
    config: Dict[str, Any]       # Model configuration
    total_time: float            # Total execution time (seconds)
    time_per_step: float         # Average time per reasoning step
    time_per_sample: float       # Average time per sample
    throughput: float            # Samples per second
    memory_estimate_mb: float    # Estimated memory usage (MB)
    num_steps: int               # Number of reasoning steps
    batch_size: int              # Batch size used
```

### LLM Integration Module (trlinkos_llm_layer.py)

#### ReasoningConfig

Configuration for the TRLINKOS reasoning layer.

```python
ReasoningConfig(
    input_dim: int = 4096,              # LLM hidden dimension
    output_dim: int = 256,              # Reasoning output dimension
    z_dim: int = 128,                   # Internal state dimension
    hidden_dim: int = 256,              # Hidden layer dimension
    num_experts: int = 4,               # Number of dCaAP experts
    max_reasoning_steps: int = 8,       # Maximum reasoning iterations
    inner_recursions: int = 3,          # Inner recursions per step
    enable_backtracking: bool = True,   # Enable state restoration
    backtrack_threshold: float = 0.1,   # Backtrack threshold
    enable_fractal_branching: bool = False,  # Enable fractal exploration
    use_attention_pooling: bool = True, # Use attention-based pooling
    project_to_llm_dim: bool = True,    # Project output back to LLM dim
)
```

#### TRLinkOSReasoningLayer

Main reasoning layer for LLM integration.

```python
TRLinkOSReasoningLayer(config: ReasoningConfig)
```

**Methods:**
- **`reason(hidden_states, attention_mask, scorer)`**: Perform recursive reasoning on LLM hidden states
- **`reason_with_adapter(adapter, input_ids, attention_mask, scorer)`**: Reason using an LLM adapter
- **`get_reasoning_trace(dag)`**: Get detailed reasoning trace from the DAG

#### LLM Adapters

- **`LLMAdapter`**: Abstract base class for LLM adapters
- **`HuggingFaceAdapter(model_name, device, ...)`**: Adapter for HuggingFace models
- **`MockLLMAdapter(model_name, hidden_dim)`**: Mock adapter for testing

#### ChainOfThoughtAugmenter

Augments chain-of-thought reasoning with T-RLINKOS.

```python
ChainOfThoughtAugmenter(reasoning_layer, adapter=None)
```

**Methods:**
- **`add_thought(thought_embedding, thought_text)`**: Process a single thought
- **`get_chain_trace()`**: Get the full chain-of-thought trace
- **`verify_chain()`**: Verify chain integrity
- **`reset()`**: Reset thought history

#### Factory Function

- **`create_reasoning_layer_for_llm(model_name, reasoning_steps, num_experts)`**: Create a reasoning layer configured for a specific LLM

## 🔗 References

### Scientific Papers

1. **dCaAP Activation**
   - Gidon, A., et al. (2020). "Dendritic action potentials and computation in human layer 2/3 cortical neurons." *Science*, 367(6473), 83-87. [DOI](https://www.science.org/doi/10.1126/science.aax6239)
   - Hashemi, M., & Tetzlaff, C. (2025). "Computational principles of dendritic action potentials." *bioRxiv*. [Link](https://www.biorxiv.org/content/10.1101/2025.06.10.658823v1)

2. **Torque Clustering**
   - Yang, J., & Lin, Z. (2025). "Torque Clustering." *IEEE Transactions on Pattern Analysis and Machine Intelligence*. [GitHub](https://github.com/JieYangBruce/TorqueClustering)

## 🧪 Running Tests

### Complete System Test

Run all tests with the comprehensive test runner:

```bash
# Run complete system test (NumPy + LLM Layer + PyTorch)
python run_all_tests.py

# Skip PyTorch tests (if torch is not installed)
python run_all_tests.py --skip-pytorch
```

Expected output:
```
======================================================================
T-RLINKOS TRM++ FRACTAL DAG - COMPLETE SYSTEM TEST
======================================================================
...
======================================================================
TEST SUMMARY
======================================================================
✅ PASS | Core NumPy Implementation Tests (30.20s)
✅ PASS | LLM Reasoning Layer Tests (0.86s)
✅ PASS | PyTorch TRM Implementation Tests (0.02s)
✅ PASS | Quick XOR Training Test (1.71s)
----------------------------------------------------------------------
Total: 4 tests | Passed: 4 | Failed: 0
Duration: 32.79s
======================================================================

🎉 ALL TESTS PASSED! 🎉
```

### Empirical Validation

Run the comprehensive empirical validation suite:

```bash
# Run full validation with interactive output
python empirical_validation.py

# Generate JSON report
python empirical_validation.py --output validation_report.json

# Quiet mode (summary only)
python empirical_validation.py --quiet
```

Expected output:
```
======================================================================
T-RLINKOS TRM++ EMPIRICAL VALIDATION
======================================================================
Running 11 validation tests...

Running: dCaAP Activation... ✅ PASS (score: 0.87, 0.01s)
Running: Torque Router... ✅ PASS (score: 1.00, 0.00s)
Running: Merkle-DAG... ✅ PASS (score: 1.00, 0.00s)
Running: Backtracking... ✅ PASS (score: 0.80, 0.02s)
Running: LLM Integration... ✅ PASS (score: 1.00, 0.14s)
...

======================================================================
VALIDATION SUMMARY
======================================================================
Total:  11 validations
Passed: 11 (100.0%)
Failed: 0
Average Score: 0.97
======================================================================

🎉 ALL VALIDATIONS PASSED! 🎉
```

### Individual Module Tests

```bash
# Run the built-in tests
python t_rlinkos_trm_fractal_dag.py

# Run the LLM reasoning layer tests
python trlinkos_llm_layer.py

# Run XOR training (requires PyTorch) - Direct method
python train_trlinkos_xor.py

# Or use the unified training launcher (recommended)
python launch_training.py                    # Default XOR training
python launch_training.py --epochs 100       # Custom epochs
python launch_training.py --device cuda      # Use GPU
./launch_training.sh --help                  # See all options
```

Expected output for NumPy tests:
```
[Test 1] y_pred shape: (8, 32)
[Test 1] Nombre de noeuds dans le DAG: 80
[Test 1] Best node step: 5 score: -0.6291...
...
[Test 13] ✅ Model serialization fonctionne correctement!
[Test 14] ✅ Formal benchmarks fonctionnent correctement!
✅ Tous les tests passent avec succès!
```

LLM layer expected output:
```
============================================================
TRLINKOS LLM Reasoning Layer - Tests
============================================================
[Test 1] ✅ ReasoningConfig works correctly!
[Test 2] ✅ MockLLMAdapter works correctly!
...
[Test 11] ✅ End-to-end pipeline works correctly!
============================================================
✅ All TRLINKOS LLM Reasoning Layer tests passed!
============================================================
```

## 📄 License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📊 Project Status

| Component | Status |
|-----------|--------|
| LinearNP | ✅ Complete |
| GELU/Softmax | ✅ Complete |
| dCaAP Activation | ✅ Complete |
| DCaAPCell | ✅ Complete |
| TorqueRouter | ✅ Complete |
| TRLinkosCore | ✅ Complete |
| FractalMerkleDAG | ✅ Complete |
| TRLinkosTRM | ✅ Complete |
| Backtracking | ✅ Complete |
| Fractal Branching in TRM | ✅ Complete |
| TextEncoder | ✅ Complete |
| ImageEncoder | ✅ Complete |
| Dataset/DataLoader | ✅ Complete |
| Training Pipeline | ✅ Complete |
| Loss Functions | ✅ Complete |
| Model Serialization | ✅ Complete |
| Formal Benchmarks | ✅ Complete |
| **LLM Reasoning Layer** | ✅ Complete |
| **LLM Adapters (Mistral, LLaMA)** | ✅ Complete |
| **Chain-of-Thought Augmenter** | ✅ Complete |
| **PyTorch TRM Implementation** | ✅ Complete |
| **XOR Training Script** | ✅ Complete |

## 🗺️ Roadmap

### Phase 1 (Short term) - Completed ✅

| Feature | Status | Description |
|---------|--------|-------------|
| Text/Image Encoders | ✅ Done | `TextEncoder` and `ImageEncoder` classes |
| Loss Functions | ✅ Done | MSE, Cross-entropy, Cosine similarity |
| forward_recursive_fractal | ✅ Done | Fractal exploration during reasoning |
| Functional Backtracking | ✅ Done | State restoration to best-scoring nodes |
| Model Serialization | ✅ Done | `save_model()` and `load_model()` functions |
| Formal Benchmarks | ✅ Done | `benchmark_forward_recursive()`, `run_benchmark_suite()` |

### Phase 2 (Medium term) - Completed ✅

| Feature | Status | Description |
|---------|--------|-------------|
| PyTorch/GPU Porting | ✅ Done | GPU acceleration via PyTorch (`trlinkos_trm_torch.py`) |
| XOR Training Example | ✅ Done | Training script for XOR problem (`train_trlinkos_xor.py`) |
| Numba Optimization | ✅ Done | JIT compilation for NumPy operations (`numba_optimizations.py`) |
| Multi-GPU Support | ✅ Done | Distributed training and inference (`multi_gpu_support.py`) |
| HuggingFace Integration | ✅ Done | Integration with transformers ecosystem (`huggingface_integration.py`) |
| Pre-trained Encoders | ✅ Done | BERT, ViT, etc. encoder support |
| ONNX Export | ✅ Done | Model export for production deployment (`onnx_export.py`) |

### Phase 3 (Long term) - Completed ✅

| Feature | Status | Description |
|---------|--------|-------------|
| Neuromorphic Version | ✅ Done | Spike-based implementation for neuromorphic hardware (`neuromorphic.py`) |
| LLM Integration | ✅ Done | Reasoning layer for LLMs (Mistral, LLaMA, etc.) via `trlinkos_llm_layer.py` |

## 🚀 Advanced Features

### Numba/JIT Optimization

Achieve 2-5x speedup with optional JIT compilation via Numba:

```python
# Automatically uses JIT-compiled versions when numba is installed
from t_rlinkos_trm_fractal_dag import TRLinkosTRM
import numpy as np

model = TRLinkosTRM(64, 32, 64)
x = np.random.randn(1024, 64)  # Large batch

# Automatically optimized with Numba (if installed)
y_pred, dag = model.forward_recursive(x, max_steps=10)

# Check optimization status
from numba_optimizations import get_optimization_info
info = get_optimization_info()
print(f"JIT enabled: {info['jit_enabled']}")
```

**Performance improvements:**
- `dcaap_activation`: ~3-5x faster
- Matrix operations: ~2-3x faster
- Distance computations: ~3-4x faster

Install: `pip install numba>=0.55.0`

### Multi-GPU Support

Train on multiple GPUs with DataParallel or DistributedDataParallel:

```python
import torch
from trlinkos_trm_torch import TRLinkosTRMTorch
from multi_gpu_support import wrap_data_parallel, setup_distributed, wrap_distributed_data_parallel

# Option 1: DataParallel (single-node multi-GPU)
model = TRLinkosTRMTorch(64, 32, 64)
model = wrap_data_parallel(model, device_ids=[0, 1, 2, 3])
model = model.cuda()

# Option 2: DistributedDataParallel (multi-node multi-GPU)
setup_distributed(rank=0, world_size=4)
model = TRLinkosTRMTorch(64, 32, 64).cuda()
model = wrap_distributed_data_parallel(model)

# Option 3: Gradient Accumulation (simulate larger batches)
from multi_gpu_support import GradientAccumulator
accumulator = GradientAccumulator(accumulation_steps=4)

for i, (x, y) in enumerate(dataloader):
    with accumulator.context(i):
        output = model(x.cuda())
        loss = criterion(output, y.cuda())
        accumulator.backward(loss)

    if accumulator.should_step(i):
        optimizer.step()
        optimizer.zero_grad()
```

### HuggingFace Integration

Use pre-trained encoders from HuggingFace:

```python
from huggingface_integration import PretrainedTextEncoder, PretrainedVisionEncoder
from huggingface_integration import create_trlinkos_with_encoder, list_available_models

# List available models
models = list_available_models(model_type="text")
for model in models:
    print(f"{model['alias']}: {model['description']}")

# Text encoding with BERT
encoder = PretrainedTextEncoder(
    "bert-base",  # or "bert-base-uncased"
    output_dim=64,
    pooling="mean",
    revision="<commit-hash>",  # Pin version for security
)
embeddings = encoder.encode(["Hello world", "AI reasoning"])
print(embeddings.shape)  # (2, 64)

# Vision encoding with ViT
encoder = PretrainedVisionEncoder("vit-base", output_dim=64)
from PIL import Image
images = [Image.open("cat.jpg"), Image.open("dog.jpg")]
embeddings = encoder.encode(images)

# Full integration: Pre-trained encoder + T-RLINKOS
encoder, model = create_trlinkos_with_encoder(
    encoder_name="bert-base",
    encoder_type="text",
    output_dim=32,
    z_dim=64,
    num_experts=4,
)

# Use for reasoning
texts = ["The capital of France is Paris", "AI will transform society"]
text_embeddings = encoder.encode(texts)
y_pred, dag = model.forward_recursive(text_embeddings, max_steps=8)
```

**Supported models:**
- Text: BERT, GPT-2, RoBERTa, DistilBERT, LLaMA, Mistral
- Vision: ViT (Vision Transformer)

Install: `pip install transformers>=4.30.0`

### ONNX Export

Export models for production deployment:

```python
# PyTorch model export (recommended)
import torch
from trlinkos_trm_torch import TRLinkosTRMTorch
from onnx_export import export_torch_model_to_onnx, ONNXPredictor

model = TRLinkosTRMTorch(64, 32, 64)
export_torch_model_to_onnx(
    model,
    "trlinkos.onnx",
    input_shape=(1, 64),
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

# Inference with ONNX Runtime
predictor = ONNXPredictor("trlinkos.onnx")
output = predictor.predict(input_data)

# Benchmark inference
results = predictor.benchmark(input_shape=(32, 64), num_iterations=100)
print(f"Throughput: {results['throughput']:.1f} samples/sec")
print(f"Latency: {results['avg_time_per_sample']*1000:.2f} ms")

# NumPy model export (parameters only)
from t_rlinkos_trm_fractal_dag import TRLinkosTRM
from onnx_export import export_numpy_model_to_onnx

model = TRLinkosTRM(64, 32, 64)
export_numpy_model_to_onnx(model, "trlinkos_params.npz")
```

**Benefits:**
- Cross-platform deployment (Windows, Linux, macOS)
- Hardware acceleration (CPU, CUDA, TensorRT)
- Optimized inference
- No Python dependency

Install: `pip install onnx>=1.12.0 onnxruntime>=1.12.0`

### Neuromorphic Computing

Experimental spike-based implementation for neuromorphic hardware:

```python
from neuromorphic import NeuromorphicTRLinkosTRM, NeuromorphicConfig

# Create neuromorphic model
config = NeuromorphicConfig(
    dt=1.0,  # Time step (ms)
    v_thresh=-50.0,  # Spike threshold (mV)
    encoding_rate_max=200.0,  # Max firing rate (Hz)
)

model = NeuromorphicTRLinkosTRM(
    x_dim=64,
    y_dim=32,
    z_dim=64,
    config=config,
)

# Continuous input -> Spike-based processing -> Continuous output
input_data = np.random.rand(4, 64)
output = model.forward(input_data, time_steps=100)

# Or work directly with spikes
spike_trains = model.encode_to_spikes(input_data, time_steps=100, encoding="rate")
output_spikes = model.forward_spikes(spike_trains, time_steps=100)
output = model.decode_from_spikes(output_spikes)
```

**Features:**
- Spiking dCaAP neurons with dendritic computation
- Rate and temporal encoding
- Event-driven computation
- Low-power operation
- Adaptive thresholds

**Target hardware:**
- Intel Loihi
- IBM TrueNorth
- SpiNNaker
- General CPU/GPU (simulation)

⚠️ **Note:** This is an experimental research implementation. For production, use standard NumPy or PyTorch versions.

## 📦 New Features

### Model Serialization

Save and load trained models:

```python
from t_rlinkos_trm_fractal_dag import TRLinkosTRM, save_model, load_model

# Create and train model
model = TRLinkosTRM(64, 32, 64)
# ... training ...

# Save model
save_model(model, "my_model.npz")

# Load model
loaded_model = load_model("my_model.npz")
y_pred, dag = loaded_model.forward_recursive(x_batch)
```

### Formal Benchmarks

Run benchmarks to measure performance:

```python
from t_rlinkos_trm_fractal_dag import (
    TRLinkosTRM, benchmark_forward_recursive,
    run_benchmark_suite, print_benchmark_results
)

# Single benchmark
model = TRLinkosTRM(64, 32, 64)
result = benchmark_forward_recursive(model, batch_size=32, max_steps=16)
print(f"Throughput: {result.throughput:.1f} samples/sec")
print(f"Time per step: {result.time_per_step * 1000:.2f} ms")

# Full benchmark suite
results = run_benchmark_suite()
print_benchmark_results(results)
```

### LLM Reasoning Layer (NEW)

Use T-RLINKOS as a reasoning layer for any open-source LLM (Mistral, LLaMA, GPT-2, BERT, etc.):

```python
from trlinkos_llm_layer import (
    TRLinkOSReasoningLayer,
    ReasoningConfig,
    MockLLMAdapter,
    HuggingFaceAdapter,
    create_reasoning_layer_for_llm,
)
import numpy as np

# Method 1: Use factory function for common LLMs
reasoning_layer, config = create_reasoning_layer_for_llm("llama-7b")

# Method 2: Create with custom config
config = ReasoningConfig(
    input_dim=4096,        # LLaMA-7B hidden dimension
    output_dim=256,        # Reasoning output dimension
    z_dim=128,             # Internal state dimension
    num_experts=4,         # Number of dCaAP experts
    max_reasoning_steps=8, # Reasoning iterations
    enable_backtracking=True,
)
reasoning_layer = TRLinkOSReasoningLayer(config)

# Use with LLM hidden states
llm_hidden_states = np.random.randn(4, 128, 4096)  # [B, seq_len, hidden_dim]
output, dag = reasoning_layer.reason(llm_hidden_states)
print(f"Output shape: {output.shape}")  # (4, 256)

# Get reasoning trace for interpretability
trace = reasoning_layer.get_reasoning_trace(dag)
print(f"DAG nodes: {trace['num_nodes']}")
```

#### With HuggingFace Models

```python
from trlinkos_llm_layer import TRLinkOSReasoningLayer, ReasoningConfig, HuggingFaceAdapter

# Create adapter for any HuggingFace model
# For production, specify a revision (commit hash) for security and reproducibility
adapter = HuggingFaceAdapter(
    model_name="mistralai/Mistral-7B-v0.1",  # Or "meta-llama/Llama-2-7b-hf"
    device="cuda",  # or "cpu"
    # Example: Pin to a specific commit hash for production security
    # Get the actual commit hash from: https://huggingface.co/mistralai/Mistral-7B-v0.1/commits/main
    revision="<commit-hash>",  # Replace with actual commit hash, e.g., "26bca36bde8333b5d7f72e9ed20ccda6a618af24"
)

# Create reasoning layer
config = ReasoningConfig(input_dim=adapter.get_hidden_dim())
reasoning_layer = TRLinkOSReasoningLayer(config)

# Tokenize and reason
tokens = adapter.tokenize(["What is the capital of France?"])
output, dag = reasoning_layer.reason_with_adapter(
    adapter,
    tokens["input_ids"],
    tokens["attention_mask"],
)
```

#### Chain-of-Thought Augmentation

```python
from trlinkos_llm_layer import TRLinkOSReasoningLayer, ChainOfThoughtAugmenter, ReasoningConfig
import numpy as np

config = ReasoningConfig(input_dim=768, output_dim=256)
reasoning_layer = TRLinkOSReasoningLayer(config)
cot = ChainOfThoughtAugmenter(reasoning_layer)

# Process chain of thoughts
thought1 = np.random.randn(768)  # First thought embedding
enhanced1, trace1 = cot.add_thought(thought1, "Let me think about this...")

thought2 = np.random.randn(768)  # Second thought embedding
enhanced2, trace2 = cot.add_thought(thought2, "Building on that...")

# Get full reasoning chain
chain_trace = cot.get_chain_trace()
print(f"Chain length: {len(chain_trace)}")
```

#### Supported LLMs

The reasoning layer supports any LLM that provides hidden states:

| Model Family | Example Models | Hidden Dim |
|--------------|----------------|------------|
| LLaMA | llama-7b, llama-13b, llama-70b | 4096, 5120, 8192 |
| Mistral | mistral-7b | 4096 |
| GPT-2 | gpt2, gpt2-medium, gpt2-large, gpt2-xl | 768-1600 |
| BERT | bert-base, bert-large | 768, 1024 |
| Any HuggingFace | Via HuggingFaceAdapter | Auto-detected |

### PyTorch Implementation

The project includes a full PyTorch implementation for GPU acceleration (`trlinkos_trm_torch.py`):

```python
import torch
from trlinkos_trm_torch import TRLinkosTRMTorch

# Create PyTorch model (GPU-accelerated)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = TRLinkosTRMTorch(
    x_dim=2,        # Input dimension
    y_dim=1,        # Output dimension
    z_dim=8,        # Internal state dimension
    hidden_dim=32,  # Hidden layer size
    num_experts=4,  # Number of dCaAP experts
    num_branches=4, # Number of dendritic branches
).to(device)

# Forward pass
x = torch.randn(32, 2).to(device)
y_pred = model(x, max_steps=6, inner_recursions=2)
```

#### Key PyTorch Components

| Component | Description |
|-----------|-------------|
| `LinearTorch` | PyTorch linear layer wrapper |
| `DCaAPCellTorch` | PyTorch implementation of dCaAP neuron |
| `TorqueRouterTorch` | PyTorch Torque Clustering router |
| `TRLinkosCoreTorch` | PyTorch core with MoE architecture |
| `TRLinkosTRMTorch` | Full PyTorch model for end-to-end training |

### 🚀 Training Launcher (Recommended)

The unified training launcher (`launch_training.py`) provides an easy way to start training on various datasets:

```bash
# Quick start - XOR training (default)
python launch_training.py

# Custom training with parameters
python launch_training.py --epochs 100 --lr 0.001 --batch-size 128

# GPU training
python launch_training.py --device cuda --epochs 50

# Training modes (XOR, text, image)
python launch_training.py --mode xor         # Logical XOR problem
python launch_training.py --mode text        # Text classification (sentiment)
python launch_training.py --mode image       # Image classification (light/dark)

# Using shell wrapper
./launch_training.sh --mode text --epochs 30

# See all options
python launch_training.py --help
```

**Features:**
- 🎯 Multiple training modes: XOR (logical), text classification, image classification
- ⚙️ Flexible hyperparameter configuration via command-line
- 🔄 Automatic device detection (CPU/GPU)
- 📊 Progress tracking and metrics visualization
- 🛡️ Error handling and validation
- 📝 Detailed logging and summaries

**Training Modes:**

1. **XOR Mode** (`--mode xor`): Trains on the classic XOR logical problem
   - Perfect for testing non-linear learning capabilities
   - Quick convergence (usually ~30 epochs)
   - 100% accuracy achievable

2. **Text Mode** (`--mode text`): Trains on sentiment classification
   - Dataset: Toy text dataset with positive/negative sentiment
   - Uses TextEncoder with word embeddings
   - Good for testing NLP capabilities

3. **Image Mode** (`--mode image`): Trains on synthetic image classification
   - Dataset: Synthetic 28x28 RGB images (light vs dark)
   - Uses ImageEncoder with CNN architecture
   - Demonstrates vision capabilities

**Example Outputs:**

```bash
# XOR training (5 epochs)
$ python launch_training.py --mode xor --epochs 5
# Output: 100% accuracy, correctly predicts all 4 XOR cases

# Text training (30 epochs)
$ python launch_training.py --mode text --epochs 30
# Output: ~100% accuracy on sentiment classification
#   'This is great news' -> positif
#   'I feel terrible today' -> négatif

# Image training (5 epochs)
$ python launch_training.py --mode image --epochs 5
# Output: 100% accuracy classifying light/dark images
```

### XOR Training Example

Train the model on the XOR problem (`train_trlinkos_xor.py`):

```bash
# Run XOR training directly
python train_trlinkos_xor.py

# Or use the launcher (recommended)
python launch_training.py --mode xor --epochs 50
```

The training script demonstrates:
- Mixed precision training with `autocast` and `GradScaler`
- XOR dataset generation
- Training loop with accuracy tracking
- Model evaluation on test samples

```python
# Expected output after training:
# Epoch 050 | Loss=0.0123 | Acc=1.0000
# X test:
#  [[0. 0.]
#   [0. 1.]
#   [1. 0.]
#   [1. 1.]]
# Probs:
#  [[0.02]
#   [0.98]
#   [0.97]
#   [0.03]]
# Preds:
#  [[0.]
#   [1.]
#   [1.]
#   [0.]]
```

### Utility Scripts

#### Data Download (`download_data.py`)

Utility for downloading data from URLs:

```python
from download_data import download_data

# Download a file from URL
download_data("https://example.com/data.csv", "local_data.csv")
```

**Features:**
- HTTP/HTTPS download support via `requests`
- Error handling for network issues
- Progress feedback

#### Google Scraper (`google_scraper.py`)

Web scraper for Google search results:

```bash
# Command line usage
python google_scraper.py "search query" --num_results 10 --output results.json
```

```python
# Programmatic usage
from google_scraper import google_scrape, save_results_to_file

# Perform search
results = google_scrape("machine learning", num_results=10)

# Save results to JSON
save_results_to_file(results, "search_results.json")

# Each result contains:
# - title: Page title
# - link: URL
# - snippet: Description excerpt
```

**Features:**
- Real-time Google search scraping
- Configurable number of results
- JSON output format
- Rate limiting (2s delay) to avoid blocking

## 📚 Documentation

Complete documentation available in this repository:

| Document | Description |
|----------|-------------|
| [CI_CD.md](CI_CD.md) | 🆕 **CI/CD Pipeline, Pre-commit Hooks, and Code Quality Guide** |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 🆕 **Development Guidelines and Contribution Workflow** |
| [BILAN_TECHNIQUE_IA.md](BILAN_TECHNIQUE_IA.md) | 🆕 **Comprehensive technical analysis: Is T-RLINKOS an AI?** (French) |
| [ANALYSE_IMPACT_TECHNOLOGIQUE.md](ANALYSE_IMPACT_TECHNOLOGIQUE.md) | 🆕 **Complete ruthless technology impact analysis - Expert R&D assessment** (French, 1254 lines) |
| [SYNTHESE_ANALYSE_IMPACT.md](SYNTHESE_ANALYSE_IMPACT.md) | 🆕 **Quick synthesis of technology impact analysis** (French, executive summary) |
| [BLUEPRINTS_INTEGRATION.md](BLUEPRINTS_INTEGRATION.md) | AI Architecture Blueprints integration guide |
| [THE-BLUEPRINTS.md](THE-BLUEPRINTS.md) | AI Architecture Blueprints patterns catalog |
| [AUDIT_COHERENCE.md](AUDIT_COHERENCE.md) | Promise/implementation coherence audit (French) |

### Future Documentation (Roadmap)

The following documentation is planned for future releases:

| Document | Description | Status |
|----------|-------------|--------|
| ROADMAP_TRLINKOS_V2.md | Development roadmap and planned features | 🔲 Planned |
| TRM_WHITEPAPER_TECHNIQUE.md | Technical whitepaper with architecture details | 🔲 Planned |
| DCAAPCELL_TECHNOTE.md | Technical note on dCaAP neuron implementation | 🔲 Planned |
| TORQUE_ROUTER_NOTE.md | Technical note on Torque Clustering router | 🔲 Planned |
| FRACTAL_DAG_SCIENTIFIC_NOTE.md | Scientific note on Fractal Merkle-DAG structure | 🔲 Planned |

## 🛠️ Development

### CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment:

- **Automated Testing**: Tests run on Python 3.8, 3.9, 3.10, 3.11, and 3.12
- **Code Quality**: Automatic linting with Black, isort, and Flake8
- **Coverage Reporting**: Code coverage tracked and reported to Codecov
- **Security Checks**: Bandit and Safety security scanning

### Pre-commit Hooks

Pre-commit hooks ensure code quality before commits:

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

Configured hooks:
- Code formatting (Black, isort)
- Linting (Flake8)
- Security checks (Bandit)
- File consistency checks

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
make test

# Run tests with coverage
make test-cov

# Format code
make format

# Run linters
make lint
```

For detailed contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

### Code Quality Tools

| Tool | Purpose | Configuration |
|------|---------|---------------|
| Black | Code formatter | `pyproject.toml` |
| isort | Import sorter | `pyproject.toml` |
| Flake8 | Linter | `.flake8` |
| Pytest | Testing framework | `pyproject.toml` |
| Coverage | Code coverage | `.coveragerc` |
| Bandit | Security linter | `.pre-commit-config.yaml` |
