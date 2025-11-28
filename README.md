# T-RLINKOS TRM++ Fractal DAG

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.20%2B-blue.svg)](https://numpy.org/)

**T-RLINKOS TRM++ (Tiny Recursive Linkos Model ++)** is a pure NumPy implementation of a recursive reasoning architecture inspired by neuroscience research and modern clustering techniques.

## 🧠 Overview

This project implements a recursive reasoning model that combines:

- **dCaAP-inspired neurons** (Dendritic Calcium Action Potential) based on [Gidon et al., Science 2020](https://www.science.org/doi/10.1126/science.aax6239) and [Hashemi & Tetzlaff, bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.06.10.658823v1)
- **Torque Clustering router** for expert selection based on [Yang & Lin, TPAMI 2025](https://github.com/JieYangBruce/TorqueClustering)
- **Fractal Merkle-DAG** for reasoning trace auditing and backtracking
- **Mixture of Experts (MoE)** architecture with biologically-inspired activation functions

The model is designed to be **framework-agnostic**, using only NumPy for computation, making it portable and easy to understand.

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

## 📦 Installation

### Prerequisites

- Python 3.8+
- NumPy 1.20+

### Install

```bash
# Clone the repository
git clone https://github.com/RektaPro/TrlinkosTrmFractalDag.git
cd TrlinkosTrmFractalDag

# Install dependencies
pip install numpy

# For PyTorch version (GPU support)
pip install torch

# For web scraping utilities
pip install requests beautifulsoup4
```

## 📁 Project Structure

```
TrlinkosTrmFractalDagWeb/
├── t_rlinkos_trm_fractal_dag.py   # Core NumPy implementation
├── trlinkos_trm_torch.py          # PyTorch GPU implementation
├── trlinkos_llm_layer.py          # LLM reasoning layer integration
├── train_trlinkos_xor.py          # XOR training example (PyTorch)
├── download_data.py               # Data download utility
├── google_scraper.py              # Google search scraper
├── google_homepage.html           # Sample downloaded HTML data
├── ai_results.json                # AI results data file
├── README.md                      # This documentation
├── AUDIT_COHERENCE.md             # Coherence analysis (French)
├── ANALYSE_IMPACT_TECHNOLOGIQUE.md    # Tech impact analysis (French)
├── ANALYSE_IMPACT_CONNEXION_INTERNET.md # Internet connectivity analysis (French)
└── LICENSE                        # BSD 3-Clause License
```

### Core Files

| File | Description | Dependencies |
|------|-------------|--------------|
| `t_rlinkos_trm_fractal_dag.py` | Pure NumPy recursive reasoning model | NumPy |
| `trlinkos_trm_torch.py` | PyTorch version with GPU support | PyTorch |
| `trlinkos_llm_layer.py` | LLM integration layer | NumPy, t_rlinkos_trm_fractal_dag |
| `train_trlinkos_xor.py` | XOR training script | PyTorch, trlinkos_trm_torch |

### Utility Files

| File | Description | Dependencies |
|------|-------------|--------------|
| `download_data.py` | HTTP/HTTPS file downloader | requests |
| `google_scraper.py` | Google search result scraper | requests, beautifulsoup4 |

## 🚀 Quick Start

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

## 📖 Usage Examples

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

```bash
# Run the built-in tests
python t_rlinkos_trm_fractal_dag.py

# Run the LLM reasoning layer tests
python trlinkos_llm_layer.py
```

Expected output:
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

### Phase 2 (Medium term) - In Progress

| Feature | Status | Description |
|---------|--------|-------------|
| PyTorch/GPU Porting | ✅ Done | GPU acceleration via PyTorch (`trlinkos_trm_torch.py`) |
| XOR Training Example | ✅ Done | Training script for XOR problem (`train_trlinkos_xor.py`) |
| Numba Optimization | 🔲 Planned | JIT compilation for NumPy operations |
| Multi-GPU Support | 🔲 Planned | Distributed training and inference |
| HuggingFace Integration | 🔲 Planned | Integration with transformers ecosystem |
| Pre-trained Encoders | 🔲 Planned | BERT, ViT, etc. encoder support |
| ONNX Export | 🔲 Planned | Model export for production deployment |

### Phase 3 (Long term) - Research

| Feature | Status | Description |
|---------|--------|-------------|
| Neuromorphic Version | 🔲 Research | Intel Loihi, IBM TrueNorth support |
| LLM Integration | ✅ Done | Reasoning layer for LLMs (Mistral, LLaMA, etc.) via `trlinkos_llm_layer.py` |

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

### XOR Training Example

Train the model on the XOR problem (`train_trlinkos_xor.py`):

```bash
# Run XOR training
python train_trlinkos_xor.py
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

See the [docs/](docs/) folder for detailed documentation:

| Document | Description |
|----------|-------------|
| [ROADMAP_TRLINKOS_V2.md](docs/ROADMAP_TRLINKOS_V2.md) | Development roadmap and planned features |
| [TRM_WHITEPAPER_TECHNIQUE.md](docs/TRM_WHITEPAPER_TECHNIQUE.md) | Technical whitepaper with architecture details |
| [DCAAPCELL_TECHNOTE.md](docs/DCAAPCELL_TECHNOTE.md) | Technical note on dCaAP neuron implementation |
| [TORQUE_ROUTER_NOTE.md](docs/TORQUE_ROUTER_NOTE.md) | Technical note on Torque Clustering router |
| [FRACTAL_DAG_SCIENTIFIC_NOTE.md](docs/FRACTAL_DAG_SCIENTIFIC_NOTE.md) | Scientific note on Fractal Merkle-DAG structure |
| [IMPACT_ANALYSIS_V2.md](docs/IMPACT_ANALYSIS_V2.md) | Comprehensive technological impact analysis |

See also:
- [AUDIT_COHERENCE.md](AUDIT_COHERENCE.md) for detailed coherence analysis
- [ANALYSE_IMPACT_TECHNOLOGIQUE.md](ANALYSE_IMPACT_TECHNOLOGIQUE.md) for technological impact analysis (French)
- [ANALYSE_IMPACT_CONNEXION_INTERNET.md](ANALYSE_IMPACT_CONNEXION_INTERNET.md) for internet connectivity impact analysis (French)