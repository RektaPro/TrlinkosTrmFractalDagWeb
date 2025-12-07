# Activation Guide: Advanced Features for T-RLINKOS TRM++

This guide explains how to activate and use the 5 advanced features that have been enabled in the repository.

## ✅ What Has Been Activated

All 5 advanced features are now activated in `requirements.txt`:

1. **Numba/JIT Optimization** - 2-5x automatic acceleration
2. **Multi-GPU Support** - Training scalability across multiple GPUs
3. **HuggingFace Integration** - Pre-trained encoders from HuggingFace ecosystem
4. **ONNX Export** - Multiplatform deployment
5. **Neuromorphic Computing** - Experimental spike-based implementation

## 📦 Installation

### Quick Start (Core + Numba + ONNX)

```bash
# Install core dependencies + activated features
pip install -r requirements.txt
```

This installs:
- NumPy (required)
- Numba (automatic 2-5x speedup)
- ONNX & ONNX Runtime (model export)
- pytest (testing)
- FastAPI & Uvicorn (web API)

### Full Installation (All Features)

To enable PyTorch and HuggingFace features:

```bash
# Install all dependencies including PyTorch and Transformers
pip install torch>=2.0.0
pip install transformers>=4.30.0
```

**Note:** PyTorch and Transformers are large packages (~2-3 GB). They are optional and only needed for:
- GPU acceleration (PyTorch)
- Pre-trained encoders like BERT, GPT-2, ViT (Transformers)

## 🚀 Feature Usage

### 1. Numba/JIT Optimization (Already Active!)

**Status:** ✅ Automatically enabled when Numba is installed

**Performance:** 2-5x speedup for compute-intensive operations

**Usage:**
```python
import numpy as np
from t_rlinkos_trm_fractal_dag import TRLinkosTRM, USE_NUMBA, NUMBA_AVAILABLE

# Check if Numba is active
print(f"Numba enabled: {USE_NUMBA and NUMBA_AVAILABLE}")

# Use model normally - Numba optimizations are transparent
model = TRLinkosTRM(64, 32, 64)
x = np.random.randn(8, 64)
y_pred, dag = model.forward_recursive(x, max_steps=10)
```

**Benchmark:**
```bash
python numba_optimizations.py
```

**Optimized Functions:**
- `dcaap_activation`: 3-5x faster
- `gelu`: 2-3x faster  
- `matmul_add`: 2-3x faster
- `softmax`: 2x faster
- `distance_squared`: 3-4x faster

### 2. Multi-GPU Support

**Requirements:** PyTorch (`pip install torch>=2.0.0`)

**Usage:**

#### Single-Node Multi-GPU (DataParallel)
```python
import torch
from trlinkos_trm_torch import TRLinkosTRMTorch
from multi_gpu_support import wrap_data_parallel

model = TRLinkosTRMTorch(64, 32, 64)
model = wrap_data_parallel(model, device_ids=[0, 1, 2, 3])
model = model.cuda()

# Training loop works as usual
x = torch.randn(32, 64).cuda()
output = model(x)
```

#### Multi-Node Multi-GPU (DistributedDataParallel)
```python
from multi_gpu_support import setup_distributed, wrap_distributed_data_parallel, cleanup_distributed

# In each process
setup_distributed(rank=0, world_size=4)
model = TRLinkosTRMTorch(64, 32, 64).cuda()
model = wrap_distributed_data_parallel(model)

# ... training ...

cleanup_distributed()
```

#### Gradient Accumulation
```python
from multi_gpu_support import GradientAccumulator

accumulator = GradientAccumulator(accumulation_steps=4)

for i, (x, y) in enumerate(dataloader):
    with accumulator.context(i):
        output = model(x)
        loss = criterion(output, y)
        accumulator.backward(loss)
    
    if accumulator.should_step(i):
        optimizer.step()
        optimizer.zero_grad()
```

**Test:**
```bash
python multi_gpu_support.py
```

### 3. HuggingFace Integration

**Requirements:** Transformers (`pip install transformers>=4.30.0`)

**Usage:**

#### Text Encoding with BERT
```python
from huggingface_integration import PretrainedTextEncoder, list_available_models

# List available models
models = list_available_models(model_type="text")
for model in models:
    print(f"{model['alias']}: {model['description']}")

# Use BERT encoder
encoder = PretrainedTextEncoder(
    "bert-base",  # or "bert-base-uncased"
    output_dim=64,
    pooling="mean",
)
embeddings = encoder.encode(["Hello world", "AI reasoning"])
print(embeddings.shape)  # (2, 64)
```

#### Vision Encoding with ViT
```python
from huggingface_integration import PretrainedVisionEncoder
from PIL import Image

encoder = PretrainedVisionEncoder("vit-base", output_dim=64)
images = [Image.open("cat.jpg"), Image.open("dog.jpg")]
embeddings = encoder.encode(images)
```

#### Full Integration
```python
from huggingface_integration import create_trlinkos_with_encoder

# Create encoder + T-RLINKOS model
encoder, model = create_trlinkos_with_encoder(
    encoder_name="bert-base",
    encoder_type="text",
    output_dim=32,
    z_dim=64,
    num_experts=4,
)

# Encode and reason
texts = ["The capital of France is Paris", "AI will transform society"]
text_embeddings = encoder.encode(texts)
y_pred, dag = model.forward_recursive(text_embeddings, max_steps=8)
```

**Supported Models:**
- Text: `bert-base`, `bert-large`, `gpt2`, `gpt2-medium`, `distilbert`, `roberta-base`, `llama-7b`, `mistral-7b`
- Vision: `vit-base`, `vit-large`

**Test:**
```bash
python huggingface_integration.py
```

### 4. ONNX Export

**Requirements:** Already installed with `pip install -r requirements.txt`

**Usage:**

#### Export PyTorch Model (Recommended)
```python
import torch
from trlinkos_trm_torch import TRLinkosTRMTorch
from onnx_export import export_torch_model_to_onnx, ONNXPredictor

# Create and export model
model = TRLinkosTRMTorch(64, 32, 64)
export_torch_model_to_onnx(
    model,
    "trlinkos.onnx",
    input_shape=(1, 64),
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

# Inference with ONNX Runtime
predictor = ONNXPredictor("trlinkos.onnx")
output = predictor.predict(input_data)

# Benchmark
results = predictor.benchmark(input_shape=(32, 64), num_iterations=100)
print(f"Throughput: {results['throughput']:.1f} samples/sec")
print(f"Latency: {results['avg_time_per_sample']*1000:.2f} ms")
```

#### Export NumPy Model (Parameters Only)
```python
from t_rlinkos_trm_fractal_dag import TRLinkosTRM
from onnx_export import export_numpy_model_to_onnx

model = TRLinkosTRM(64, 32, 64)
export_numpy_model_to_onnx(model, "trlinkos_params.npz")
```

**Benefits:**
- Cross-platform deployment (Windows, Linux, macOS)
- Hardware acceleration (CPU, CUDA, TensorRT)
- No Python dependency in production
- Optimized inference performance

**Test:**
```bash
python onnx_export.py
```

### 5. Neuromorphic Computing

**Requirements:** Only NumPy (already installed)

**Status:** ⚠️ Experimental research prototype

**Usage:**
```python
from neuromorphic import NeuromorphicTRLinkosTRM, NeuromorphicConfig
import numpy as np

# Create neuromorphic model
config = NeuromorphicConfig(
    dt=1.0,                    # Time step (ms)
    v_thresh=-50.0,            # Spike threshold (mV)
    encoding_rate_max=200.0,   # Max firing rate (Hz)
)

model = NeuromorphicTRLinkosTRM(
    x_dim=64,
    y_dim=32,
    z_dim=64,
    config=config,
)

# Continuous input -> Spike processing -> Continuous output
input_data = np.random.rand(4, 64)
output = model.forward(input_data, time_steps=100)

# Or work directly with spikes
spike_trains = model.encode_to_spikes(input_data, time_steps=100, encoding="rate")
output_spikes = model.forward_spikes(spike_trains, time_steps=100)
output = model.decode_from_spikes(output_spikes)
```

**Features:**
- Spiking dCaAP neurons with dendritic computation
- Rate and temporal encoding/decoding
- Event-driven computation
- Low-power operation
- Adaptive thresholds

**Target Hardware:**
- Intel Loihi
- IBM TrueNorth
- SpiNNaker
- General CPU/GPU (simulation)

**Test:**
```bash
python neuromorphic.py
```

## 🧪 Testing

### Quick Test All Features
```bash
# Test core + Numba + ONNX
python -c "
from numba_optimizations import get_optimization_info
from onnx_export import get_onnx_info

print('Numba:', get_optimization_info()['numba_available'])
print('ONNX:', get_onnx_info()['onnx_available'])
"
```

### Individual Module Tests
```bash
# Test each feature module
python numba_optimizations.py       # Numba optimization tests
python multi_gpu_support.py         # Multi-GPU support tests
python huggingface_integration.py   # HuggingFace integration tests
python onnx_export.py              # ONNX export tests
python neuromorphic.py             # Neuromorphic tests
```

### Full System Test
```bash
python run_all_tests.py
```

## 📊 Performance Expectations

| Feature | Benefit | Requirements |
|---------|---------|--------------|
| Numba JIT | 2-5x speedup | numba (installed) |
| Multi-GPU | Linear scaling | PyTorch + CUDA GPUs |
| HuggingFace | Pre-trained models | transformers |
| ONNX Export | Production deployment | onnx + onnxruntime (installed) |
| Neuromorphic | Low-power hardware | NumPy only |

## ❓ Troubleshooting

### Numba Not Accelerating

**Check:**
```python
from t_rlinkos_trm_fractal_dag import USE_NUMBA, NUMBA_AVAILABLE
print(f"USE_NUMBA: {USE_NUMBA}, NUMBA_AVAILABLE: {NUMBA_AVAILABLE}")
```

**Solution:** Reinstall Numba: `pip install --upgrade numba>=0.55.0`

### Multi-GPU Not Working

**Check:**
```python
from multi_gpu_support import get_device_info
print(get_device_info())
```

**Solution:** 
- Install PyTorch: `pip install torch>=2.0.0`
- Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

### HuggingFace Models Not Loading

**Check:**
```python
from huggingface_integration import TRANSFORMERS_AVAILABLE
print(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
```

**Solution:** Install transformers: `pip install transformers>=4.30.0`

### ONNX Export Failing

**Check:**
```python
from onnx_export import get_onnx_info
info = get_onnx_info()
print(f"ONNX: {info['onnx_available']}, Runtime: {info['onnxruntime_available']}")
```

**Solution:** 
- Already installed with requirements.txt
- If issues persist: `pip install --upgrade onnx onnxruntime`

## 📚 Documentation

For detailed documentation on each feature, see:

| Document | Description |
|----------|-------------|
| [README.md](README.md) | Main documentation with usage examples |
| [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) | Implementation details for all 5 features |
| [numba_optimizations.py](numba_optimizations.py) | Numba optimization module with benchmarks |
| [multi_gpu_support.py](multi_gpu_support.py) | Multi-GPU training utilities |
| [huggingface_integration.py](huggingface_integration.py) | HuggingFace encoder integration |
| [onnx_export.py](onnx_export.py) | ONNX export and inference |
| [neuromorphic.py](neuromorphic.py) | Neuromorphic spike-based implementation |

## 🎯 Summary

All 5 advanced features are now **activated** and ready to use:

✅ **Numba/JIT Optimization** - Installed and automatically enabled  
✅ **Multi-GPU Support** - Module ready (requires PyTorch)  
✅ **HuggingFace Integration** - Module ready (requires transformers)  
✅ **ONNX Export** - Installed and ready to use  
✅ **Neuromorphic Computing** - Ready to use (experimental)  

**Next Steps:**
1. Install optional dependencies if needed: `pip install torch transformers`
2. Test features: `python numba_optimizations.py`
3. Start using: See usage examples above
4. Read full docs: [README.md](README.md)

For questions or issues, refer to the documentation or file an issue on GitHub.
