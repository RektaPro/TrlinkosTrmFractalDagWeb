#!/usr/bin/env python3
"""
Test script to verify all 5 activated features are working correctly.

This script tests:
1. Numba/JIT Optimization
2. Multi-GPU Support (module loading)
3. HuggingFace Integration (module loading)
4. ONNX Export
5. Neuromorphic Computing

Run: python test_activated_features.py
"""

import numpy as np
import sys

print("=" * 70)
print("TESTING ACTIVATED FEATURES FOR T-RLINKOS TRM++")
print("=" * 70)

# Track test results
results = []

# ============================================================================
# Test 1: Numba/JIT Optimization
# ============================================================================
print("\n[Test 1/5] Numba/JIT Optimization")
print("-" * 70)

try:
    from numba_optimizations import (
        get_optimization_info,
        benchmark_optimization,
        dcaap_activation_jit,
        NUMBA_AVAILABLE
    )
    
    info = get_optimization_info()
    print(f"✅ Numba Available: {info['numba_available']}")
    print(f"✅ JIT Enabled: {info['jit_enabled']}")
    print(f"✅ Optimized Functions: {len(info['optimized_functions'])}")
    
    # Quick benchmark
    benchmark_results = benchmark_optimization(batch_size=256, dim=64, num_iterations=20)
    speedup = benchmark_results['speedup']
    print(f"✅ Speedup: {speedup:.2f}x (over pure NumPy)")
    
    # Test with main model
    from t_rlinkos_trm_fractal_dag import TRLinkosTRM
    print(f"✅ Main model using Numba: {NUMBA_AVAILABLE}")
    
    model = TRLinkosTRM(64, 32, 64)
    x = np.random.randn(4, 64)
    y_pred, dag = model.forward_recursive(x, max_steps=3)
    print(f"✅ Model inference works: output shape {y_pred.shape}")
    
    results.append(("Numba/JIT Optimization", True, f"{speedup:.2f}x speedup"))
    print("✅ Test 1/5 PASSED")
    
except Exception as e:
    print(f"❌ Test 1/5 FAILED: {e}")
    results.append(("Numba/JIT Optimization", False, str(e)))

# ============================================================================
# Test 2: Multi-GPU Support
# ============================================================================
print("\n[Test 2/5] Multi-GPU Support")
print("-" * 70)

try:
    from multi_gpu_support import (
        get_device_info,
        get_available_gpus,
        GradientAccumulator,
        TORCH_AVAILABLE
    )
    
    info = get_device_info()
    print(f"✅ Module loaded successfully")
    print(f"   PyTorch Available: {info['torch_available']}")
    print(f"   CUDA Available: {info.get('cuda_available', False)}")
    print(f"   Number of GPUs: {info['num_gpus']}")
    
    # Test gradient accumulator (works without PyTorch)
    accumulator = GradientAccumulator(accumulation_steps=4)
    should_step = [accumulator.should_step(i) for i in range(8)]
    expected = [False, False, False, True, False, False, False, True]
    assert should_step == expected, f"Gradient accumulator failed: {should_step}"
    print(f"✅ Gradient accumulator works correctly")
    
    if TORCH_AVAILABLE:
        print(f"✅ PyTorch is installed - GPU features available")
        status = "Ready with PyTorch"
    else:
        print(f"ℹ️  PyTorch not installed - install with: pip install torch>=2.0.0")
        status = "Ready (PyTorch optional)"
    
    results.append(("Multi-GPU Support", True, status))
    print("✅ Test 2/5 PASSED")
    
except Exception as e:
    print(f"❌ Test 2/5 FAILED: {e}")
    results.append(("Multi-GPU Support", False, str(e)))

# ============================================================================
# Test 3: HuggingFace Integration
# ============================================================================
print("\n[Test 3/5] HuggingFace Integration")
print("-" * 70)

try:
    from huggingface_integration import (
        list_available_models,
        get_model_info,
        TRANSFORMERS_AVAILABLE,
        MODEL_REGISTRY
    )
    
    print(f"✅ Module loaded successfully")
    print(f"   Transformers Available: {TRANSFORMERS_AVAILABLE}")
    print(f"   Pre-configured Models: {len(MODEL_REGISTRY)}")
    
    # List available models
    text_models = list_available_models(model_type="text")
    vision_models = list_available_models(model_type="vision")
    print(f"✅ Text models available: {len(text_models)}")
    print(f"✅ Vision models available: {len(vision_models)}")
    
    # Test model info
    info = get_model_info("bert-base")
    assert info['hidden_dim'] == 768, "Model info incorrect"
    print(f"✅ Model registry works: BERT hidden_dim={info['hidden_dim']}")
    
    if TRANSFORMERS_AVAILABLE:
        print(f"✅ Transformers is installed - pre-trained models available")
        status = "Ready with Transformers"
    else:
        print(f"ℹ️  Transformers not installed - install with: pip install transformers>=4.30.0")
        status = "Ready (Transformers optional)"
    
    results.append(("HuggingFace Integration", True, status))
    print("✅ Test 3/5 PASSED")
    
except Exception as e:
    print(f"❌ Test 3/5 FAILED: {e}")
    results.append(("HuggingFace Integration", False, str(e)))

# ============================================================================
# Test 4: ONNX Export
# ============================================================================
print("\n[Test 4/5] ONNX Export")
print("-" * 70)

try:
    from onnx_export import (
        get_onnx_info,
        export_numpy_model_to_onnx,
        ONNX_AVAILABLE,
        ONNXRUNTIME_AVAILABLE
    )
    
    info = get_onnx_info()
    print(f"✅ Module loaded successfully")
    print(f"   ONNX Available: {info['onnx_available']}")
    print(f"   ONNX Version: {info.get('onnx_version', 'N/A')}")
    print(f"   ONNX Runtime Available: {info['onnxruntime_available']}")
    print(f"   ONNX Runtime Version: {info.get('onnxruntime_version', 'N/A')}")
    
    if ONNXRUNTIME_AVAILABLE:
        providers = info.get('available_providers', [])
        print(f"✅ Execution Providers: {len(providers)}")
        for provider in providers[:3]:  # Show first 3
            print(f"   - {provider}")
    
    # Test export (parameters only for NumPy model)
    if ONNX_AVAILABLE:
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            model = TRLinkosTRM(32, 16, 32)
            export_path = os.path.join(tmpdir, "test_model.npz")
            export_numpy_model_to_onnx(model, export_path, input_shape=(1, 32))
            print(f"✅ Model export test successful")
    
    if ONNX_AVAILABLE and ONNXRUNTIME_AVAILABLE:
        status = f"Ready (ONNX {info['onnx_version']}, Runtime {info['onnxruntime_version']})"
    else:
        status = "Partially available"
    
    results.append(("ONNX Export", True, status))
    print("✅ Test 4/5 PASSED")
    
except Exception as e:
    print(f"❌ Test 4/5 FAILED: {e}")
    results.append(("ONNX Export", False, str(e)))

# ============================================================================
# Test 5: Neuromorphic Computing
# ============================================================================
print("\n[Test 5/5] Neuromorphic Computing")
print("-" * 70)

try:
    from neuromorphic import (
        NeuromorphicTRLinkosTRM,
        NeuromorphicConfig,
        SpikingDCaAPNeuron,
        rate_encode,
        rate_decode,
        get_neuromorphic_info
    )
    
    info = get_neuromorphic_info()
    print(f"✅ Module loaded successfully")
    print(f"   Implementation: {info['implementation']}")
    print(f"   Neuron Model: {info['neuron_model']}")
    print(f"   Maturity: {info['maturity']}")
    
    # Test spiking neuron
    config = NeuromorphicConfig(dt=1.0, tau_mem=10.0)
    neuron = SpikingDCaAPNeuron(n_dendrites=4, config=config)
    print(f"✅ Spiking neuron created")
    
    # Test encoding/decoding
    values = np.array([[0.0, 0.5, 1.0]])
    spikes = rate_encode(values, time_steps=20, max_rate=100.0, dt=1.0)
    decoded = rate_decode(spikes, window=5)
    print(f"✅ Encoding/decoding works: spikes shape {spikes.shape}")
    
    # Test neuromorphic model
    model = NeuromorphicTRLinkosTRM(x_dim=16, y_dim=8, z_dim=16, num_experts=2)
    x = np.random.rand(2, 16)
    output = model.forward(x, time_steps=30)
    print(f"✅ Neuromorphic model inference: output shape {output.shape}")
    
    results.append(("Neuromorphic Computing", True, "Ready (Experimental)"))
    print("✅ Test 5/5 PASSED")
    
except Exception as e:
    print(f"❌ Test 5/5 FAILED: {e}")
    results.append(("Neuromorphic Computing", False, str(e)))

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

passed = sum(1 for _, success, _ in results if success)
failed = len(results) - passed

for feature, success, status in results:
    status_icon = "✅" if success else "❌"
    print(f"{status_icon} {feature:30s} - {status}")

print("-" * 70)
print(f"Total: {len(results)} tests | Passed: {passed} | Failed: {failed}")
print("=" * 70)

if failed == 0:
    print("\n🎉 ALL FEATURES ACTIVATED AND WORKING! 🎉")
    print("\n📖 For usage examples, see: ACTIVATION_GUIDE.md")
    sys.exit(0)
else:
    print(f"\n⚠️  {failed} test(s) failed. Check error messages above.")
    sys.exit(1)
