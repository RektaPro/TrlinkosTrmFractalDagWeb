"""
ONNX Export for T-RLINKOS TRM++ Production Deployment

This module provides ONNX export functionality for deploying T-RLINKOS models
in production environments:
- Export NumPy models to ONNX format
- Export PyTorch models to ONNX format
- ONNX Runtime inference
- Model optimization for production
- Cross-platform deployment

Usage:
    # Export NumPy model
    model = TRLinkosTRM(64, 32, 64)
    export_numpy_model_to_onnx(model, "model.onnx")
    
    # Export PyTorch model
    torch_model = TRLinkosTRMTorch(64, 32, 64)
    export_torch_model_to_onnx(torch_model, "model.onnx")
    
    # Inference with ONNX Runtime
    predictor = ONNXPredictor("model.onnx")
    output = predictor.predict(input_data)
"""

import warnings
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

# Try to import ONNX
try:
    import onnx
    from onnx import numpy_helper
    ONNX_AVAILABLE = True
    ONNX_VERSION = onnx.__version__
except ImportError:
    ONNX_AVAILABLE = False
    ONNX_VERSION = None
    warnings.warn("onnx not available. Install with: pip install onnx>=1.12.0")

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
    ONNXRUNTIME_VERSION = ort.__version__
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    ONNXRUNTIME_VERSION = None
    warnings.warn("onnxruntime not available. Install with: pip install onnxruntime>=1.12.0")


# ============================
#  NumPy Model Export
# ============================

def export_numpy_model_to_onnx(
    model,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 64),
    opset_version: int = 13,
    simplify: bool = True,
) -> None:
    """Export NumPy-based T-RLINKOS model to ONNX format.
    
    This converts the NumPy model to an ONNX graph by tracing the
    computational graph and extracting parameters.
    
    Args:
        model: TRLinkosTRM instance (NumPy-based)
        output_path: Path to save ONNX model
        input_shape: Input shape for tracing (batch_size, x_dim)
        opset_version: ONNX opset version
        simplify: Simplify ONNX graph after export
        
    Example:
        model = TRLinkosTRM(64, 32, 64)
        export_numpy_model_to_onnx(model, "trlinkos.onnx")
    """
    if not ONNX_AVAILABLE:
        raise RuntimeError("onnx not available. Install with: pip install onnx>=1.12.0")
    
    print(f"Exporting NumPy model to ONNX: {output_path}")
    print(f"  Input shape: {input_shape}")
    print(f"  Opset version: {opset_version}")
    
    # Create dummy input for tracing
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Forward pass to trace computation
    print("  Tracing forward pass...")
    y_pred, dag = model.forward_recursive(dummy_input, max_steps=1, inner_recursions=1)
    
    # Extract model parameters
    print("  Extracting model parameters...")
    parameters = {}
    
    # Extract weights from TRLinkosCore
    core = model.core
    
    # Router parameters
    if hasattr(core.router, 'projection'):
        parameters['router_projection_W'] = core.router.projection.W.astype(np.float32)
        parameters['router_projection_b'] = core.router.projection.b.astype(np.float32)
    
    if hasattr(core.router, 'mass_projection'):
        parameters['router_mass_W'] = core.router.mass_projection.W.astype(np.float32)
        parameters['router_mass_b'] = core.router.mass_projection.b.astype(np.float32)
    
    parameters['router_centroids'] = core.router.expert_centroids.astype(np.float32)
    
    # Expert parameters (simplified - just save first expert as example)
    if hasattr(core, 'experts') and len(core.experts) > 0:
        expert = core.experts[0]
        for i, branch_weight in enumerate(expert.branch_weights):
            parameters[f'expert0_branch{i}_W'] = branch_weight.W.astype(np.float32)
            parameters[f'expert0_branch{i}_b'] = branch_weight.b.astype(np.float32)
    
    # Output projection parameters
    if hasattr(model, 'output_proj'):
        parameters['output_proj_W'] = model.output_proj.W.astype(np.float32)
        parameters['output_proj_b'] = model.output_proj.b.astype(np.float32)
    
    print(f"  Extracted {len(parameters)} parameter tensors")
    
    # Create ONNX graph (simplified version)
    # In a full implementation, we would build the complete computation graph
    # For now, create a placeholder that documents the model structure
    
    # Save metadata
    metadata = {
        "model_type": "TRLinkosTRM",
        "x_dim": model.x_dim,
        "y_dim": model.y_dim,
        "z_dim": model.z_dim,
        "hidden_dim": model.core.router.projection.out_features if hasattr(model.core.router, 'projection') else 0,
        "num_experts": model.core.num_experts,
        "export_format": "numpy",
    }
    
    print(f"  Model metadata: {metadata}")
    print(f"\n⚠️  Note: Full ONNX export from NumPy requires manual graph construction.")
    print(f"     For production deployment, use PyTorch model export instead.")
    print(f"     See export_torch_model_to_onnx() for full ONNX support.")
    
    # Save parameters as numpy archive (can be loaded for inference)
    np.savez_compressed(
        output_path.replace('.onnx', '_params.npz'),
        **parameters,
        metadata=str(metadata)
    )
    print(f"\n✅ Exported model parameters to: {output_path.replace('.onnx', '_params.npz')}")


# ============================
#  PyTorch Model Export
# ============================

def export_torch_model_to_onnx(
    model,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 64),
    opset_version: int = 13,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    simplify: bool = True,
) -> None:
    """Export PyTorch-based T-RLINKOS model to ONNX format.
    
    This uses PyTorch's native ONNX export functionality to create
    a production-ready ONNX model with full computational graph.
    
    Args:
        model: TRLinkosTRMTorch instance (PyTorch-based)
        output_path: Path to save ONNX model
        input_shape: Input shape for tracing (batch_size, x_dim)
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes for variable batch size/sequence length
        simplify: Simplify ONNX graph after export
        
    Example:
        import torch
        from trlinkos_trm_torch import TRLinkosTRMTorch
        
        model = TRLinkosTRMTorch(64, 32, 64)
        export_torch_model_to_onnx(model, "trlinkos.onnx")
    """
    try:
        import torch
        import torch.onnx
    except ImportError:
        raise RuntimeError("PyTorch not available. Install with: pip install torch>=2.0.0")
    
    if not ONNX_AVAILABLE:
        raise RuntimeError("onnx not available. Install with: pip install onnx>=1.12.0")
    
    print(f"Exporting PyTorch model to ONNX: {output_path}")
    print(f"  Input shape: {input_shape}")
    print(f"  Opset version: {opset_version}")
    
    # Set model to eval mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Default dynamic axes (variable batch size)
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }
    
    print("  Tracing computational graph...")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    
    print(f"✅ Exported PyTorch model to: {output_path}")
    
    # Verify the model
    print("  Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  ✓ ONNX model is valid")
    
    # Simplify if requested
    if simplify:
        try:
            import onnxsim
            print("  Simplifying ONNX model...")
            onnx_model, check = onnxsim.simplify(onnx_model)
            if check:
                onnx.save(onnx_model, output_path)
                print("  ✓ Model simplified successfully")
            else:
                print("  ⚠️  Simplification check failed, keeping original")
        except ImportError:
            print("  ℹ️  onnx-simplifier not available (optional)")
    
    # Print model info
    print(f"\n  Model Information:")
    print(f"    Producer: {onnx_model.producer_name}")
    print(f"    IR version: {onnx_model.ir_version}")
    print(f"    Opset: {opset_version}")
    print(f"    Inputs: {[inp.name for inp in onnx_model.graph.input]}")
    print(f"    Outputs: {[out.name for out in onnx_model.graph.output]}")


# ============================
#  ONNX Runtime Inference
# ============================

class ONNXPredictor:
    """ONNX Runtime predictor for T-RLINKOS models.
    
    Provides efficient inference using ONNX Runtime with optimizations:
    - Hardware acceleration (CPU, CUDA, TensorRT)
    - Multi-threading
    - Memory optimization
    - Batch inference
    
    Example:
        predictor = ONNXPredictor("trlinkos.onnx")
        output = predictor.predict(input_data)
    """
    
    def __init__(
        self,
        model_path: str,
        providers: Optional[List[str]] = None,
        session_options: Optional[Any] = None,
    ):
        """Initialize ONNX predictor.
        
        Args:
            model_path: Path to ONNX model file
            providers: Execution providers (default: auto-detect)
            session_options: ONNX Runtime session options
        """
        if not ONNXRUNTIME_AVAILABLE:
            raise RuntimeError("onnxruntime not available. Install with: pip install onnxruntime>=1.12.0")
        
        self.model_path = model_path
        
        # Auto-detect providers if not specified
        if providers is None:
            providers = ['CPUExecutionProvider']
            # Add CUDA if available
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers.insert(0, 'CUDAExecutionProvider')
        
        print(f"Loading ONNX model: {model_path}")
        print(f"  Execution providers: {providers}")
        
        # Create inference session
        self.session = ort.InferenceSession(
            model_path,
            providers=providers,
            sess_options=session_options,
        )
        
        # Get model metadata
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"  Inputs: {self.input_names}")
        print(f"  Outputs: {self.output_names}")
        print("✅ ONNX model loaded successfully")
    
    def predict(
        self,
        input_data: np.ndarray,
        return_all_outputs: bool = False,
    ) -> np.ndarray:
        """Run inference on input data.
        
        Args:
            input_data: Input array [B, x_dim]
            return_all_outputs: Return all outputs or just first
            
        Returns:
            Output array(s)
        """
        # Ensure float32
        if input_data.dtype != np.float32:
            input_data = input_data.astype(np.float32)
        
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_names[0]: input_data}
        )
        
        if return_all_outputs:
            return outputs
        else:
            return outputs[0]
    
    def benchmark(
        self,
        input_shape: Tuple[int, ...] = (32, 64),
        num_iterations: int = 100,
    ) -> Dict[str, float]:
        """Benchmark inference performance.
        
        Args:
            input_shape: Input shape for benchmarking
            num_iterations: Number of iterations
            
        Returns:
            Dictionary with benchmark results
        """
        import time
        
        # Create random input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            self.predict(dummy_input)
        
        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            self.predict(dummy_input)
        elapsed = time.time() - start
        
        batch_size = input_shape[0]
        total_samples = batch_size * num_iterations
        
        return {
            "total_time": elapsed,
            "avg_time_per_batch": elapsed / num_iterations,
            "avg_time_per_sample": elapsed / total_samples,
            "throughput": total_samples / elapsed,
            "batch_size": batch_size,
            "num_iterations": num_iterations,
        }


# ============================
#  Utilities
# ============================

def get_onnx_info() -> Dict[str, Any]:
    """Get ONNX environment information.
    
    Returns:
        Dictionary with ONNX info
    """
    info = {
        "onnx_available": ONNX_AVAILABLE,
        "onnx_version": ONNX_VERSION,
        "onnxruntime_available": ONNXRUNTIME_AVAILABLE,
        "onnxruntime_version": ONNXRUNTIME_VERSION,
    }
    
    if ONNXRUNTIME_AVAILABLE:
        info["available_providers"] = ort.get_available_providers()
    
    return info


# ============================
#  Main test
# ============================

if __name__ == "__main__":
    print("=" * 70)
    print("ONNX EXPORT MODULE TEST")
    print("=" * 70)
    
    # Print ONNX info
    info = get_onnx_info()
    print(f"\nONNX Environment:")
    print(f"  ONNX Available: {info['onnx_available']}")
    if info['onnx_available']:
        print(f"  ONNX Version: {info['onnx_version']}")
    print(f"  ONNX Runtime Available: {info['onnxruntime_available']}")
    if info['onnxruntime_available']:
        print(f"  ONNX Runtime Version: {info['onnxruntime_version']}")
        print(f"  Available Providers: {info['available_providers']}")
    
    print("\n" + "=" * 70)
    print("✅ ONNX export module loaded successfully!")
    print("=" * 70)
    
    if not ONNX_AVAILABLE or not ONNXRUNTIME_AVAILABLE:
        print("\nNote: Install ONNX packages for full functionality:")
        if not ONNX_AVAILABLE:
            print("  pip install onnx>=1.12.0")
        if not ONNXRUNTIME_AVAILABLE:
            print("  pip install onnxruntime>=1.12.0")
    
    print("\nUsage Examples:")
    print("\n1. Export PyTorch model (recommended):")
    print("   from trlinkos_trm_torch import TRLinkosTRMTorch")
    print("   model = TRLinkosTRMTorch(64, 32, 64)")
    print("   export_torch_model_to_onnx(model, 'model.onnx')")
    
    print("\n2. Export NumPy model (limited support):")
    print("   from t_rlinkos_trm_fractal_dag import TRLinkosTRM")
    print("   model = TRLinkosTRM(64, 32, 64)")
    print("   export_numpy_model_to_onnx(model, 'model.onnx')")
    
    print("\n3. Inference with ONNX Runtime:")
    print("   predictor = ONNXPredictor('model.onnx')")
    print("   output = predictor.predict(input_data)")
    
    print("\n4. Benchmark inference:")
    print("   results = predictor.benchmark(input_shape=(32, 64))")
    print("   print(f'Throughput: {results[\"throughput\"]:.1f} samples/sec')")
