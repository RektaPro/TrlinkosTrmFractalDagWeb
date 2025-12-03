"""
Multi-GPU Distributed Training Support for T-RLINKOS TRM++

This module provides utilities for distributed training across multiple GPUs:
- DataParallel for single-node multi-GPU training
- DistributedDataParallel for multi-node multi-GPU training
- Automatic device placement and synchronization
- Gradient accumulation for large batch sizes

Usage:
    # Single-node multi-GPU
    model = TRLinkosTRMTorch(...)
    model = wrap_data_parallel(model, device_ids=[0, 1, 2, 3])
    
    # Multi-node multi-GPU
    setup_distributed(rank=0, world_size=4)
    model = TRLinkosTRMTorch(...)
    model = wrap_distributed_data_parallel(model)
    # ... training ...
    cleanup_distributed()
"""

import os
from typing import Optional, List, Dict, Any, TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DataParallel, DistributedDataParallel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for type hints
    class nn:
        class Module:
            pass
    warnings.warn("PyTorch not available. Multi-GPU support requires torch>=2.0.0")


# ============================
#  Device Management
# ============================

def get_available_gpus() -> List[int]:
    """Get list of available GPU device IDs.
    
    Returns:
        List of GPU device IDs (empty list if no GPUs available)
    """
    if not TORCH_AVAILABLE:
        return []
    
    if not torch.cuda.is_available():
        return []
    
    return list(range(torch.cuda.device_count()))


def get_device_info() -> Dict[str, Any]:
    """Get information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    if not TORCH_AVAILABLE:
        return {
            "torch_available": False,
            "cuda_available": False,
            "num_gpus": 0,
            "gpu_names": [],
        }
    
    info = {
        "torch_available": True,
        "cuda_available": torch.cuda.is_available(),
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": [],
        "distributed_available": dist.is_available(),
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            info["gpu_names"].append(torch.cuda.get_device_name(i))
    
    return info


# ============================
#  DataParallel (Single-node Multi-GPU)
# ============================

def wrap_data_parallel(
    model: nn.Module,
    device_ids: Optional[List[int]] = None,
    output_device: Optional[int] = None,
) -> nn.Module:
    """Wrap model with DataParallel for single-node multi-GPU training.
    
    DataParallel splits the batch across multiple GPUs and synchronizes
    gradients after each backward pass. Good for single-node setups.
    
    Args:
        model: PyTorch model to wrap
        device_ids: List of GPU device IDs to use (default: all available)
        output_device: GPU to gather outputs on (default: first GPU)
        
    Returns:
        Wrapped model (or original model if only 1 GPU or no CUDA)
        
    Example:
        model = TRLinkosTRMTorch(64, 32, 64)
        model = wrap_data_parallel(model, device_ids=[0, 1, 2, 3])
        model = model.cuda()
        
        # Training loop
        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
    """
    if not TORCH_AVAILABLE:
        warnings.warn("PyTorch not available. Returning original model.")
        return model
    
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available. Returning original model.")
        return model
    
    available_gpus = get_available_gpus()
    if len(available_gpus) <= 1:
        warnings.warn(f"Only {len(available_gpus)} GPU available. DataParallel not needed.")
        return model
    
    # Use all GPUs by default
    if device_ids is None:
        device_ids = available_gpus
    
    # Use first GPU as output device by default
    if output_device is None:
        output_device = device_ids[0]
    
    print(f"Wrapping model with DataParallel on GPUs: {device_ids}")
    print(f"Output device: {output_device}")
    
    return DataParallel(model, device_ids=device_ids, output_device=output_device)


# ============================
#  DistributedDataParallel (Multi-node Multi-GPU)
# ============================

def setup_distributed(
    rank: int,
    world_size: int,
    backend: str = "nccl",
    init_method: Optional[str] = None,
) -> None:
    """Initialize distributed training environment.
    
    Call this at the start of each process in multi-node distributed training.
    
    Args:
        rank: Rank of current process (0 to world_size-1)
        world_size: Total number of processes
        backend: Communication backend ('nccl' for GPU, 'gloo' for CPU)
        init_method: Initialization method (default: env://)
        
    Example:
        # In each process
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        setup_distributed(rank, world_size)
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. Cannot setup distributed training.")
    
    if not dist.is_available():
        raise RuntimeError("torch.distributed not available.")
    
    # Use environment variables by default
    if init_method is None:
        init_method = "env://"
    
    # Set CUDA device for this process
    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(rank % torch.cuda.device_count())
    
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    
    print(f"Initialized distributed training: rank={rank}, world_size={world_size}, backend={backend}")


def cleanup_distributed() -> None:
    """Clean up distributed training environment.
    
    Call this at the end of training to properly shut down.
    """
    if TORCH_AVAILABLE and dist.is_initialized():
        dist.destroy_process_group()
        print("Cleaned up distributed training.")


def wrap_distributed_data_parallel(
    model: nn.Module,
    device_id: Optional[int] = None,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """Wrap model with DistributedDataParallel for multi-node multi-GPU training.
    
    DDP provides better performance than DataParallel by:
    - Overlapping computation and communication
    - Reducing memory overhead per GPU
    - Supporting multi-node training
    
    Args:
        model: PyTorch model to wrap (should already be on correct device)
        device_id: GPU device ID for this process (default: current device)
        find_unused_parameters: Set True if model has unused parameters
        
    Returns:
        Wrapped model
        
    Example:
        # In each process
        setup_distributed(rank, world_size)
        
        model = TRLinkosTRMTorch(64, 32, 64).cuda()
        model = wrap_distributed_data_parallel(model)
        
        # Training loop (each process sees different data)
        for x, y in dataloader:
            x, y = x.cuda(), y.cuda()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        cleanup_distributed()
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available. Cannot wrap with DDP.")
    
    if not dist.is_initialized():
        raise RuntimeError("Distributed training not initialized. Call setup_distributed() first.")
    
    # Auto-detect device if not specified
    if device_id is None:
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
        else:
            device_id = None  # CPU
    
    print(f"Wrapping model with DistributedDataParallel (device_id={device_id})")
    
    return DistributedDataParallel(
        model,
        device_ids=[device_id] if device_id is not None else None,
        output_device=device_id,
        find_unused_parameters=find_unused_parameters,
    )


# ============================
#  Gradient Accumulation
# ============================

class GradientAccumulator:
    """Helper for gradient accumulation with multi-GPU training.
    
    Gradient accumulation allows training with larger effective batch sizes
    than can fit in GPU memory by accumulating gradients over multiple
    mini-batches before updating weights.
    
    Example:
        accumulator = GradientAccumulator(accumulation_steps=4)
        
        for i, (x, y) in enumerate(dataloader):
            x, y = x.cuda(), y.cuda()
            
            with accumulator.context(i):
                output = model(x)
                loss = criterion(output, y)
                accumulator.backward(loss, optimizer)
            
            if accumulator.should_step(i):
                optimizer.step()
                optimizer.zero_grad()
    """
    
    def __init__(self, accumulation_steps: int = 1):
        """Initialize gradient accumulator.
        
        Args:
            accumulation_steps: Number of mini-batches to accumulate
        """
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
    
    def context(self, step: int):
        """Get context manager for gradient scaling.
        
        Args:
            step: Current training step
            
        Returns:
            Context manager (no-op if accumulation_steps=1)
        """
        # Could add autocast here if needed
        class NoOpContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return NoOpContext()
    
    def backward(self, loss, optimizer=None):
        """Backward pass with gradient scaling.
        
        Automatically scales loss by 1/accumulation_steps.
        
        Args:
            loss: Loss tensor
            optimizer: Optional optimizer (for compatibility)
        """
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
    
    def should_step(self, step: int) -> bool:
        """Check if optimizer should step at this iteration.
        
        Args:
            step: Current training step
            
        Returns:
            True if should update weights
        """
        return (step + 1) % self.accumulation_steps == 0


# ============================
#  Utilities
# ============================

def synchronize_models(model: nn.Module) -> None:
    """Synchronize model parameters across all processes.
    
    Broadcasts model parameters from rank 0 to all other processes.
    Useful after model initialization to ensure all processes start
    with the same weights.
    
    Args:
        model: Model to synchronize
    """
    if not TORCH_AVAILABLE or not dist.is_initialized():
        return
    
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    
    print(f"Synchronized model parameters from rank 0")


def reduce_tensor(tensor, average: bool = True):
    """Reduce tensor across all processes.
    
    Args:
        tensor: Tensor to reduce
        average: If True, compute average; if False, compute sum
        
    Returns:
        Reduced tensor
    """
    if not TORCH_AVAILABLE or not dist.is_initialized():
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    
    if average:
        rt = rt / dist.get_world_size()
    
    return rt


# ============================
#  Main test
# ============================

if __name__ == "__main__":
    print("=" * 70)
    print("MULTI-GPU SUPPORT MODULE TEST")
    print("=" * 70)
    
    # Print device info
    info = get_device_info()
    print(f"\nDevice Information:")
    print(f"  PyTorch Available: {info['torch_available']}")
    print(f"  CUDA Available: {info['cuda_available']}")
    print(f"  Number of GPUs: {info['num_gpus']}")
    if info['gpu_names']:
        print(f"  GPU Names:")
        for i, name in enumerate(info['gpu_names']):
            print(f"    GPU {i}: {name}")
    print(f"  Distributed Available: {info.get('distributed_available', False)}")
    
    # Test available GPUs
    gpus = get_available_gpus()
    print(f"\nAvailable GPU IDs: {gpus}")
    
    # Test gradient accumulator
    print("\n--- Test: GradientAccumulator ---")
    accumulator = GradientAccumulator(accumulation_steps=4)
    print(f"Accumulation steps: {accumulator.accumulation_steps}")
    
    for i in range(8):
        should_step = accumulator.should_step(i)
        print(f"Step {i}: should_step={should_step}")
    
    print("\n" + "=" * 70)
    print("✅ Multi-GPU support module loaded successfully!")
    print("=" * 70)
    
    # Print usage examples
    print("\nUsage Examples:")
    print("\n1. Single-node multi-GPU (DataParallel):")
    print("   model = wrap_data_parallel(model, device_ids=[0, 1, 2, 3])")
    print("   model = model.cuda()")
    
    print("\n2. Multi-node multi-GPU (DistributedDataParallel):")
    print("   setup_distributed(rank, world_size)")
    print("   model = model.cuda()")
    print("   model = wrap_distributed_data_parallel(model)")
    print("   # ... training ...")
    print("   cleanup_distributed()")
    
    print("\n3. Gradient Accumulation:")
    print("   accumulator = GradientAccumulator(accumulation_steps=4)")
    print("   for i, (x, y) in enumerate(dataloader):")
    print("       with accumulator.context(i):")
    print("           loss = criterion(model(x), y)")
    print("           accumulator.backward(loss)")
    print("       if accumulator.should_step(i):")
    print("           optimizer.step()")
    print("           optimizer.zero_grad()")
