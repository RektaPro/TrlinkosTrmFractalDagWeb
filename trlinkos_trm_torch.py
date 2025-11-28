# trlinkos_trm_torch.py
"""
Implementation of T-RLINKOS TRM++ using PyTorch.

This version is harmonized with the NumPy implementation (t_rlinkos_trm_fractal_dag.py)
and is GPU-ready for training.

Architecture (same as NumPy version):
- dCaAP-inspired neuron cell (DCaAPCellTorch) -> corresponds to DCaAPCell
- Torque Clustering router (TorqueRouterTorch) -> corresponds to TorqueRouter
- TRM core (TRLinkosCoreTorch) -> corresponds to TRLinkosCore
- TRLinkosTRMTorch high-level model -> corresponds to TRLinkosTRM

The key difference from the NumPy version:
- Uses PyTorch tensors and autograd for gradient computation
- Supports GPU acceleration via device management
- Uses nn.Module for parameter management
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================
#  Helper functions (NumPy equivalent: gelu)
# ============================


def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU activation function.

    Corresponds to: gelu() in t_rlinkos_trm_fractal_dag.py

    Args:
        x: Input tensor of any shape

    Returns:
        GELU activation applied element-wise
    """
    return F.gelu(x)


# ============================
#  dCaAP-inspired neuron cell (corresponds to DCaAPCell in NumPy)
# ============================


def dcaap_activation(
    x: torch.Tensor,
    threshold: torch.Tensor
) -> torch.Tensor:
    """dCaAP activation function (dendritic Calcium Action Potential).

    Corresponds to: dcaap_activation() in t_rlinkos_trm_fractal_dag.py

    Based on Gidon et al., Science 2020 and Hashemi & Tetzlaff, bioRxiv 2025.
    Reference: https://www.biorxiv.org/content/10.1101/2025.06.10.658823v1

    The dCaAP function is non-monotone and enables anti-coincidence detection:
    - Maximum amplitude near the threshold
    - Reduced amplitude for very strong stimuli
    - Zero for stimuli below the threshold

    dCaAP(x) = 4 * σ(x-θ) * (1 - σ(x-θ)) * (x > θ)

    Args:
        x: Input tensor [B, num_branches, H] or [B, D]
        threshold: Threshold tensor, broadcastable with x

    Returns:
        dCaAP activation [same shape as x]
    """
    s = torch.sigmoid(x - threshold)
    # Non-monotone bell-shaped activation
    activation = 4.0 * s * (1.0 - s)
    # Mask values below threshold
    mask = (x > threshold).float()
    return activation * mask


class DCaAPCellTorch(nn.Module):
    """Neuron inspired by dCaAP (Gidon et al., Science 2020; Hashemi & Tetzlaff, bioRxiv 2025).

    Corresponds to: DCaAPCell in t_rlinkos_trm_fractal_dag.py

    Implements the dendritic calcium action potential model:
    - Multiple dendritic branches with local integration
    - Non-monotone dCaAP activation (anti-coincidence detection)
    - Calcium gate for temporal accumulation
    - Intrinsic XOR capability (unlike standard activations)

    Architecture:
    - Input: concat(x, y, z) distributed to dendritic branches
    - Dendritic branches: local integration with dCaAP activation
    - Calcium gate: accumulation + adaptive threshold
    - Output: new z (internal state)

    Args:
        input_dim: Dimension of concatenated input (x_dim + y_dim + z_dim)
        hidden_dim: Hidden dimension for branch outputs
        z_dim: Dimension of the internal state z
        num_branches: Number of dendritic branches (default: 4)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        z_dim: int,
        num_branches: int = 4
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.num_branches = num_branches

        # Synapses for each dendritic branch (combined into single matrix for efficiency)
        # Corresponds to: branch_weights list in NumPy version
        self.branch_weights = nn.Linear(input_dim, hidden_dim * num_branches)

        # Adaptive thresholds for each branch (learnable parameters)
        # Corresponds to: branch_thresholds in NumPy version
        self.branch_thresholds = nn.Parameter(
            torch.zeros(num_branches, hidden_dim)
        )

        # Somatic integration of branches
        # Corresponds to: soma_integration in NumPy version
        self.soma = nn.Linear(hidden_dim * num_branches, hidden_dim)

        # Calcium gate for temporal accumulation
        # Corresponds to: calcium_gate in NumPy version
        self.gate_linear = nn.Linear(hidden_dim, 1)

        # Projection to z space
        # Corresponds to: output_projection in NumPy version
        self.z_proj = nn.Linear(hidden_dim, z_dim)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with dCaAP mechanism.

        Implements dendritic integration with dCaAP activation:
        1. Distribute input to dendritic branches
        2. Local integration with non-monotone dCaAP activation
        3. Somatic summation of dendritic signals
        4. Calcium gate for state update

        Args:
            x: External input [B, dx]
            y: Current response [B, dy]
            z: Internal state [B, dz]

        Returns:
            z_next: New internal state [B, dz]
        """
        # 1) Concatenate inputs
        h_in = torch.cat([x, y, z], dim=-1)  # [B, input_dim]

        # 2) Dendritic branch integration
        h_branches = self.branch_weights(h_in)  # [B, hidden_dim * num_branches]
        h_branches = h_branches.view(-1, self.num_branches, self.hidden_dim)  # [B, num_branches, H]

        # 3) dCaAP activation with adaptive threshold
        h_activated = dcaap_activation(h_branches, self.branch_thresholds)  # [B, num_branches, H]

        # 4) Somatic integration
        h_branches_flat = h_activated.view(h_activated.size(0), -1)  # [B, H * num_branches]
        h_soma = gelu(self.soma(h_branches_flat))  # [B, H]

        # 5) Calcium gate for temporal accumulation
        gate = torch.sigmoid(self.gate_linear(h_soma))  # [B, 1]

        # 6) Projection to z space and gated update
        proposal = self.z_proj(h_soma)  # [B, dz]

        # Interpolation controlled by calcium gate
        z_next = z + gate * (proposal - z)
        return z_next


# ============================
#  Torque Clustering Router (corresponds to TorqueRouter in NumPy)
# ============================


class TorqueRouterTorch(nn.Module):
    """Router based on Torque Clustering (Yang & Lin, TPAMI 2025).

    Corresponds to: TorqueRouter in t_rlinkos_trm_fractal_dag.py

    Implements the Torque Clustering concept for expert routing:
    - Torque = Mass × R² (distance squared)
    - Mass: local density based on representations
    - R²: squared distance to expert centroids

    Reference: https://github.com/JieYangBruce/TorqueClustering

    The routing formula: w_i = softmax(mass / (R_i² + ε))
    - Projects concat(x, y, z) to latent space
    - Computes squared distances to expert centroids
    - Applies mass-weighted affinity scores

    Args:
        x_dim: Dimension of input x
        y_dim: Dimension of response y
        z_dim: Dimension of internal state z
        num_experts: Number of experts to route to (default: 4)
        proj_dim: Dimension of projection space (default: 64)
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        z_dim: int,
        num_experts: int = 4,
        proj_dim: int = 64
    ) -> None:
        super().__init__()
        self.input_dim = x_dim + y_dim + z_dim
        self.num_experts = num_experts
        self.proj_dim = proj_dim

        # Projection for computing representations
        # Corresponds to: projection in NumPy version
        self.proj = nn.Linear(self.input_dim, proj_dim)

        # Parameters for local mass computation
        # Corresponds to: mass_projection in NumPy version
        self.mass_proj = nn.Linear(proj_dim, 1)

        # Expert centroids (learnable)
        # Corresponds to: expert_centroids in NumPy version
        self.centroids = nn.Parameter(torch.randn(num_experts, proj_dim))

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        """Compute routing weights based on Torque Clustering.

        Torque = Mass × R² for each (sample, expert) pair.
        Experts with high torque receive more weight.

        Args:
            x: External input [B, dx]
            y: Current response [B, dy]
            z: Internal state [B, dz]

        Returns:
            weights: Normalized routing weights [B, num_experts]
        """
        # 1) Concatenate inputs
        h_in = torch.cat([x, y, z], dim=-1)  # [B, input_dim]

        # 2) Project to representation space
        h_raw = gelu(self.proj(h_in))  # [B, proj_dim]

        # 3) Compute squared distances (R²) to expert centroids
        # ||h - c||² = ||h||² + ||c||² - 2*h@c.T
        h_norm2 = (h_raw ** 2).sum(dim=-1, keepdim=True)  # [B, 1]
        c_norm2 = (self.centroids ** 2).sum(dim=-1).unsqueeze(0)  # [1, E]
        cross = -2.0 * torch.matmul(h_raw, self.centroids.t())  # [B, E]
        dist2 = h_norm2 + c_norm2 + cross  # [B, E]

        # 4) Compute local mass (density)
        # Corresponds to: _compute_mass in NumPy version
        mass = F.softplus(self.mass_proj(h_raw)) + 1.0  # [B, 1], minimum 1.0

        # 5) Compute affinity score based on Torque Clustering
        # score = mass / (R² + ε), closer experts get higher scores
        eps = 1e-6
        affinity = mass / (dist2 + eps)  # [B, E]

        # 6) Normalize via softmax to get routing weights
        weights = F.softmax(affinity, dim=-1)  # [B, E]

        return weights


# ============================
#  TRM Core (corresponds to TRLinkosCore in NumPy)
# ============================


class TRLinkosCoreTorch(nn.Module):
    """Core of the Tiny Recursive Model T-RLINKOS (PyTorch version).

    Corresponds to: TRLinkosCore in t_rlinkos_trm_fractal_dag.py

    Implements:
    - Multiple dCaAP experts controlled by TorqueRouter
    - Response update module for y

    Args:
        x_dim: Dimension of input x
        y_dim: Dimension of response y
        z_dim: Dimension of internal state z
        hidden_dim: Hidden dimension for experts (default: 64)
        num_experts: Number of dCaAP experts (default: 4)
        num_branches: Number of dendritic branches per expert (default: 4)
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        z_dim: int,
        hidden_dim: int = 64,
        num_experts: int = 4,
        num_branches: int = 4,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts

        # Router based on Torque Clustering
        # Corresponds to: router in NumPy version
        self.router = TorqueRouterTorch(x_dim, y_dim, z_dim, num_experts=num_experts)

        # dCaAP experts
        # Corresponds to: experts list in NumPy version
        input_dim = x_dim + y_dim + z_dim
        self.experts = nn.ModuleList([
            DCaAPCellTorch(input_dim, hidden_dim, z_dim, num_branches=num_branches)
            for _ in range(num_experts)
        ])

        # Response update from [y, z]
        # Corresponds to: answer_dense1 and answer_dense2 in NumPy version
        self.answer_dense1 = nn.Linear(y_dim + z_dim, hidden_dim)
        self.answer_dense2 = nn.Linear(hidden_dim, y_dim)

    def step_reasoning(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        inner_recursions: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One TRM "recursion" = multiple internal z updates, then y update.

        Corresponds to: step_reasoning() in NumPy version

        Args:
            x: Encoded input [B, dx]
            y: Current response state [B, dy]
            z: Current internal state [B, dz]
            inner_recursions: Number of internal recursions

        Returns:
            y_next: New response [B, dy]
            z: New internal state [B, dz]
        """
        # 1) Inner recursion on z
        for _ in range(inner_recursions):
            # Compute routing weights
            weights = self.router(x, y, z)  # [B, E]

            # Apply all experts and stack results
            z_candidates = []
            for expert in self.experts:
                z_next_e = expert(x, y, z)  # [B, dz]
                z_candidates.append(z_next_e)
            z_stack = torch.stack(z_candidates, dim=1)  # [B, E, dz]

            # Torque-weighted mixture
            weights_exp = weights.unsqueeze(-1)  # [B, E, 1]
            z = (weights_exp * z_stack).sum(dim=1)  # [B, dz]

        # 2) Response update
        yz = torch.cat([y, z], dim=-1)  # [B, y_dim + z_dim]
        h = gelu(self.answer_dense1(yz))  # [B, hidden_dim]
        y = self.answer_dense2(h)  # [B, y_dim]

        return y, z


# ============================
#  Main Model (corresponds to TRLinkosTRM in NumPy)
# ============================


class TRLinkosTRMTorch(nn.Module):
    """T-RLINKOS: Tiny Recursive Model ++ (PyTorch version).

    Corresponds to: TRLinkosTRM in t_rlinkos_trm_fractal_dag.py

    Recursive architecture for reasoning:
    - TRM Core (TRLinkosCoreTorch) with dCaAP experts and Torque router
    - GPU-ready with device management

    Note: This version doesn't include the Merkle-DAG tracing for simplicity,
    as it's primarily designed for training. The NumPy version includes
    full DAG support for interpretability.

    Args:
        x_dim: Dimension of input x
        y_dim: Dimension of response y
        z_dim: Dimension of internal state z
        hidden_dim: Hidden dimension for experts (default: 64)
        num_experts: Number of dCaAP experts (default: 4)
        num_branches: Number of dendritic branches per expert (default: 4)
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        z_dim: int,
        hidden_dim: int = 64,
        num_experts: int = 4,
        num_branches: int = 4,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        # Simple encoder for x
        # Corresponds to: x_encoder in NumPy version
        self.x_encoder = nn.Linear(x_dim, x_dim)

        # TRM Core
        # Corresponds to: core in NumPy version
        self.core = TRLinkosCoreTorch(
            x_dim=x_dim,
            y_dim=y_dim,
            z_dim=z_dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_branches=num_branches,
        )

        # Initial states (learnable parameters)
        # Corresponds to: y_init and z_init in NumPy version
        self.y_init = nn.Parameter(torch.zeros(1, y_dim))
        self.z_init = nn.Parameter(torch.zeros(1, z_dim))

    def forward(
        self,
        x: torch.Tensor,
        max_steps: int = 8,
        inner_recursions: int = 2,
    ) -> torch.Tensor:
        """Complete recursive reasoning loop.

        Corresponds to: forward_recursive() in NumPy version
        (simplified without DAG tracing for training efficiency)

        Args:
            x: Input tensor [B, x_dim]
            max_steps: Maximum reasoning steps (default: 8)
            inner_recursions: Inner recursions per step (default: 2)

        Returns:
            y_final: Final response [B, y_dim]
        """
        B = x.size(0)
        device = x.device

        # Encode input
        x_enc = self.x_encoder(x)  # [B, x_dim]

        # Initialize states with learnable parameters
        # Expand to batch size
        y = self.y_init.expand(B, -1)  # [B, y_dim]
        z = self.z_init.expand(B, -1)  # [B, z_dim]

        # Recursive reasoning loop
        for _ in range(max_steps):
            y, z = self.core.step_reasoning(x_enc, y, z, inner_recursions=inner_recursions)

        return y

    def to(self, device: torch.device) -> "TRLinkosTRMTorch":
        """Move model to specified device.

        Ensures all parameters and buffers are on the target device.

        Args:
            device: Target device (e.g., torch.device("cuda") or "cpu")

        Returns:
            self for method chaining
        """
        return super().to(device)


# ============================
#  XOR Training Example
# ============================


def make_xor_dataset(
    n_samples: int = 1024,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create XOR dataset for training.

    The XOR problem is a classic test for non-linear models.
    It cannot be solved by a single linear layer.

    Args:
        n_samples: Number of samples to generate
        device: Target device for tensors (default: CPU)

    Returns:
        X: Input tensor [n_samples, 2] with binary values {0, 1}
        y: Target tensor [n_samples, 1] with XOR results
    """
    if device is None:
        device = torch.device("cpu")

    X = torch.randint(0, 2, (n_samples, 2), device=device).float()
    y = ((X[:, 0] != X[:, 1]).float()).unsqueeze(-1)  # XOR: 1 if different, 0 if same
    return X, y


def train_xor_example(
    device: Optional[torch.device] = None,
    n_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    verbose: bool = True
) -> TRLinkosTRMTorch:
    """Train a TRM model on the XOR problem.

    This is a minimal training example demonstrating:
    - Dataset creation
    - Model instantiation with device management
    - Training loop with optimizer and loss
    - Simple progress logging

    Args:
        device: Target device (default: auto-detect CUDA/CPU)
        n_epochs: Number of training epochs (default: 50)
        batch_size: Batch size for training (default: 64)
        learning_rate: Learning rate for Adam optimizer (default: 1e-3)
        verbose: Whether to print progress (default: True)

    Returns:
        Trained model
    """
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    # Auto-detect device if not specified
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"Training XOR example on device: {device}")

    # Create model
    model = TRLinkosTRMTorch(
        x_dim=2,
        y_dim=1,
        z_dim=8,
        hidden_dim=32,
        num_experts=4,
        num_branches=4,
    ).to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create dataset
    X, y = make_xor_dataset(2048, device=device)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in dataloader:
            optimizer.zero_grad()

            # Forward pass
            logits = model(xb, max_steps=6, inner_recursions=2)  # [B, 1]
            loss = criterion(logits, yb)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        # Log progress
        avg_loss = total_loss / total
        acc = correct / total
        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == n_epochs):
            print(f"Epoch {epoch:03d} | Loss={avg_loss:.4f} | Acc={acc:.4f}")

    return model


def test_xor_model(model: TRLinkosTRMTorch, device: Optional[torch.device] = None) -> None:
    """Test a trained model on all XOR combinations.

    Args:
        model: Trained TRM model
        device: Device for testing (default: auto-detect)
    """
    if device is None:
        device = next(model.parameters()).device

    # All XOR combinations
    X_test = torch.tensor([
        [0., 0.],  # Expected: 0
        [0., 1.],  # Expected: 1
        [1., 0.],  # Expected: 1
        [1., 1.],  # Expected: 0
    ], device=device)

    expected = torch.tensor([[0.], [1.], [1.], [0.]], device=device)

    with torch.no_grad():
        logits = model(X_test, max_steps=6, inner_recursions=2)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

    print("\nXOR Test Results:")
    print("=" * 40)
    print(f"{'Input':<15} {'Expected':<10} {'Predicted':<10} {'Prob':<10}")
    print("-" * 40)
    for i in range(4):
        x_str = f"[{int(X_test[i, 0].item())}, {int(X_test[i, 1].item())}]"
        exp_str = str(int(expected[i, 0].item()))
        pred_str = str(int(preds[i, 0].item()))
        prob_str = f"{probs[i, 0].item():.3f}"
        print(f"{x_str:<15} {exp_str:<10} {pred_str:<10} {prob_str:<10}")
    print("=" * 40)

    accuracy = (preds == expected).float().mean().item()
    print(f"Test Accuracy: {accuracy * 100:.1f}%")


if __name__ == "__main__":
    """Minimal XOR training example.

    This demonstrates:
    1. Creating a TRM model with proper device management
    2. Training on the XOR problem (a classic non-linear test)
    3. Evaluating the trained model

    Run with: python trlinkos_trm_torch.py
    """
    print("=" * 60)
    print("T-RLINKOS TRM++ PyTorch - XOR Training Example")
    print("=" * 60)
    print()

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    # Train model
    print("Training...")
    print("-" * 40)
    model = train_xor_example(
        device=device,
        n_epochs=50,
        batch_size=64,
        learning_rate=1e-3,
        verbose=True
    )

    # Test model
    print()
    test_xor_model(model, device)

    print()
    print("Training complete!")
    print("=" * 60)
