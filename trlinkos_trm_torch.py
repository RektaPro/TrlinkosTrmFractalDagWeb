# trlinkos_trm_torch.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearTorch(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def gelu(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x)


class DCaAPCellTorch(nn.Module):
    """
    Optimized PyTorch version of DCaAPCell.

    - Input: concat(x, y, z) -> [B, input_dim]
    - Local dendritic branches
    - Non-monotonic dCaAP activation
    - Calcium gate for updating z
    """

    def __init__(self, input_dim: int, hidden_dim: int, z_dim: int, num_branches: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.num_branches = num_branches

        # Combine branch computations into a single matrix operation
        self.branch_weights = nn.Linear(input_dim, hidden_dim * num_branches)
        self.branch_thresholds = nn.Parameter(
            torch.zeros(num_branches, hidden_dim)
        )

        # Somatic integration
        self.soma = nn.Linear(hidden_dim * num_branches, hidden_dim)
        # Calcium gate
        self.gate_linear = nn.Linear(hidden_dim, 1)
        # Projection to z
        self.z_proj = nn.Linear(hidden_dim, z_dim)

    def dcaap_activation(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        # x: [B, num_branches, H], theta: [num_branches, H]
        s = torch.sigmoid(x - theta)
        return 4.0 * s * (1.0 - s) * (x > theta)

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x: [B, dx]
        y: [B, dy]
        z: [B, dz]
        Returns: z_next [B, dz]
        """
        h_in = torch.cat([x, y, z], dim=-1)  # [B, D]
        h_branches = self.branch_weights(h_in).view(-1, self.num_branches, self.hidden_dim)  # [B, num_branches, H]
        h_activated = self.dcaap_activation(h_branches, self.branch_thresholds)  # [B, num_branches, H]

        h_branches_flat = h_activated.view(h_activated.size(0), -1)  # [B, H * num_branches]
        h_soma = gelu(self.soma(h_branches_flat))  # [B, H]

        gate = torch.sigmoid(self.gate_linear(h_soma))  # [B, 1]
        proposal = self.z_proj(h_soma)  # [B, dz]

        z_next = z + gate * (proposal - z)
        return z_next


class TorqueRouterTorch(nn.Module):
    """
    Version PyTorch de TorqueRouter.

    - Projette concat(x,y,z) vers un espace latent
    - Calcule distances quadratiques vers des centroïdes experts
    - Masse locale m
    - Poids de routing w_i = softmax(m / (R_i^2 + eps))
    """

    def __init__(self, x_dim: int, y_dim: int, z_dim: int, num_experts: int = 4, proj_dim: int = 64):
        super().__init__()
        self.input_dim = x_dim + y_dim + z_dim
        self.num_experts = num_experts
        self.proj = nn.Linear(self.input_dim, proj_dim)
        self.mass_proj = nn.Linear(proj_dim, 1)
        # Centroïdes d'experts
        self.centroids = nn.Parameter(torch.randn(num_experts, proj_dim))

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x: [B, dx], y: [B, dy], z: [B, dz]
        Retour: weights [B, num_experts]
        """
        h_in = torch.cat([x, y, z], dim=-1)
        h_raw = gelu(self.proj(h_in))   # [B, P]

        # distances au carré [B, E]
        # ||h||^2
        h_norm2 = (h_raw ** 2).sum(dim=-1, keepdim=True)            # [B,1]
        # ||c||^2
        c_norm2 = (self.centroids ** 2).sum(dim=-1).unsqueeze(0)    # [1,E]
        # -2 h c^T
        cross = -2.0 * torch.matmul(h_raw, self.centroids.t())      # [B,E]
        dist2 = h_norm2 + c_norm2 + cross

        mass = F.softplus(self.mass_proj(h_raw)) + 1.0              # [B,1]
        eps = 1e-6
        affinity = mass / (dist2 + eps)                             # [B,E]

        weights = F.softmax(affinity, dim=-1)
        return weights


class TRLinkosCoreTorch(nn.Module):
    """
    Coeur TRM avec experts dCaAP et routing Torque (PyTorch).
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        z_dim: int,
        hidden_dim: int = 64,
        num_experts: int = 4,
        num_branches: int = 4,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts

        self.router = TorqueRouterTorch(x_dim, y_dim, z_dim, num_experts=num_experts)

        input_dim = x_dim + y_dim + z_dim
        self.experts = nn.ModuleList([
            DCaAPCellTorch(input_dim, hidden_dim, z_dim, num_branches=num_branches)
            for _ in range(num_experts)
        ])

        # Mise à jour de y à partir de [y, z]
        self.answer_dense1 = nn.Linear(y_dim + z_dim, hidden_dim)
        self.answer_dense2 = nn.Linear(hidden_dim, y_dim)

    def step_reasoning(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        inner_recursions: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Effectue inner_recursions étapes internes sur (y,z).
        """
        for _ in range(inner_recursions):
            weights = self.router(x, y, z)          # [B,E]

            z_candidates = []
            for expert in self.experts:
                z_next_e = expert(x, y, z)          # [B,dz]
                z_candidates.append(z_next_e)
            z_stack = torch.stack(z_candidates, dim=1)  # [B,E,dz]

            # Mélange pondéré
            weights_exp = weights.unsqueeze(-1)         # [B,E,1]
            z = (weights_exp * z_stack).sum(dim=1)      # [B,dz]

            yz = torch.cat([y, z], dim=-1)
            h = gelu(self.answer_dense1(yz))
            y = self.answer_dense2(h)

        return y, z


class TRLinkosTRMTorch(nn.Module):
    """
    Version PyTorch end-to-end du TRM (sans DAG pour l'entraînement).
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        z_dim: int,
        hidden_dim: int = 64,
        num_experts: int = 4,
        num_branches: int = 4,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        self.x_encoder = nn.Linear(x_dim, x_dim)
        self.core = TRLinkosCoreTorch(
            x_dim=x_dim,
            y_dim=y_dim,
            z_dim=z_dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            num_branches=num_branches,
        )

        # États init (paramètres appris)
        self.y_init = nn.Parameter(torch.zeros(1, y_dim))
        self.z_init = nn.Parameter(torch.zeros(1, z_dim))

    def forward(
        self,
        x: torch.Tensor,
        max_steps: int = 8,
        inner_recursions: int = 2,
    ) -> torch.Tensor:
        """
        x: [B, dx]
        Retour: y_final [B, dy]
        """
        B = x.size(0)
        x_enc = self.x_encoder(x)

        y = self.y_init.expand(B, -1)
        z = self.z_init.expand(B, -1)

        for _ in range(max_steps):
            y, z = self.core.step_reasoning(x_enc, y, z, inner_recursions=inner_recursions)

        return y
