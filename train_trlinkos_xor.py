# train_trlinkos_xor.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast

from trlinkos_trm_torch import TRLinkosTRMTorch


def make_xor_dataset(n_samples: int = 1024, device: str = "cpu"):
    # XOR sur {0,1}^2 avec bruit léger optionnel
    X = torch.randint(0, 2, (n_samples, 2)).float()
    y = ((X[:, 0] != X[:, 1]).float()).unsqueeze(-1)  # [N,1]
    return X.to(device), y.to(device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # TRM: entrée 2D, sortie 1D, z_dim=8
    model = TRLinkosTRMTorch(
        x_dim=2,
        y_dim=1,
        z_dim=8,
        hidden_dim=32,
        num_experts=4,
        num_branches=4,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler()  # For mixed precision training

    X, y = make_xor_dataset(2048, device=device)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    n_epochs = 50

    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in dataloader:
            optimizer.zero_grad()

            with autocast():  # Mixed precision context
                logits = model(xb, max_steps=6, inner_recursions=2)  # [B,1]
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        avg_loss = total_loss / total
        acc = correct / total
        print(f"Epoch {epoch:03d} | Loss={avg_loss:.4f} | Acc={acc:.4f}")

    # Test rapide
    X_test = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]).to(device)
    with torch.no_grad():
        logits = model(X_test, max_steps=6, inner_recursions=2)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
    print("X test:\n", X_test.cpu().numpy())
    print("Probs:\n", probs.cpu().numpy())
    print("Preds:\n", preds.cpu().numpy())


if __name__ == "__main__":
    main()
