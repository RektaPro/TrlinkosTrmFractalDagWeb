# datasets.py
"""
Datasets pour l'entraînement de TRLinkosTRM.

Contient:
- XORDataset: Dataset XOR étendu (classique pour tester les réseaux neuronaux)
- ToyTextDataset: Dataset texte jouet pour la classification simple
"""

import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List
import numpy as np


class XORDataset(Dataset):
    """Dataset XOR étendu pour tester TRLinkosTRM.

    Le problème XOR est un classique pour tester les réseaux neuronaux
    car il nécessite une représentation non-linéaire. C'est un bon test
    pour vérifier que TRLinkosTRM fonctionne correctement.

    Ce dataset génère des échantillons XOR avec possibilité d'ajouter
    du bruit et d'étendre les dimensions.
    """

    def __init__(
        self,
        n_samples: int = 1024,
        noise_std: float = 0.0,
        extended_dim: int = 0,
        seed: Optional[int] = None,
    ):
        """Initialise le dataset XOR.

        Args:
            n_samples: Nombre d'échantillons à générer.
            noise_std: Écart-type du bruit gaussien à ajouter (0 = pas de bruit).
            extended_dim: Dimensions supplémentaires (remplies de bruit).
            seed: Graine aléatoire pour la reproductibilité.
        """
        super().__init__()
        self.n_samples = n_samples
        self.noise_std = noise_std
        self.extended_dim = extended_dim

        # Fixer la graine si fournie
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Générer les données XOR de base
        self.X = torch.randint(0, 2, (n_samples, 2)).float()
        self.y = ((self.X[:, 0] != self.X[:, 1]).float()).unsqueeze(-1)  # [N, 1]

        # Ajouter du bruit aux entrées si demandé
        if noise_std > 0:
            self.X = self.X + torch.randn_like(self.X) * noise_std

        # Étendre les dimensions si demandé
        if extended_dim > 0:
            extra_dims = torch.randn(n_samples, extended_dim) * 0.1
            self.X = torch.cat([self.X, extra_dims], dim=-1)

    @property
    def x_dim(self) -> int:
        """Dimension des entrées."""
        return self.X.shape[1]

    @property
    def y_dim(self) -> int:
        """Dimension des sorties."""
        return self.y.shape[1]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    def to(self, device: str) -> "XORDataset":
        """Déplace les données vers un device (cpu/cuda)."""
        self.X = self.X.to(device)
        self.y = self.y.to(device)
        return self


class ToyTextDataset(Dataset):
    """Dataset texte jouet pour la classification simple.

    Génère des échantillons texte-label pour tester l'encodeur de texte
    avec TRLinkosTRM. Les textes sont des phrases simples avec des
    patterns reconnaissables.
    """

    def __init__(
        self,
        n_samples_per_class: int = 100,
        seed: Optional[int] = None,
    ):
        """Initialise le dataset texte jouet.

        Args:
            n_samples_per_class: Nombre d'échantillons par classe.
            seed: Graine aléatoire pour la reproductibilité.
        """
        super().__init__()

        if seed is not None:
            np.random.seed(seed)

        self.samples: List[Tuple[str, int]] = []

        # Patterns pour chaque classe
        class_patterns = {
            0: ["good", "great", "excellent", "wonderful", "amazing", "positive", "happy"],
            1: ["bad", "terrible", "awful", "horrible", "negative", "sad", "poor"],
        }

        # Templates de phrases
        templates = [
            "This is {} news",
            "I feel {} today",
            "The result is {}",
            "It was a {} experience",
            "Everything seems {}",
        ]

        # Générer les échantillons
        for class_idx, words in class_patterns.items():
            for _ in range(n_samples_per_class):
                word = np.random.choice(words)
                template = np.random.choice(templates)
                text = template.format(word)
                self.samples.append((text, class_idx))

        # Mélanger les échantillons
        np.random.shuffle(self.samples)

    @property
    def num_classes(self) -> int:
        """Nombre de classes."""
        return 2

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.samples[idx]

    def get_texts_and_labels(self) -> Tuple[List[str], torch.Tensor]:
        """Retourne toutes les données sous forme de listes.

        Utile pour encoder tous les textes en une seule fois.

        Returns:
            (texts, labels): Liste de textes et tenseur de labels.
        """
        texts = [s[0] for s in self.samples]
        labels = torch.tensor([s[1] for s in self.samples], dtype=torch.float32)
        return texts, labels


class EncodedDataset(Dataset):
    """Dataset wrapper pour des données pré-encodées.

    Permet de créer un dataset à partir de tenseurs X et y pré-calculés,
    par exemple après avoir encodé des textes avec TextEncoder.
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        """Initialise le dataset avec des tenseurs pré-encodés.

        Args:
            X: Tenseur des entrées [N, x_dim].
            y: Tenseur des cibles [N, y_dim] ou [N].
        """
        super().__init__()
        if len(X) != len(y):
            raise ValueError(f"X et y doivent avoir le même nombre d'échantillons: {len(X)} != {len(y)}")

        self.X = X
        self.y = y.unsqueeze(-1) if y.dim() == 1 else y

    @property
    def x_dim(self) -> int:
        """Dimension des entrées."""
        return self.X.shape[1]

    @property
    def y_dim(self) -> int:
        """Dimension des sorties."""
        return self.y.shape[1]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    def to(self, device: str) -> "EncodedDataset":
        """Déplace les données vers un device (cpu/cuda)."""
        self.X = self.X.to(device)
        self.y = self.y.to(device)
        return self


def create_xor_dataloaders(
    n_train: int = 2048,
    n_val: int = 256,
    batch_size: int = 64,
    noise_std: float = 0.0,
    extended_dim: int = 0,
    seed: int = 42,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Crée des DataLoaders pour l'entraînement et la validation XOR.

    Args:
        n_train: Nombre d'échantillons d'entraînement.
        n_val: Nombre d'échantillons de validation.
        batch_size: Taille des batches.
        noise_std: Écart-type du bruit gaussien.
        extended_dim: Dimensions supplémentaires.
        seed: Graine aléatoire.

    Returns:
        (train_loader, val_loader): DataLoaders d'entraînement et validation.
    """
    train_dataset = XORDataset(
        n_samples=n_train,
        noise_std=noise_std,
        extended_dim=extended_dim,
        seed=seed,
    )
    val_dataset = XORDataset(
        n_samples=n_val,
        noise_std=noise_std,
        extended_dim=extended_dim,
        seed=seed + 1,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Tests
    print("--- Test XORDataset ---")
    xor_dataset = XORDataset(n_samples=100, noise_std=0.1, extended_dim=2, seed=42)
    print(f"Taille du dataset: {len(xor_dataset)}")
    print(f"x_dim: {xor_dataset.x_dim}")
    print(f"y_dim: {xor_dataset.y_dim}")

    x, y = xor_dataset[0]
    print(f"Premier échantillon - x: {x.shape}, y: {y.shape}")

    # Vérifier les proportions XOR (environ 50/50)
    ones = sum(1 for _, y in xor_dataset if y.item() == 1)
    print(f"Proportion de 1: {ones/len(xor_dataset):.2%}")

    print("✅ XORDataset fonctionne!")

    print("\n--- Test ToyTextDataset ---")
    text_dataset = ToyTextDataset(n_samples_per_class=50, seed=42)
    print(f"Taille du dataset: {len(text_dataset)}")
    print(f"Nombre de classes: {text_dataset.num_classes}")

    text, label = text_dataset[0]
    print(f"Premier échantillon - texte: '{text}', label: {label}")

    texts, labels = text_dataset.get_texts_and_labels()
    print(f"Nombre de textes: {len(texts)}")
    print(f"Shape des labels: {labels.shape}")

    print("✅ ToyTextDataset fonctionne!")

    print("\n--- Test EncodedDataset ---")
    X = torch.randn(100, 16)
    y = torch.randint(0, 2, (100,)).float()
    encoded_dataset = EncodedDataset(X, y)
    print(f"Taille du dataset: {len(encoded_dataset)}")
    print(f"x_dim: {encoded_dataset.x_dim}")
    print(f"y_dim: {encoded_dataset.y_dim}")

    print("✅ EncodedDataset fonctionne!")

    print("\n--- Test DataLoaders ---")
    train_loader, val_loader = create_xor_dataloaders(
        n_train=256,
        n_val=64,
        batch_size=32,
        seed=42,
    )
    print(f"Batches d'entraînement: {len(train_loader)}")
    print(f"Batches de validation: {len(val_loader)}")

    for x_batch, y_batch in train_loader:
        print(f"Batch shape - x: {x_batch.shape}, y: {y_batch.shape}")
        break

    print("✅ DataLoaders fonctionnent!")

    print("\n✅ datasets.py fonctionne correctement!")
