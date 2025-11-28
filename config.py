# config.py
"""
Configuration pour l'entraînement de TRLinkosTRM.

Contient la dataclass TrainingConfig avec tous les hyperparamètres
nécessaires pour l'entraînement du modèle.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration de l'entraînement pour TRLinkosTRM.

    Attributes:
        lr: Taux d'apprentissage (learning rate).
        batch_size: Taille des batches pour l'entraînement.
        num_epochs: Nombre d'époques d'entraînement.
        device: Device d'entraînement ('cpu' ou 'cuda').
        seed: Graine aléatoire pour la reproductibilité.
        max_steps: Nombre d'étapes de raisonnement récursif.
        inner_recursions: Nombre de récursions internes par étape.
        log_interval: Intervalle (en époques) pour l'affichage des logs.
        use_amp: Activer le mixed precision training (AMP).
        gradient_clip: Valeur maximale pour le gradient clipping (0 = désactivé).
        weight_decay: Coefficient de régularisation L2.
        warmup_epochs: Nombre d'époques de warmup pour le learning rate.
    """

    lr: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 50
    device: str = "cpu"
    seed: int = 42
    max_steps: int = 6
    inner_recursions: int = 2
    log_interval: int = 1
    use_amp: bool = False
    gradient_clip: float = 1.0
    weight_decay: float = 0.0
    warmup_epochs: int = 0

    def __post_init__(self) -> None:
        """Validation des paramètres après initialisation."""
        if self.lr <= 0:
            raise ValueError(f"lr doit être positif, reçu: {self.lr}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size doit être positif, reçu: {self.batch_size}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs doit être positif, reçu: {self.num_epochs}")
        if self.max_steps <= 0:
            raise ValueError(f"max_steps doit être positif, reçu: {self.max_steps}")
        if self.inner_recursions <= 0:
            raise ValueError(f"inner_recursions doit être positif, reçu: {self.inner_recursions}")
        if self.device not in ("cpu", "cuda"):
            # Accept any device string but warn for unusual values
            if not self.device.startswith("cuda:"):
                import warnings
                warnings.warn(f"Unusual device: {self.device}")

    def to_dict(self) -> dict:
        """Convertit la configuration en dictionnaire."""
        return {
            "lr": self.lr,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "device": self.device,
            "seed": self.seed,
            "max_steps": self.max_steps,
            "inner_recursions": self.inner_recursions,
            "log_interval": self.log_interval,
            "use_amp": self.use_amp,
            "gradient_clip": self.gradient_clip,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        """Crée une configuration à partir d'un dictionnaire."""
        return cls(**d)


if __name__ == "__main__":
    # Test de la configuration
    config = TrainingConfig()
    print("Configuration par défaut:")
    print(config)
    print("\nEn dictionnaire:")
    print(config.to_dict())

    # Test avec paramètres personnalisés
    config_custom = TrainingConfig(
        lr=0.001,
        batch_size=32,
        num_epochs=100,
        device="cuda",
        seed=123,
    )
    print("\nConfiguration personnalisée:")
    print(config_custom)

    print("\n✅ config.py fonctionne correctement!")
