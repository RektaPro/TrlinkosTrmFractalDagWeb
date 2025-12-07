# training.py
"""
Pipeline d'entraînement pour TRLinkosTRM.

Contient:
- Trainer: Classe d'entraînement avec boucle d'entraînement complète
- train_trlinkos_on_toy_dataset: Fonction exemple pour entraîner sur XOR
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable, Any

from config import TrainingConfig
from datasets import XORDataset, create_xor_dataloaders


class Trainer:
    """Pipeline d'entraînement pour TRLinkosTRM.

    Implémente une boucle d'entraînement complète avec:
    - Support pour l'optimisation Adam/SGD
    - Mixed precision training (AMP) optionnel
    - Gradient clipping
    - Logging de la loss et des métriques
    - Validation optionnelle
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        config: TrainingConfig,
        dataloader_train: DataLoader,
        dataloader_val: Optional[DataLoader] = None,
    ):
        """Initialise le trainer.

        Args:
            model: Modèle TRLinkosTRM (ou autre modèle compatible).
            optimizer: Optimiseur PyTorch (Adam, SGD, etc.).
            loss_fn: Fonction de loss (BCEWithLogitsLoss, MSELoss, etc.).
            config: Configuration d'entraînement.
            dataloader_train: DataLoader pour l'entraînement.
            dataloader_val: DataLoader pour la validation (optionnel).
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val

        # Déplacer le modèle vers le device
        self.device = config.device
        self.model = self.model.to(self.device)

        # Scaler pour AMP (mixed precision)
        self.scaler = torch.amp.GradScaler(device=self.device) if config.use_amp else None

        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "epoch": [],
        }

    def _get_device_type(self) -> str:
        """Extract device type for AMP autocast.

        Returns:
            Device type string ('cpu' or 'cuda').
        """
        if ":" in self.device:
            return self.device.split(":")[0]
        return self.device

    def _get_lr(self, epoch: int) -> float:
        """Calcule le learning rate avec warmup optionnel.

        Args:
            epoch: Époque actuelle.

        Returns:
            Learning rate ajusté.
        """
        if self.config.warmup_epochs > 0 and epoch < self.config.warmup_epochs:
            # Warmup linéaire
            return self.config.lr * (epoch + 1) / self.config.warmup_epochs
        return self.config.lr

    def _update_lr(self, epoch: int) -> None:
        """Met à jour le learning rate de l'optimiseur."""
        new_lr = self._get_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    def _train_epoch(self) -> tuple:
        """Effectue une époque d'entraînement.

        Returns:
            (loss_moyenne, accuracy_moyenne)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in self.dataloader_train:
            # Déplacer vers le device
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            if self.config.use_amp and self.scaler is not None:
                with torch.amp.autocast(device_type=self._get_device_type()):
                    logits = self.model(
                        x_batch,
                        max_steps=self.config.max_steps,
                        inner_recursions=self.config.inner_recursions,
                    )
                    loss = self.loss_fn(logits, y_batch)

                # Backward pass avec scaler
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Forward pass normal
                logits = self.model(
                    x_batch,
                    max_steps=self.config.max_steps,
                    inner_recursions=self.config.inner_recursions,
                )
                loss = self.loss_fn(logits, y_batch)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip
                    )

                self.optimizer.step()

            # Accumuler les métriques
            total_loss += loss.item() * x_batch.size(0)

            # Calculer l'accuracy (pour classification binaire)
            if y_batch.shape[-1] == 1:
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += x_batch.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    @torch.no_grad()
    def _validate(self) -> tuple:
        """Effectue une validation.

        Returns:
            (loss_moyenne, accuracy_moyenne)
        """
        if self.dataloader_val is None:
            return 0.0, 0.0

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in self.dataloader_val:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            logits = self.model(
                x_batch,
                max_steps=self.config.max_steps,
                inner_recursions=self.config.inner_recursions,
            )
            loss = self.loss_fn(logits, y_batch)

            total_loss += loss.item() * x_batch.size(0)

            if y_batch.shape[-1] == 1:
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += x_batch.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy

    def train(self) -> Dict[str, List[float]]:
        """Boucle d'entraînement principale.

        Returns:
            Historique d'entraînement avec loss et accuracy par époque.
        """
        print(f"Démarrage de l'entraînement sur {self.device}")
        print(f"  - Époques: {self.config.num_epochs}")
        print(f"  - Batch size: {self.config.batch_size}")
        print(f"  - Learning rate: {self.config.lr}")
        print(f"  - Max steps: {self.config.max_steps}")
        print(f"  - Inner recursions: {self.config.inner_recursions}")
        print("-" * 50)

        for epoch in range(self.config.num_epochs):
            # Mettre à jour le learning rate (warmup)
            self._update_lr(epoch)

            # Entraînement
            train_loss, train_acc = self._train_epoch()

            # Validation
            val_loss, val_acc = self._validate()

            # Sauvegarder l'historique
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["epoch"].append(epoch)

            # Logging
            if epoch % self.config.log_interval == 0 or epoch == self.config.num_epochs - 1:
                log_msg = f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
                if self.dataloader_val is not None:
                    log_msg += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
                print(log_msg)

        print("-" * 50)
        print("Entraînement terminé!")

        return self.history

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> tuple:
        """Évalue le modèle sur un dataset.

        Args:
            dataloader: DataLoader à évaluer.

        Returns:
            (loss_moyenne, accuracy_moyenne)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            logits = self.model(
                x_batch,
                max_steps=self.config.max_steps,
                inner_recursions=self.config.inner_recursions,
            )
            loss = self.loss_fn(logits, y_batch)

            total_loss += loss.item() * x_batch.size(0)

            if y_batch.shape[-1] == 1:
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += x_batch.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return avg_loss, accuracy


def train_trlinkos_on_toy_dataset(
    num_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
) -> tuple:
    """Fonction exemple pour entraîner TRLinkosTRM sur le dataset XOR.

    Cette fonction montre comment utiliser le mini-framework d'entraînement
    pour entraîner TRLinkosTRM sur un dataset jouet (XOR).

    Args:
        num_epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        device: Training device ('cpu' or 'cuda').
        seed: Random seed for reproducibility.
        verbose: Show training logs.

    Returns:
        (model, history): Trained model and training history.

    Example:
        >>> model, history = train_trlinkos_on_toy_dataset(num_epochs=10)
        >>> print(f"Final accuracy: {history['train_acc'][-1]:.2%}")
    """
    # Import here to avoid circular imports
    from trlinkos_trm_torch import TRLinkosTRMTorch

    # Set random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create DataLoaders
    train_loader, val_loader = create_xor_dataloaders(
        n_train=2048,
        n_val=256,
        batch_size=batch_size,
        seed=seed,
    )

    # Créer le modèle
    model = TRLinkosTRMTorch(
        x_dim=2,  # XOR a 2 entrées
        y_dim=1,  # Sortie binaire
        z_dim=8,
        hidden_dim=32,
        num_experts=4,
        num_branches=4,
    )

    # Configuration
    config = TrainingConfig(
        lr=lr,
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device,
        seed=seed,
        max_steps=6,
        inner_recursions=2,
        log_interval=5 if verbose else num_epochs + 1,
    )

    # Créer l'optimiseur et la loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # Créer le trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
        dataloader_train=train_loader,
        dataloader_val=val_loader,
    )

    # Entraîner
    history = trainer.train()

    # Test final sur les 4 cas XOR
    model.eval()
    with torch.no_grad():
        X_test = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]).to(device)
        y_expected = torch.tensor([[0.], [1.], [1.], [0.]]).to(device)

        logits = model(X_test, max_steps=6, inner_recursions=2)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        if verbose:
            print("\n--- Test final sur les 4 cas XOR ---")
            print(f"Entrées:    {X_test.cpu().numpy().tolist()}")
            print(f"Attendu:    {y_expected.cpu().numpy().flatten().tolist()}")
            print(f"Prédiction: {preds.cpu().numpy().flatten().tolist()}")
            print(f"Probabilités: {probs.cpu().numpy().flatten().tolist()}")

            test_acc = (preds == y_expected).sum().item() / 4
            print(f"Accuracy test: {test_acc:.2%}")

    return model, history


def train_trlinkos_on_text_dataset(
    num_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
) -> tuple:
    """Entraîne TRLinkosTRM sur un dataset de classification de texte.

    Args:
        num_epochs: Nombre d'époques d'entraînement.
        batch_size: Taille des batches.
        lr: Taux d'apprentissage.
        device: Device d'entraînement ('cpu' ou 'cuda').
        seed: Graine aléatoire pour la reproductibilité.
        verbose: Afficher les logs d'entraînement.

    Returns:
        (model, history): Modèle entraîné et historique d'entraînement.
    """
    from trlinkos_trm_torch import TRLinkosTRMTorch
    from encoders import TextEncoder
    from datasets import ToyTextDataset, EncodedDataset

    # Fixer les graines
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Créer les datasets
    train_text_dataset = ToyTextDataset(n_samples_per_class=500, seed=seed)
    val_text_dataset = ToyTextDataset(n_samples_per_class=50, seed=seed + 1)

    # Créer l'encodeur de texte
    text_encoder = TextEncoder(
        vocab_size=256,
        embed_dim=64,
        output_dim=32,
        mode="word",
    ).to(device)

    # Encoder les textes
    train_texts, train_labels = train_text_dataset.get_texts_and_labels()
    val_texts, val_labels = val_text_dataset.get_texts_and_labels()

    with torch.no_grad():
        train_embeddings = text_encoder.encode(train_texts).to(device)
        val_embeddings = text_encoder.encode(val_texts).to(device)

    train_labels = train_labels.to(device).unsqueeze(-1)
    val_labels = val_labels.to(device).unsqueeze(-1)

    # Créer les datasets encodés
    train_dataset = EncodedDataset(train_embeddings, train_labels)
    val_dataset = EncodedDataset(val_embeddings, val_labels)

    # Créer les dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Créer le modèle
    model = TRLinkosTRMTorch(
        x_dim=32,
        y_dim=1,
        z_dim=16,
        hidden_dim=64,
        num_experts=4,
        num_branches=4,
    )

    # Configuration
    config = TrainingConfig(
        lr=lr,
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device,
        seed=seed,
        max_steps=8,
        inner_recursions=3,
        log_interval=5 if verbose else num_epochs + 1,
    )

    # Créer l'optimiseur et la loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # Créer le trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
        dataloader_train=train_loader,
        dataloader_val=val_loader,
    )

    # Entraîner
    history = trainer.train()

    if verbose:
        print("\n--- Test sur quelques exemples ---")
        test_texts = [
            "This is great news",
            "I feel terrible today",
            "Everything seems wonderful",
            "The result is awful",
        ]
        with torch.no_grad():
            test_embeddings = text_encoder.encode(test_texts).to(device)
            logits = model(test_embeddings, max_steps=8, inner_recursions=3)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

        for i, text in enumerate(test_texts):
            pred_label = "négatif" if preds[i].item() == 1 else "positif"
            print(f"  '{text}' -> {pred_label} (prob: {probs[i].item():.2f})")

    return model, history


def train_trlinkos_on_image_dataset(
    num_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
) -> tuple:
    """Entraîne TRLinkosTRM sur un dataset de classification d'images.

    Génère un dataset synthétique d'images avec 2 classes:
    - Classe 0: Images avec dominante claire (haute luminosité moyenne)
    - Classe 1: Images avec dominante sombre (basse luminosité moyenne)

    Args:
        num_epochs: Nombre d'époques d'entraînement.
        batch_size: Taille des batches.
        lr: Taux d'apprentissage.
        device: Device d'entraînement ('cpu' ou 'cuda').
        seed: Graine aléatoire pour la reproductibilité.
        verbose: Afficher les logs d'entraînement.

    Returns:
        (model, history): Modèle entraîné et historique d'entraînement.
    """
    from trlinkos_trm_torch import TRLinkosTRMTorch
    from encoders import ImageEncoder
    import numpy as np

    # Fixer les graines
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Générer un dataset d'images synthétiques
    def generate_synthetic_images(n_samples, image_size=28, seed_offset=0):
        np.random.seed(seed + seed_offset)
        images = []
        labels = []
        for i in range(n_samples):
            # Classe 0: images claires, Classe 1: images sombres
            label = i % 2
            if label == 0:
                # Image claire avec bruit
                img = np.random.rand(image_size, image_size, 3) * 0.5 + 0.5
            else:
                # Image sombre avec bruit
                img = np.random.rand(image_size, image_size, 3) * 0.5

            images.append(img)
            labels.append(label)

        # Convertir en tenseurs PyTorch [N, C, H, W]
        images = torch.tensor(np.array(images), dtype=torch.float32).permute(0, 3, 1, 2)
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)
        return images, labels

    # Générer train et val
    train_images, train_labels = generate_synthetic_images(1000, seed_offset=0)
    val_images, val_labels = generate_synthetic_images(200, seed_offset=1)

    # Créer l'encodeur d'images
    image_encoder = ImageEncoder(
        input_channels=3,
        output_dim=32,
        base_channels=16,
    ).to(device)

    # Encoder les images
    with torch.no_grad():
        train_embeddings = image_encoder.encode(train_images.to(device))
        val_embeddings = image_encoder.encode(val_images.to(device))

    train_labels = train_labels.to(device)
    val_labels = val_labels.to(device)

    # Créer les datasets
    from datasets import EncodedDataset
    train_dataset = EncodedDataset(train_embeddings, train_labels)
    val_dataset = EncodedDataset(val_embeddings, val_labels)

    # Créer les dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Créer le modèle
    model = TRLinkosTRMTorch(
        x_dim=32,
        y_dim=1,
        z_dim=16,
        hidden_dim=64,
        num_experts=4,
        num_branches=4,
    )

    # Configuration
    config = TrainingConfig(
        lr=lr,
        batch_size=batch_size,
        num_epochs=num_epochs,
        device=device,
        seed=seed,
        max_steps=8,
        inner_recursions=3,
        log_interval=5 if verbose else num_epochs + 1,
    )

    # Créer l'optimiseur et la loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # Créer le trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
        dataloader_train=train_loader,
        dataloader_val=val_loader,
    )

    # Entraîner
    history = trainer.train()

    if verbose:
        print("\n--- Classification d'images ---")
        print("  Classe 0: Images claires (haute luminosité)")
        print("  Classe 1: Images sombres (basse luminosité)")

    return model, history


if __name__ == "__main__":
    print("=" * 60)
    print("EXEMPLE D'ENTRAÎNEMENT DE TRLINKOS SUR XOR")
    print("=" * 60)

    # Déterminer le device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device utilisé: {device}")

    # Entraîner le modèle
    model, history = train_trlinkos_on_toy_dataset(
        num_epochs=30,
        batch_size=64,
        lr=1e-3,
        device=device,
        seed=42,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("RÉSUMÉ DE L'ENTRAÎNEMENT")
    print("=" * 60)
    print(f"Loss finale (train): {history['train_loss'][-1]:.4f}")
    print(f"Accuracy finale (train): {history['train_acc'][-1]:.2%}")
    if history['val_loss'][-1] > 0:
        print(f"Loss finale (val): {history['val_loss'][-1]:.4f}")
        print(f"Accuracy finale (val): {history['val_acc'][-1]:.2%}")

    print("\n✅ training.py fonctionne correctement!")
