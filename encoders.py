# encoders.py
"""
Encodeurs pour TRLinkosTRM.

Contient:
- TextEncoder: Encodeur de texte minimal (embedding bag + projection linéaire)
- ImageEncoder: Encodeur d'images minimaliste (optionnel, petit CNN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class TextEncoder(nn.Module):
    """Encodeur de texte minimal pour TRLinkosTRM.

    Architecture simple:
    - Embedding bag (moyenne des embeddings de caractères/mots)
    - Projection linéaire vers la dimension de sortie

    Cette implémentation est volontairement simple pour servir d'exemple.
    Pour des applications réelles, utiliser des modèles pré-entraînés (BERT, etc.)
    """

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 64,
        output_dim: int = 32,
        mode: str = "char",
        padding_idx: int = 0,
    ):
        """Initialise l'encodeur de texte.

        Args:
            vocab_size: Taille du vocabulaire (256 pour ASCII/caractères).
            embed_dim: Dimension des embeddings internes.
            output_dim: Dimension de sortie (doit correspondre à x_dim du modèle).
            mode: 'char' pour encodage par caractère, 'word' pour encodage par mot.
            padding_idx: Index utilisé pour le padding.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.mode = mode
        self.padding_idx = padding_idx

        # Embedding table
        self.embedding = nn.EmbeddingBag(
            vocab_size, embed_dim, mode="mean", padding_idx=padding_idx
        )

        # Projection vers output_dim avec une couche cachée (MLP simple)
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, output_dim),
        )

        # Dictionnaire pour le mode word (mapping mot -> index)
        self.word_to_idx: dict = {}
        self.next_word_idx: int = 1  # 0 est réservé au padding

    def _tokenize_char(self, text: str) -> List[int]:
        """Tokenise le texte par caractère (ASCII)."""
        return [min(ord(c), self.vocab_size - 1) for c in text]

    def _tokenize_word(self, text: str) -> List[int]:
        """Tokenise le texte par mot."""
        words = text.lower().split()
        tokens = []
        for word in words:
            if word not in self.word_to_idx:
                if self.next_word_idx < self.vocab_size:
                    self.word_to_idx[word] = self.next_word_idx
                    self.next_word_idx += 1
                else:
                    # Fallback: hash du mot pour vocabulaire plein
                    self.word_to_idx[word] = hash(word) % self.vocab_size
            tokens.append(self.word_to_idx[word])
        return tokens if tokens else [self.padding_idx]

    def encode(
        self, texts: List[str], max_length: int = 128
    ) -> torch.Tensor:
        """Encode une liste de textes en tenseurs.

        Args:
            texts: Liste de textes à encoder.
            max_length: Longueur maximale de la séquence (tronquée si dépassée).

        Returns:
            Tenseur de forme [batch_size, output_dim].
        """
        batch_tokens = []
        offsets = [0]

        for text in texts:
            if self.mode == "char":
                tokens = self._tokenize_char(text)
            else:
                tokens = self._tokenize_word(text)

            # Tronquer si nécessaire
            if len(tokens) > max_length:
                tokens = tokens[:max_length]

            batch_tokens.extend(tokens)
            offsets.append(len(batch_tokens))

        # Créer les tenseurs
        tokens_tensor = torch.tensor(batch_tokens, dtype=torch.long)
        offsets_tensor = torch.tensor(offsets[:-1], dtype=torch.long)

        # Move to same device as embedding
        device = self.embedding.weight.device
        tokens_tensor = tokens_tensor.to(device)
        offsets_tensor = offsets_tensor.to(device)

        # Embedding bag + projection
        embedded = self.embedding(tokens_tensor, offsets_tensor)
        output = self.projection(embedded)

        return output

    def forward(self, texts: List[str], max_length: int = 128) -> torch.Tensor:
        """Forward pass (alias pour encode)."""
        return self.encode(texts, max_length)


class ImageEncoder(nn.Module):
    """Encodeur d'images minimaliste pour TRLinkosTRM.

    Architecture CNN simple:
    - 2 couches convolutives avec pooling
    - Aplatissement + projection linéaire

    Cette implémentation est volontairement simple pour servir d'exemple.
    Pour des applications réelles, utiliser des modèles pré-entraînés (ResNet, ViT, etc.)
    """

    def __init__(
        self,
        input_channels: int = 3,
        output_dim: int = 32,
        base_channels: int = 16,
    ):
        """Initialise l'encodeur d'images.

        Args:
            input_channels: Nombre de canaux d'entrée (3 pour RGB, 1 pour grayscale).
            output_dim: Dimension de sortie (doit correspondre à x_dim du modèle).
            base_channels: Nombre de canaux de base pour les convolutions.
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim

        # Architecture CNN simple
        self.conv_layers = nn.Sequential(
            # Conv1: input_channels -> base_channels
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.MaxPool2d(2),  # Réduit la taille par 2

            # Conv2: base_channels -> base_channels*2
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Taille fixe 4x4
        )

        # Projection vers output_dim
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 2 * 4 * 4, output_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode un batch d'images.

        Args:
            images: Tenseur de forme [batch_size, channels, height, width].
                    Les images doivent être normalisées entre 0 et 1.

        Returns:
            Tenseur de forme [batch_size, output_dim].
        """
        features = self.conv_layers(images)
        output = self.projection(features)
        return output

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Alias pour forward (pour cohérence avec TextEncoder)."""
        return self.forward(images)


if __name__ == "__main__":
    # Tests
    print("--- Test TextEncoder ---")
    text_encoder = TextEncoder(vocab_size=256, embed_dim=32, output_dim=16, mode="char")

    texts = ["Hello world!", "Bonjour le monde", "Test encoding"]
    embeddings = text_encoder.encode(texts)
    print(f"Input: {len(texts)} textes")
    print(f"Output shape: {embeddings.shape}")
    assert embeddings.shape == (3, 16), f"Shape incorrecte: {embeddings.shape}"

    # Test mode word
    word_encoder = TextEncoder(vocab_size=256, embed_dim=32, output_dim=16, mode="word")
    word_embeddings = word_encoder.encode(texts)
    print(f"Word mode output shape: {word_embeddings.shape}")

    print("✅ TextEncoder fonctionne!")

    print("\n--- Test ImageEncoder ---")
    image_encoder = ImageEncoder(input_channels=3, output_dim=16, base_channels=8)

    # Images factices RGB 32x32
    fake_images = torch.randn(4, 3, 32, 32)
    image_embeddings = image_encoder.encode(fake_images)
    print(f"Input: {fake_images.shape}")
    print(f"Output shape: {image_embeddings.shape}")
    assert image_embeddings.shape == (4, 16), f"Shape incorrecte: {image_embeddings.shape}"

    # Test grayscale
    gray_encoder = ImageEncoder(input_channels=1, output_dim=16, base_channels=8)
    gray_images = torch.randn(2, 1, 28, 28)
    gray_embeddings = gray_encoder.encode(gray_images)
    print(f"Grayscale output shape: {gray_embeddings.shape}")

    print("✅ ImageEncoder fonctionne!")

    print("\n✅ encoders.py fonctionne correctement!")
