"""
Tests for the mini-training framework.

Tests cover:
- config.py: TrainingConfig dataclass
- encoders.py: TextEncoder and ImageEncoder
- datasets.py: XORDataset, ToyTextDataset, EncodedDataset
- training.py: Trainer class
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TrainingConfig
from encoders import TextEncoder, ImageEncoder
from datasets import XORDataset, ToyTextDataset, EncodedDataset, create_xor_dataloaders
from training import Trainer


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """Should have sensible default values."""
        config = TrainingConfig()
        assert config.lr == 1e-3
        assert config.batch_size == 64
        assert config.num_epochs == 50
        assert config.device == "cpu"
        assert config.seed == 42

    def test_custom_values(self):
        """Should accept custom values."""
        config = TrainingConfig(
            lr=0.01,
            batch_size=32,
            num_epochs=100,
            device="cuda",
            seed=123,
        )
        assert config.lr == 0.01
        assert config.batch_size == 32
        assert config.num_epochs == 100
        assert config.device == "cuda"
        assert config.seed == 123

    def test_to_dict(self):
        """Should convert to dictionary."""
        config = TrainingConfig()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "lr" in d
        assert "batch_size" in d
        assert d["lr"] == config.lr

    def test_from_dict(self):
        """Should create from dictionary."""
        d = {"lr": 0.001, "batch_size": 32, "num_epochs": 10}
        config = TrainingConfig.from_dict(d)
        assert config.lr == 0.001
        assert config.batch_size == 32
        assert config.num_epochs == 10

    def test_invalid_lr_raises_error(self):
        """Should raise error for invalid lr."""
        with pytest.raises(ValueError):
            TrainingConfig(lr=-0.001)

    def test_invalid_batch_size_raises_error(self):
        """Should raise error for invalid batch_size."""
        with pytest.raises(ValueError):
            TrainingConfig(batch_size=0)

    def test_invalid_num_epochs_raises_error(self):
        """Should raise error for invalid num_epochs."""
        with pytest.raises(ValueError):
            TrainingConfig(num_epochs=-1)


class TestTextEncoder:
    """Tests for TextEncoder class."""

    def test_initialization(self):
        """Should initialize with correct parameters."""
        encoder = TextEncoder(vocab_size=256, embed_dim=32, output_dim=16)
        assert encoder.vocab_size == 256
        assert encoder.embed_dim == 32
        assert encoder.output_dim == 16

    def test_char_mode_encoding(self):
        """Should encode texts in char mode."""
        encoder = TextEncoder(vocab_size=256, embed_dim=32, output_dim=16, mode="char")
        texts = ["Hello", "World"]
        embeddings = encoder.encode(texts)
        assert embeddings.shape == (2, 16)

    def test_word_mode_encoding(self):
        """Should encode texts in word mode."""
        encoder = TextEncoder(vocab_size=256, embed_dim=32, output_dim=16, mode="word")
        texts = ["Hello world", "Test text"]
        embeddings = encoder.encode(texts)
        assert embeddings.shape == (2, 16)

    def test_batch_encoding(self):
        """Should handle batch encoding."""
        encoder = TextEncoder(vocab_size=256, embed_dim=64, output_dim=32, mode="char")
        texts = ["a", "bb", "ccc", "dddd", "eeeee"]
        embeddings = encoder.encode(texts)
        assert embeddings.shape == (5, 32)

    def test_forward_alias(self):
        """forward() should be an alias for encode()."""
        encoder = TextEncoder(vocab_size=256, embed_dim=32, output_dim=16)
        texts = ["Test"]
        emb1 = encoder.encode(texts)
        emb2 = encoder.forward(texts)
        torch.testing.assert_close(emb1, emb2)


class TestImageEncoder:
    """Tests for ImageEncoder class."""

    def test_initialization(self):
        """Should initialize with correct parameters."""
        encoder = ImageEncoder(input_channels=3, output_dim=16, base_channels=8)
        assert encoder.input_channels == 3
        assert encoder.output_dim == 16

    def test_rgb_encoding(self):
        """Should encode RGB images."""
        encoder = ImageEncoder(input_channels=3, output_dim=16, base_channels=8)
        images = torch.randn(4, 3, 32, 32)
        embeddings = encoder.encode(images)
        assert embeddings.shape == (4, 16)

    def test_grayscale_encoding(self):
        """Should encode grayscale images."""
        encoder = ImageEncoder(input_channels=1, output_dim=16, base_channels=8)
        images = torch.randn(2, 1, 28, 28)
        embeddings = encoder.encode(images)
        assert embeddings.shape == (2, 16)

    def test_forward_alias(self):
        """forward() should be an alias for encode()."""
        encoder = ImageEncoder(input_channels=3, output_dim=16, base_channels=8)
        images = torch.randn(2, 3, 16, 16)
        emb1 = encoder.encode(images)
        emb2 = encoder.forward(images)
        torch.testing.assert_close(emb1, emb2)


class TestXORDataset:
    """Tests for XORDataset class."""

    def test_basic_creation(self):
        """Should create dataset with correct size."""
        dataset = XORDataset(n_samples=100, seed=42)
        assert len(dataset) == 100
        assert dataset.x_dim == 2
        assert dataset.y_dim == 1

    def test_xor_labels(self):
        """Labels should follow XOR logic."""
        dataset = XORDataset(n_samples=1000, noise_std=0.0, seed=42)
        correct = 0
        for x, y in dataset:
            expected = float(x[0].item() != x[1].item())
            if abs(y.item() - expected) < 0.01:
                correct += 1
        assert correct == 1000  # All labels should be correct XOR

    def test_extended_dim(self):
        """Should handle extended dimensions."""
        dataset = XORDataset(n_samples=50, extended_dim=3, seed=42)
        assert dataset.x_dim == 5  # 2 + 3

    def test_noise_addition(self):
        """Should add noise when specified."""
        dataset1 = XORDataset(n_samples=100, noise_std=0.0, seed=42)
        dataset2 = XORDataset(n_samples=100, noise_std=0.5, seed=42)
        # With noise, X values won't be exactly 0 or 1
        x_noiseless, _ = dataset1[0]
        x_noisy, _ = dataset2[0]
        # Noiseless should be 0 or 1
        assert all(abs(v - 0.0) < 0.01 or abs(v - 1.0) < 0.01 for v in x_noiseless[:2])

    def test_to_device(self):
        """Should move to device."""
        dataset = XORDataset(n_samples=50, seed=42)
        dataset = dataset.to("cpu")
        x, y = dataset[0]
        assert x.device.type == "cpu"


class TestToyTextDataset:
    """Tests for ToyTextDataset class."""

    def test_basic_creation(self):
        """Should create dataset with correct size."""
        dataset = ToyTextDataset(n_samples_per_class=50, seed=42)
        assert len(dataset) == 100  # 50 per class * 2 classes
        assert dataset.num_classes == 2

    def test_returns_text_and_label(self):
        """Should return text and label."""
        dataset = ToyTextDataset(n_samples_per_class=10, seed=42)
        text, label = dataset[0]
        assert isinstance(text, str)
        assert isinstance(label, int)
        assert label in [0, 1]

    def test_get_texts_and_labels(self):
        """Should return all texts and labels."""
        dataset = ToyTextDataset(n_samples_per_class=20, seed=42)
        texts, labels = dataset.get_texts_and_labels()
        assert len(texts) == 40
        assert labels.shape == (40,)


class TestEncodedDataset:
    """Tests for EncodedDataset class."""

    def test_basic_creation(self):
        """Should create from tensors."""
        X = torch.randn(100, 16)
        y = torch.randn(100, 4)
        dataset = EncodedDataset(X, y)
        assert len(dataset) == 100
        assert dataset.x_dim == 16
        assert dataset.y_dim == 4

    def test_1d_labels(self):
        """Should handle 1D label tensor."""
        X = torch.randn(50, 8)
        y = torch.randn(50)  # 1D
        dataset = EncodedDataset(X, y)
        assert dataset.y_dim == 1

    def test_size_mismatch_raises_error(self):
        """Should raise error for size mismatch."""
        X = torch.randn(100, 16)
        y = torch.randn(50, 4)  # Different size
        with pytest.raises(ValueError):
            EncodedDataset(X, y)


class TestCreateXORDataloaders:
    """Tests for create_xor_dataloaders function."""

    def test_returns_two_loaders(self):
        """Should return train and val loaders."""
        train_loader, val_loader = create_xor_dataloaders(
            n_train=100, n_val=20, batch_size=10, seed=42
        )
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

    def test_batch_size(self):
        """Batches should have correct size."""
        train_loader, _ = create_xor_dataloaders(
            n_train=100, n_val=20, batch_size=10, seed=42
        )
        for x, y in train_loader:
            assert x.shape[0] == 10
            break


class TestTrainer:
    """Tests for Trainer class."""

    def test_initialization(self):
        """Should initialize correctly."""
        from trlinkos_trm_torch import TRLinkosTRMTorch

        model = TRLinkosTRMTorch(x_dim=2, y_dim=1, z_dim=4, hidden_dim=16, num_experts=2)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.BCEWithLogitsLoss()
        config = TrainingConfig(num_epochs=2, batch_size=16)

        train_loader, val_loader = create_xor_dataloaders(
            n_train=64, n_val=16, batch_size=16, seed=42
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=config,
            dataloader_train=train_loader,
            dataloader_val=val_loader,
        )

        assert trainer.model is not None
        assert trainer.device == "cpu"

    def test_short_training(self):
        """Should complete a short training run."""
        from trlinkos_trm_torch import TRLinkosTRMTorch

        model = TRLinkosTRMTorch(
            x_dim=2, y_dim=1, z_dim=4, hidden_dim=16, num_experts=2, num_branches=2
        )
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.BCEWithLogitsLoss()
        config = TrainingConfig(
            num_epochs=2, batch_size=32, max_steps=2, inner_recursions=1, log_interval=10
        )

        train_loader, val_loader = create_xor_dataloaders(
            n_train=64, n_val=16, batch_size=32, seed=42
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=config,
            dataloader_train=train_loader,
            dataloader_val=val_loader,
        )

        history = trainer.train()

        assert len(history["train_loss"]) == 2
        assert len(history["train_acc"]) == 2
        assert len(history["val_loss"]) == 2

    def test_evaluate(self):
        """Should evaluate on a dataloader."""
        from trlinkos_trm_torch import TRLinkosTRMTorch

        model = TRLinkosTRMTorch(
            x_dim=2, y_dim=1, z_dim=4, hidden_dim=16, num_experts=2, num_branches=2
        )
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.BCEWithLogitsLoss()
        config = TrainingConfig(num_epochs=1, batch_size=16, max_steps=2, inner_recursions=1)

        train_loader, val_loader = create_xor_dataloaders(
            n_train=32, n_val=16, batch_size=16, seed=42
        )

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            config=config,
            dataloader_train=train_loader,
        )

        loss, acc = trainer.evaluate(val_loader)
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
