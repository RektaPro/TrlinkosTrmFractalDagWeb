"""
Pytest configuration for T-RLINKOS TRM++ tests.
"""

import pytest


def pytest_collection_modifyitems(config, items):
    """
    Automatically skip tests that require optional dependencies.
    """
    skip_torch = pytest.mark.skip(reason="torch not available")
    skip_transformers = pytest.mark.skip(reason="transformers not available")

    # Check for optional dependencies
    try:
        import torch  # noqa: F401

        torch_available = True
    except ImportError:
        torch_available = False

    try:
        import transformers  # noqa: F401

        transformers_available = True
    except ImportError:
        transformers_available = False

    for item in items:
        # Skip tests that require torch
        if not torch_available and "test_training_framework" in item.nodeid:
            item.add_marker(skip_torch)

        # Skip tests that require transformers
        if not transformers_available and any(
            keyword in item.nodeid for keyword in ["huggingface", "transformers"]
        ):
            item.add_marker(skip_transformers)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "torch: marks tests as requiring torch (deselect with '-m \"not torch\"')"
    )
    config.addinivalue_line("markers", "transformers: marks tests as requiring transformers")
