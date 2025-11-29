"""Model tools for MCP - Model persistence operations."""

from typing import Any, Dict, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from t_rlinkos_trm_fractal_dag import (
    TRLinkosTRM,
    save_model as _save_model,
    load_model as _load_model,
)


# Global model instance
_model: Optional[TRLinkosTRM] = None
_config: Dict[str, int] = {
    "x_dim": 64,
    "y_dim": 32,
    "z_dim": 64,
    "hidden_dim": 256,
    "num_experts": 4,
}


def _get_model() -> TRLinkosTRM:
    """Get or create the global model instance."""
    global _model
    if _model is None:
        _model = TRLinkosTRM(
            x_dim=_config["x_dim"],
            y_dim=_config["y_dim"],
            z_dim=_config["z_dim"],
            hidden_dim=_config["hidden_dim"],
            num_experts=_config["num_experts"],
        )
    return _model


def _set_model(model: TRLinkosTRM) -> None:
    """Set the global model instance."""
    global _model, _config
    _model = model
    _config = {
        "x_dim": model.x_dim,
        "y_dim": model.y_dim,
        "z_dim": model.z_dim,
        "hidden_dim": model.core.answer_dense1.W.shape[0],
        "num_experts": model.core.num_experts,
    }


def load_model(filepath: str) -> Dict[str, Any]:
    """Load a saved TRLinkosTRM model from disk.

    Args:
        filepath: Path to the saved model file (.npz)

    Returns:
        Dict with status and config
    """
    try:
        model = _load_model(filepath)
        _set_model(model)
        return {
            "status": "success",
            "filepath": filepath,
            "config": _config.copy(),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def save_model(filepath: str) -> Dict[str, Any]:
    """Save the current TRLinkosTRM model to disk.

    Args:
        filepath: Path to save the model (.npz)

    Returns:
        Dict with status
    """
    try:
        model = _get_model()
        _save_model(model, filepath)
        return {
            "status": "success",
            "filepath": filepath,
            "config": _config.copy(),
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_model_config() -> Dict[str, Any]:
    """Get the current model configuration.

    Returns:
        Dict with model configuration
    """
    _ = _get_model()  # Ensure model is initialized
    return {
        "config": _config.copy(),
        "initialized": _model is not None,
    }
