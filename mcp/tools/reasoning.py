"""Reasoning tools for MCP - TRLinkosTRM operations."""

from typing import Any, Dict, List, Optional
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from t_rlinkos_trm_fractal_dag import (
    TRLinkosTRM,
    mse_loss,
    cosine_similarity_loss,
)


# Global model instance (initialized lazily)
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


def reason_step(
    x: List[float],
    y: Optional[List[float]] = None,
    z: Optional[List[float]] = None,
    inner_recursions: int = 3,
) -> Dict[str, Any]:
    """Execute a single reasoning step with TRLinkosTRM.

    Args:
        x: Input feature vector
        y: Current response state (optional, defaults to zeros)
        z: Current internal state (optional, defaults to zeros)
        inner_recursions: Number of inner recursions

    Returns:
        Dict with y_next, z_next states
    """
    model = _get_model()

    x_np = np.array(x, dtype=np.float64).reshape(1, -1)

    if y is None:
        y_np = np.zeros((1, _config["y_dim"]), dtype=np.float64)
    else:
        y_np = np.array(y, dtype=np.float64).reshape(1, -1)

    if z is None:
        z_np = np.zeros((1, _config["z_dim"]), dtype=np.float64)
    else:
        z_np = np.array(z, dtype=np.float64).reshape(1, -1)

    # Encode x
    x_enc = model.x_encoder(x_np)

    # Execute step
    y_next, z_next = model.core.step_reasoning(
        x_enc, y_np, z_np, inner_recursions=inner_recursions
    )

    return {
        "y_next": y_next[0].tolist(),
        "z_next": z_next[0].tolist(),
        "inner_recursions": inner_recursions,
    }


def run_trm_recursive(
    x: List[float],
    max_steps: int = 10,
    inner_recursions: int = 3,
    backtrack: bool = False,
    fractal_branching: bool = False,
) -> Dict[str, Any]:
    """Run complete recursive reasoning loop with TRLinkosTRM.

    Args:
        x: Input feature vector
        max_steps: Maximum number of reasoning steps
        inner_recursions: Number of inner recursions per step
        backtrack: Enable backtracking to best states
        fractal_branching: Enable fractal exploration

    Returns:
        Dict with y_pred, dag_stats
    """
    model = _get_model()

    x_np = np.array(x, dtype=np.float64).reshape(1, -1)

    if fractal_branching:
        y_pred, dag = model.forward_recursive_fractal(
            x_np,
            max_steps=max_steps,
            inner_recursions=inner_recursions,
            backtrack=backtrack,
        )
    else:
        y_pred, dag = model.forward_recursive(
            x_np,
            max_steps=max_steps,
            inner_recursions=inner_recursions,
            backtrack=backtrack,
        )

    # Get statistics
    best_node = dag.get_best_node()
    depth_stats = dag.get_depth_statistics()

    return {
        "y_pred": y_pred[0].tolist(),
        "dag_stats": {
            "num_nodes": len(dag.nodes),
            "max_depth": max(depth_stats.keys()) if depth_stats else 0,
            "depth_statistics": depth_stats,
            "best_node_step": best_node.step if best_node else None,
            "best_node_score": best_node.score if best_node else None,
        },
        "config": {
            "max_steps": max_steps,
            "inner_recursions": inner_recursions,
            "backtrack": backtrack,
            "fractal_branching": fractal_branching,
        },
    }


def torque_route(
    x: List[float],
    y: List[float],
    z: List[float],
) -> Dict[str, Any]:
    """Compute expert routing weights using Torque Clustering algorithm.

    Args:
        x: Input features
        y: Current response
        z: Internal state

    Returns:
        Dict with routing weights per expert
    """
    model = _get_model()

    x_np = np.array(x, dtype=np.float64).reshape(1, -1)
    y_np = np.array(y, dtype=np.float64).reshape(1, -1)
    z_np = np.array(z, dtype=np.float64).reshape(1, -1)

    weights = model.core.router.forward(x_np, y_np, z_np)

    return {
        "weights": weights[0].tolist(),
        "num_experts": _config["num_experts"],
        "top_expert": int(np.argmax(weights[0])),
    }


def dcaap_forward(
    x: List[float],
    y: List[float],
    z: List[float],
) -> Dict[str, Any]:
    """Execute forward pass through a dCaAP cell.

    Args:
        x: External input
        y: Current response
        z: Internal state

    Returns:
        Dict with z_next state
    """
    model = _get_model()

    x_np = np.array(x, dtype=np.float64).reshape(1, -1)
    y_np = np.array(y, dtype=np.float64).reshape(1, -1)
    z_np = np.array(z, dtype=np.float64).reshape(1, -1)

    # Use first expert
    z_next = model.core.experts[0].forward(x_np, y_np, z_np)

    return {
        "z_next": z_next[0].tolist(),
        "input_dim": x_np.shape[1] + y_np.shape[1] + z_np.shape[1],
    }


def evaluate_score(
    y_pred: List[float],
    y_target: List[float],
    metric: str = "mse",
) -> Dict[str, Any]:
    """Evaluate a prediction score.

    Args:
        y_pred: Predicted values
        y_target: Target values
        metric: Scoring metric (mse, cosine, mae)

    Returns:
        Dict with score and metric info
    """
    y_pred_np = np.array(y_pred, dtype=np.float64).reshape(1, -1)
    y_target_np = np.array(y_target, dtype=np.float64).reshape(1, -1)

    if metric == "mse":
        score = mse_loss(y_pred_np, y_target_np)
    elif metric == "cosine":
        score = cosine_similarity_loss(y_pred_np, y_target_np)
    elif metric == "mae":
        score = float(np.mean(np.abs(y_pred_np - y_target_np)))
    else:
        return {"error": f"Unknown metric: {metric}"}

    return {
        "score": score,
        "metric": metric,
        "lower_is_better": True,
    }
