"""DAG tools for MCP - Fractal Merkle-DAG operations."""

from typing import Any, Dict, List, Optional
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from t_rlinkos_trm_fractal_dag import FractalMerkleDAG


# Global DAG storage
_dags: Dict[str, FractalMerkleDAG] = {}
_dag_counter = 0


def _create_dag(store_states: bool = True) -> str:
    """Create a new DAG and return its ID."""
    global _dag_counter
    dag_id = f"dag_{_dag_counter}"
    _dag_counter += 1
    _dags[dag_id] = FractalMerkleDAG(store_states=store_states)
    return dag_id


def _get_dag(dag_id: str) -> Optional[FractalMerkleDAG]:
    """Get a DAG by ID."""
    return _dags.get(dag_id)


def dag_add_node(
    dag_id: str,
    step: int,
    y: List[float],
    z: List[float],
    parents: Optional[List[str]] = None,
    score: Optional[float] = None,
) -> Dict[str, Any]:
    """Add a reasoning step node to the Fractal Merkle-DAG.

    Args:
        dag_id: DAG identifier
        step: Step number
        y: Response state
        z: Internal state
        parents: Parent node IDs
        score: Optional score for the node

    Returns:
        Dict with node_id
    """
    dag = _get_dag(dag_id)
    if dag is None:
        # Create new DAG
        dag_id = _create_dag()
        dag = _dags[dag_id]

    y_np = np.array(y, dtype=np.float64).reshape(1, -1)
    z_np = np.array(z, dtype=np.float64).reshape(1, -1)

    node_id = dag.add_step(
        step=step,
        y=y_np,
        z=z_np,
        parents=parents or [],
        score=score,
    )

    return {
        "dag_id": dag_id,
        "node_id": node_id,
        "step": step,
        "score": score,
    }


def dag_best_path(dag_id: str) -> Dict[str, Any]:
    """Get the best reasoning path from root to the highest-scoring node.

    Args:
        dag_id: DAG identifier

    Returns:
        Dict with path nodes and best score
    """
    dag = _get_dag(dag_id)
    if dag is None:
        return {"error": f"DAG not found: {dag_id}"}

    best_node = dag.get_best_node()
    if best_node is None:
        return {"error": "No nodes in DAG", "dag_id": dag_id}

    # Get path from root to best node
    path = dag.get_fractal_path(best_node.node_id)

    return {
        "dag_id": dag_id,
        "best_node_id": best_node.node_id,
        "best_score": best_node.score,
        "path_length": len(path),
        "path": [
            {
                "node_id": node.node_id,
                "step": node.step,
                "depth": node.depth,
                "score": node.score,
            }
            for node in path
        ],
    }


def dag_get_state(dag_id: str) -> Dict[str, Any]:
    """Get the current state of a Fractal Merkle-DAG.

    Args:
        dag_id: DAG identifier

    Returns:
        Dict with DAG statistics
    """
    dag = _get_dag(dag_id)
    if dag is None:
        return {"error": f"DAG not found: {dag_id}"}

    best_node = dag.get_best_node()
    depth_stats = dag.get_depth_statistics()

    return {
        "dag_id": dag_id,
        "num_nodes": len(dag.nodes),
        "max_depth": dag.max_depth,
        "store_states": dag.store_states,
        "depth_statistics": depth_stats,
        "root_nodes": dag.root_nodes,
        "best_node": {
            "node_id": best_node.node_id,
            "step": best_node.step,
            "depth": best_node.depth,
            "score": best_node.score,
        } if best_node else None,
    }


def fractal_branch(
    dag_id: str,
    parent_node_id: str,
    y: List[float],
    z: List[float],
    score: Optional[float] = None,
) -> Dict[str, Any]:
    """Create a fractal branch in the reasoning DAG.

    Args:
        dag_id: DAG identifier
        parent_node_id: Parent node to branch from
        y: Initial y state for branch
        z: Initial z state for branch
        score: Optional score

    Returns:
        Dict with branch_node_id
    """
    dag = _get_dag(dag_id)
    if dag is None:
        return {"error": f"DAG not found: {dag_id}"}

    y_np = np.array(y, dtype=np.float64).reshape(1, -1)
    z_np = np.array(z, dtype=np.float64).reshape(1, -1)

    branch_id = dag.create_branch(
        parent_node_id=parent_node_id,
        y=y_np,
        z=z_np,
        score=score,
    )

    if branch_id is None:
        return {
            "error": "Could not create branch (max depth reached or invalid parent)",
            "dag_id": dag_id,
        }

    return {
        "dag_id": dag_id,
        "branch_node_id": branch_id,
        "parent_node_id": parent_node_id,
        "score": score,
    }
