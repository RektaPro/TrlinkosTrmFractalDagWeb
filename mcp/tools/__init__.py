"""MCP tools package for T-RLINKOS TRM++.

This package provides MCP tool implementations organized by functionality:
- reasoning: Core T-RLINKOS reasoning operations
- dag: Fractal Merkle-DAG manipulation
- model: Model persistence and configuration
- repo: Repository file operations
- system: System command execution and information
"""

from .reasoning import (
    reason_step,
    run_trm_recursive,
    torque_route,
    dcaap_forward,
    evaluate_score,
)

from .dag import (
    dag_add_node,
    dag_best_path,
    dag_get_state,
    fractal_branch,
)

from .model import (
    load_model,
    save_model,
    get_model_config,
)

from .repo import (
    get_repo_state,
    write_repo_state,
)

from .system import (
    execute_command,
    get_system_info,
    list_directory,
    get_environment_variable,
    check_command_exists,
)

__all__ = [
    # Reasoning tools
    "reason_step",
    "run_trm_recursive",
    "torque_route",
    "dcaap_forward",
    "evaluate_score",
    # DAG tools
    "dag_add_node",
    "dag_best_path",
    "dag_get_state",
    "fractal_branch",
    # Model tools
    "load_model",
    "save_model",
    "get_model_config",
    # Repository tools
    "get_repo_state",
    "write_repo_state",
    # System tools
    "execute_command",
    "get_system_info",
    "list_directory",
    "get_environment_variable",
    "check_command_exists",
]
__version__ = "1.0.0"
