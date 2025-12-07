"""MCP tools package for T-RLINKOS TRM++."""

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
    "reason_step",
    "run_trm_recursive",
    "torque_route",
    "dcaap_forward",
    "evaluate_score",
    "dag_add_node",
    "dag_best_path",
    "dag_get_state",
    "fractal_branch",
    "load_model",
    "save_model",
    "get_model_config",
    "get_repo_state",
    "write_repo_state",
    "execute_command",
    "get_system_info",
    "list_directory",
    "get_environment_variable",
    "check_command_exists",
]
