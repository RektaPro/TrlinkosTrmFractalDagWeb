"""
Minimal FastAPI Web API for T-RLINKOS TRM++.

Provides a simple REST API for inference with the recursive reasoning model.
No authentication, no persistence - just a minimal demonstration.

Endpoints:
- GET /health: Health check → {"status": "ok", "trm_config": {...}}
- POST /reason: Run recursive reasoning on input features

Usage:
    # Start the server
    uvicorn api:app --reload --port 8000

    # Or run directly
    python api.py

Example curl commands:
    # Health check
    curl http://localhost:8000/health
    # Returns: {"status": "ok", "trm_config": {"x_dim": 64, ...}}

    # Run reasoning (64 features required by default)
    curl -X POST http://localhost:8000/reason \\
        -H "Content-Type: application/json" \\
        -d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                         0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                         0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                         0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                         0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                         0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                         0.1, 0.2, 0.3, 0.4]}'

    # Returns:
    # {
    #   "output": [0.123, 0.456, ...],
    #   "dag_stats": {"num_nodes": 10, "max_depth": 0, ...}
    # }

    # With optional parameters
    curl -X POST http://localhost:8000/reason \\
        -H "Content-Type: application/json" \\
        -d '{"features": [...], "max_steps": 5, "inner_recursions": 2, "backtrack": true}'
"""

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

import numpy as np

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError as e:
    raise ImportError(
        "FastAPI is required for the web API. "
        "Install with: pip install fastapi uvicorn"
    ) from e

from t_rlinkos_trm_fractal_dag import TRLinkosTRM

# ============================
#  Pydantic Models for API
# ============================


class ReasoningRequest(BaseModel):
    """Request model for the /reason endpoint."""

    features: List[float] = Field(
        ...,
        description="Input features vector",
        min_length=1,
    )
    max_steps: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum reasoning steps",
    )
    inner_recursions: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Inner recursions per step",
    )
    backtrack: bool = Field(
        default=False,
        description="Enable backtracking to best states",
    )


class ReasoningResponse(BaseModel):
    """Response model for the /reason endpoint."""

    output: List[float] = Field(
        ...,
        description="Model output vector",
    )
    dag_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="DAG statistics: num_nodes, max_depth, etc.",
    )


class HealthResponse(BaseModel):
    """Response model for the /health endpoint."""

    status: str = Field(
        default="ok",
        description="Health status",
    )
    trm_config: Dict[str, int] = Field(
        default_factory=dict,
        description="TRM model configuration",
    )


# ============================
#  Model Configuration
# ============================

# Default model dimensions (can be configured via environment variables)
DEFAULT_X_DIM = 64
DEFAULT_Y_DIM = 32
DEFAULT_Z_DIM = 64
DEFAULT_HIDDEN_DIM = 256
DEFAULT_NUM_EXPERTS = 4


# ============================
#  Application State
# ============================


class AppState:
    """Application state container for model and config."""

    def __init__(self) -> None:
        self.model: Optional[TRLinkosTRM] = None
        self.config: Dict[str, int] = {}

    def initialize(self) -> None:
        """Initialize the TRM model."""
        self.config = {
            "x_dim": DEFAULT_X_DIM,
            "y_dim": DEFAULT_Y_DIM,
            "z_dim": DEFAULT_Z_DIM,
            "hidden_dim": DEFAULT_HIDDEN_DIM,
            "num_experts": DEFAULT_NUM_EXPERTS,
        }
        self.model = TRLinkosTRM(
            x_dim=self.config["x_dim"],
            y_dim=self.config["y_dim"],
            z_dim=self.config["z_dim"],
            hidden_dim=self.config["hidden_dim"],
            num_experts=self.config["num_experts"],
        )
        print(f"T-RLINKOS TRM++ model initialized with config: {self.config}")


# Global state instance
app_state = AppState()


# ============================
#  Lifespan Context Manager
# ============================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown."""
    # Startup
    app_state.initialize()
    yield
    # Shutdown (cleanup if needed)


# ============================
#  FastAPI Application
# ============================

app = FastAPI(
    title="T-RLINKOS TRM++ API",
    description=(
        "REST API for the T-RLINKOS Tiny Recursive Model with Fractal Merkle-DAG. "
        "Provides recursive reasoning capabilities with dCaAP-inspired neurons, "
        "Torque Clustering routing, and cryptographic audit trails."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware for browser access
# Note: In production, restrict allow_origins to specific trusted domains
# instead of using "*" which allows requests from any origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure with specific origins for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns:
        HealthResponse: Status and configuration info.

    Example:
        curl http://localhost:8000/health
        # Returns: {"status": "ok", "trm_config": {...}}
    """
    return HealthResponse(
        status="ok" if app_state.model is not None else "error",
        trm_config=app_state.config,
    )


@app.post("/reason", response_model=ReasoningResponse)
async def reason(request: ReasoningRequest) -> ReasoningResponse:
    """
    Run recursive reasoning on input features.

    The model processes the input through multiple reasoning steps,
    using dCaAP-inspired neurons and Torque Clustering routing.
    Returns the output along with DAG statistics about the reasoning process.

    Args:
        request: ReasoningRequest with features and optional parameters.

    Returns:
        ReasoningResponse: Model output and DAG statistics.

    Example:
        curl -X POST http://localhost:8000/reason \\
            -H "Content-Type: application/json" \\
            -d '{"features": [0.1, 0.2, ..., 0.64], "max_steps": 10}'

        # Returns:
        # {
        #   "output": [0.123, 0.456, ...],
        #   "dag_stats": {
        #     "num_nodes": 10,
        #     "max_depth": 0,
        #     "depth_statistics": {"0": 10},
        #     "best_node_step": 9,
        #     "best_node_score": null
        #   }
        # }
    """
    if app_state.model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # Validate input dimension
    expected_dim = app_state.config["x_dim"]
    if len(request.features) != expected_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_dim} features, got {len(request.features)}",
        )

    # Convert to numpy array and add batch dimension
    x = np.array(request.features, dtype=np.float64).reshape(1, -1)

    # Run reasoning
    try:
        y_pred, dag = app_state.model.forward_recursive(
            x,
            max_steps=request.max_steps,
            inner_recursions=request.inner_recursions,
            backtrack=request.backtrack,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reasoning error: {str(e)}")

    # Get DAG statistics
    best_node = dag.get_best_node()
    depth_stats = dag.get_depth_statistics()

    dag_stats = {
        "num_nodes": len(dag.nodes),
        "max_depth": max(depth_stats.keys()) if depth_stats else 0,
        "depth_statistics": depth_stats,
        "best_node_step": best_node.step if best_node else None,
        "best_node_score": best_node.score if best_node else None,
    }

    return ReasoningResponse(
        output=y_pred[0].tolist(),
        dag_stats=dag_stats,
    )


# ============================
#  Main Entry Point
# ============================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
