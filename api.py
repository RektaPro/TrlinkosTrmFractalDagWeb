"""
FastAPI Web API for T-RLINKOS TRM++.

Provides a comprehensive REST API for inference with the recursive reasoning model,
integrating synergies from all project modules including:
- DAG visualization (dag_visualizer.py)
- Text/Image encoding (t_rlinkos_trm_fractal_dag.py)
- LLM reasoning integration (trlinkos_llm_layer.py)
- Performance benchmarking (t_rlinkos_trm_fractal_dag.py)

Endpoints:
- GET /health: Health check → {"status": "ok", "trm_config": {...}}
- POST /reason: Run recursive reasoning on input features
- POST /reason/batch: Batch reasoning on multiple inputs
- POST /reason/text: Reason over text input
- GET /dag/visualize: Get DAG visualization data
- GET /model/info: Get model statistics and info
- GET /benchmark: Run performance benchmark

Usage:
    # Start the server
    uvicorn api:app --reload --port 8000

    # Or run directly
    python api.py

Example curl commands:
    # Health check
    curl http://localhost:8000/health
    # Returns: {"status": "ok", "trm_config": {"x_dim": 64, ...}}

    # Run reasoning (requires 64 features by default - see DEFAULT_X_DIM)
    curl -X POST http://localhost:8000/reason \\
        -H "Content-Type: application/json" \\
        -d '{"features": [<64 float values>]}'

    # Batch reasoning
    curl -X POST http://localhost:8000/reason/batch \\
        -H "Content-Type: application/json" \\
        -d '{"batch": [[...], [...]], "max_steps": 5}'

    # Text reasoning
    curl -X POST http://localhost:8000/reason/text \\
        -H "Content-Type: application/json" \\
        -d '{"text": "What is machine learning?", "max_steps": 5}'

    # Get DAG visualization
    curl http://localhost:8000/dag/visualize?format=json

    # Run benchmark
    curl http://localhost:8000/benchmark
"""

from contextlib import asynccontextmanager
import os
import tempfile
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

import numpy as np

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, PlainTextResponse
    from pydantic import BaseModel, Field
except ImportError as e:
    raise ImportError(
        "FastAPI is required for the web API. "
        "Install with: pip install fastapi uvicorn"
    ) from e

# Core T-RLINKOS imports - these are required for the API to function
from t_rlinkos_trm_fractal_dag import (
    TRLinkosTRM,
    FractalMerkleDAG,
    TextEncoder,
    benchmark_forward_recursive,
)
from dag_visualizer import DAGVisualizer

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


class BatchReasoningRequest(BaseModel):
    """Request model for the /reason/batch endpoint."""

    batch: List[List[float]] = Field(
        ...,
        description="Batch of input feature vectors",
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


class BatchReasoningResponse(BaseModel):
    """Response model for the /reason/batch endpoint."""

    outputs: List[List[float]] = Field(
        ...,
        description="Batch of model output vectors",
    )
    dag_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="DAG statistics for the batch",
    )
    batch_size: int = Field(
        ...,
        description="Number of samples in the batch",
    )


class TextReasoningRequest(BaseModel):
    """Request model for the /reason/text endpoint."""

    text: str = Field(
        ...,
        description="Input text for reasoning",
        min_length=1,
        max_length=10000,
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


class TextReasoningResponse(BaseModel):
    """Response model for the /reason/text endpoint."""

    output: List[float] = Field(
        ...,
        description="Model output vector",
    )
    dag_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="DAG statistics",
    )
    text_embedding_dim: int = Field(
        ...,
        description="Dimension of the text embedding used",
    )


class DAGVisualizationResponse(BaseModel):
    """Response model for the /dag/visualize endpoint."""

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="DAG metadata",
    )
    nodes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of DAG nodes",
    )
    edges: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of DAG edges",
    )


class ModelInfoResponse(BaseModel):
    """Response model for the /model/info endpoint."""

    config: Dict[str, int] = Field(
        default_factory=dict,
        description="Model configuration",
    )
    total_parameters: int = Field(
        default=0,
        description="Estimated total number of parameters",
    )
    components: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model components information",
    )


class BenchmarkResponse(BaseModel):
    """Response model for the /benchmark endpoint."""

    name: str = Field(
        default="",
        description="Benchmark name",
    )
    throughput_samples_per_sec: float = Field(
        default=0.0,
        description="Throughput in samples per second",
    )
    time_per_step_ms: float = Field(
        default=0.0,
        description="Average time per reasoning step in milliseconds",
    )
    time_per_sample_ms: float = Field(
        default=0.0,
        description="Average time per sample in milliseconds",
    )
    memory_estimate_mb: float = Field(
        default=0.0,
        description="Estimated memory usage in MB",
    )
    config: Dict[str, int] = Field(
        default_factory=dict,
        description="Model configuration used",
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
# Default number of dendritic branches in DCaAPCell
DEFAULT_NUM_BRANCHES = 4


# ============================
#  Helper Functions
# ============================


def _generate_visualizer_output(
    visualizer: DAGVisualizer,
    method: Callable[[str], str],
    suffix: str,
) -> str:
    """Generate visualizer output to a temp file and read it back.

    Args:
        visualizer: The DAGVisualizer instance
        method: The visualization method to call (e.g., to_html, to_dot)
        suffix: File suffix (e.g., '.html', '.dot')

    Returns:
        The generated content as a string
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=suffix, delete=False
    ) as tmp:
        method(tmp.name)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "r", encoding="utf-8") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def _estimate_model_parameters(config: Dict[str, int]) -> Dict[str, Any]:
    """Estimate the number of parameters in the model.

    Args:
        config: Model configuration dict with x_dim, y_dim, z_dim, hidden_dim, num_experts

    Returns:
        Dict with parameter counts for each component and total
    """
    x_dim = config["x_dim"]
    y_dim = config["y_dim"]
    z_dim = config["z_dim"]
    hidden_dim = config["hidden_dim"]
    num_experts = config["num_experts"]

    # x_encoder: x_dim * x_dim + x_dim
    x_encoder_params = x_dim * x_dim + x_dim

    # Each expert DCaAPCell has multiple linear layers
    # Input dimension combines x, y, and z
    input_dim = x_dim + y_dim + z_dim

    # branch_weights: (input_dim) * (hidden_dim // num_branches) per branch
    branch_dim = hidden_dim // DEFAULT_NUM_BRANCHES
    branch_params = input_dim * branch_dim + branch_dim

    # soma: hidden_dim * hidden_dim + hidden_dim
    soma_params = hidden_dim * hidden_dim + hidden_dim

    # calcium_gate: hidden_dim * 1 + 1
    calcium_params = hidden_dim + 1

    # output: hidden_dim * z_dim + z_dim
    output_params = hidden_dim * z_dim + z_dim

    expert_params = (
        DEFAULT_NUM_BRANCHES * branch_params
        + soma_params
        + calcium_params
        + output_params
    )
    total_expert_params = num_experts * expert_params

    # Router: projection, centroids, mass_projection
    projection_dim = 64  # Default projection dimension in TorqueRouter
    router_params = (
        input_dim * projection_dim + projection_dim  # projection
        + num_experts * projection_dim  # centroids
        + input_dim + 1  # mass_projection
    )

    # Answer dense layers
    answer_params = (
        (y_dim + z_dim) * hidden_dim + hidden_dim  # dense1
        + hidden_dim * y_dim + y_dim  # dense2
    )

    total_params = (
        x_encoder_params + total_expert_params + router_params + answer_params
    )

    return {
        "x_encoder_params": x_encoder_params,
        "expert_params": expert_params,
        "total_expert_params": total_expert_params,
        "router_params": router_params,
        "answer_params": answer_params,
        "total_params": total_params,
        "num_branches": DEFAULT_NUM_BRANCHES,
    }


# ============================
#  Application State
# ============================


class AppState:
    """Application state container for model, encoders, and config."""

    def __init__(self) -> None:
        self.model: Optional[TRLinkosTRM] = None
        self.config: Dict[str, int] = {}
        self.text_encoder: Optional[TextEncoder] = None
        self.last_dag: Optional[FractalMerkleDAG] = None

    def initialize(self) -> None:
        """Initialize the TRM model and encoders."""
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
        # Initialize text encoder with output_dim matching model's x_dim
        self.text_encoder = TextEncoder(
            vocab_size=256,
            embed_dim=128,
            output_dim=self.config["x_dim"],
            mode="char",
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
        "Torque Clustering routing, and cryptographic audit trails. "
        "Includes batch processing, text encoding, DAG visualization, and benchmarking."
    ),
    version="2.0.0",
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
            - features: List of floats (length must match x_dim, default 64)
            - max_steps: Maximum reasoning steps (default 10)
            - inner_recursions: Inner recursions per step (default 3)
            - backtrack: Enable backtracking to best states (default false)

    Returns:
        ReasoningResponse: Model output and DAG statistics.

    Example:
        curl -X POST http://localhost:8000/reason \\
            -H "Content-Type: application/json" \\
            -d '{"features": [<64 float values>], "max_steps": 10}'

        # Returns:
        # {
        #   "output": [<32 float values>],
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
        # Store the last DAG for visualization
        app_state.last_dag = dag
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


@app.post("/reason/batch", response_model=BatchReasoningResponse)
async def reason_batch(request: BatchReasoningRequest) -> BatchReasoningResponse:
    """
    Run batch recursive reasoning on multiple input feature vectors.

    This endpoint processes multiple inputs in a single batch, which is more
    efficient than making separate requests for each input.

    Args:
        request: BatchReasoningRequest with batch of features and optional parameters.
            - batch: List of feature vectors (each must match x_dim)
            - max_steps: Maximum reasoning steps (default 10)
            - inner_recursions: Inner recursions per step (default 3)
            - backtrack: Enable backtracking to best states (default false)

    Returns:
        BatchReasoningResponse: List of outputs and aggregate DAG statistics.

    Example:
        curl -X POST http://localhost:8000/reason/batch \\
            -H "Content-Type: application/json" \\
            -d '{"batch": [[0.1, 0.2, ...], [0.3, 0.4, ...]], "max_steps": 5}'
    """
    if app_state.model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # Validate input dimensions
    expected_dim = app_state.config["x_dim"]
    for i, features in enumerate(request.batch):
        if len(features) != expected_dim:
            raise HTTPException(
                status_code=400,
                detail=f"Sample {i}: Expected {expected_dim} features, got {len(features)}",
            )

    # Convert to numpy array
    x = np.array(request.batch, dtype=np.float64)

    # Run reasoning
    try:
        y_pred, dag = app_state.model.forward_recursive(
            x,
            max_steps=request.max_steps,
            inner_recursions=request.inner_recursions,
            backtrack=request.backtrack,
        )
        # Store the last DAG for visualization
        app_state.last_dag = dag
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

    return BatchReasoningResponse(
        outputs=[row.tolist() for row in y_pred],
        dag_stats=dag_stats,
        batch_size=len(request.batch),
    )


@app.post("/reason/text", response_model=TextReasoningResponse)
async def reason_text(request: TextReasoningRequest) -> TextReasoningResponse:
    """
    Run recursive reasoning on text input.

    Uses the TextEncoder to convert text to embeddings, then performs
    recursive reasoning on the resulting feature vector.

    Args:
        request: TextReasoningRequest with text and optional parameters.
            - text: Input text string
            - max_steps: Maximum reasoning steps (default 10)
            - inner_recursions: Inner recursions per step (default 3)
            - backtrack: Enable backtracking to best states (default false)

    Returns:
        TextReasoningResponse: Output, DAG statistics, and embedding dimension.

    Example:
        curl -X POST http://localhost:8000/reason/text \\
            -H "Content-Type: application/json" \\
            -d '{"text": "What is machine learning?", "max_steps": 5}'
    """
    if app_state.model is None or app_state.text_encoder is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # Encode text to features
    try:
        x = app_state.text_encoder.encode([request.text], max_length=128)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text encoding error: {str(e)}")

    # Run reasoning
    try:
        y_pred, dag = app_state.model.forward_recursive(
            x,
            max_steps=request.max_steps,
            inner_recursions=request.inner_recursions,
            backtrack=request.backtrack,
        )
        # Store the last DAG for visualization
        app_state.last_dag = dag
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

    return TextReasoningResponse(
        output=y_pred[0].tolist(),
        dag_stats=dag_stats,
        text_embedding_dim=app_state.config["x_dim"],
    )


@app.get("/dag/visualize", response_model=None)
async def visualize_dag(
    format: str = Query(
        default="json",
        description="Output format: 'json', 'html', or 'dot'",
        pattern="^(json|html|dot)$",
    ),
):
    """
    Get visualization data for the last reasoning DAG.

    Returns the DAG structure from the most recent reasoning operation
    in the requested format.

    Args:
        format: Output format ('json', 'html', or 'dot')

    Returns:
        DAG visualization in the requested format:
        - json: Structured JSON with nodes and edges
        - html: Interactive D3.js visualization
        - dot: Graphviz DOT format

    Example:
        curl http://localhost:8000/dag/visualize?format=json
        curl http://localhost:8000/dag/visualize?format=html
    """
    if app_state.last_dag is None:
        raise HTTPException(
            status_code=404,
            detail="No DAG available. Run /reason first.",
        )

    visualizer = DAGVisualizer(app_state.last_dag)

    if format == "json":
        data = visualizer.to_json()
        return DAGVisualizationResponse(
            metadata=data["metadata"],
            nodes=data["nodes"],
            edges=data["edges"],
        )
    elif format == "html":
        html_content = _generate_visualizer_output(
            visualizer, visualizer.to_html, ".html"
        )
        return HTMLResponse(content=html_content)
    elif format == "dot":
        dot_content = _generate_visualizer_output(
            visualizer, visualizer.to_dot, ".dot"
        )
        return PlainTextResponse(content=dot_content, media_type="text/plain")
    else:
        raise HTTPException(status_code=400, detail=f"Unknown format: {format}")


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """
    Get detailed information about the current model.

    Returns model configuration, estimated parameter count, and
    component information.

    Returns:
        ModelInfoResponse: Model configuration and statistics.

    Example:
        curl http://localhost:8000/model/info
    """
    if app_state.model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    config = app_state.config

    # Use helper function for parameter estimation
    params = _estimate_model_parameters(config)

    components = {
        "x_encoder": {"params": params["x_encoder_params"]},
        "experts": {
            "count": config["num_experts"],
            "params_per_expert": params["expert_params"],
            "total_params": params["total_expert_params"],
        },
        "router": {"params": params["router_params"]},
        "answer_layers": {"params": params["answer_params"]},
        "text_encoder": {
            "available": app_state.text_encoder is not None,
            "output_dim": config["x_dim"],
        },
    }

    return ModelInfoResponse(
        config=config,
        total_parameters=params["total_params"],
        components=components,
    )


@app.get("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(
    batch_size: int = Query(default=8, ge=1, le=64, description="Batch size"),
    max_steps: int = Query(default=8, ge=1, le=32, description="Max reasoning steps"),
    num_runs: int = Query(default=3, ge=1, le=10, description="Number of timing runs"),
) -> BenchmarkResponse:
    """
    Run a performance benchmark on the model.

    Measures throughput, latency, and memory usage for the
    forward_recursive method.

    Args:
        batch_size: Number of samples per batch (1-64)
        max_steps: Maximum reasoning steps (1-32)
        num_runs: Number of timing runs for averaging (1-10)

    Returns:
        BenchmarkResponse: Performance metrics.

    Example:
        curl "http://localhost:8000/benchmark?batch_size=16&max_steps=10"
    """
    if app_state.model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        result = benchmark_forward_recursive(
            app_state.model,
            batch_size=batch_size,
            max_steps=max_steps,
            inner_recursions=3,
            num_runs=num_runs,
            warmup_runs=1,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark error: {str(e)}")

    return BenchmarkResponse(
        name=result.name,
        throughput_samples_per_sec=result.throughput,
        time_per_step_ms=result.time_per_step * 1000,
        time_per_sample_ms=result.time_per_sample * 1000,
        memory_estimate_mb=result.memory_estimate_mb,
        config=result.config,
    )


# ============================
#  Main Entry Point
# ============================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
