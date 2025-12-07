"""
Enhanced FastAPI Web API for T-RLINKOS TRM++ with Blueprint Integration

This enhanced API integrates all AI Architecture Blueprints from THE-BLUEPRINTS.md:
- Safety Guardrails: Input/output validation for all requests
- AI Observability: Real-time metrics and health monitoring
- Resilient Workflow: Automatic retry and error handling
- Goal Monitoring: Track progress toward reasoning objectives

New endpoints:
- GET /health/detailed: Comprehensive health check with all metrics
- GET /metrics: Get observability metrics
- GET /safety/stats: Get safety statistics
- POST /reason/safe: Safe reasoning with all blueprint features
- POST /reason/goal: Goal-oriented reasoning with progress tracking

Usage:
    # Start the enhanced server
    uvicorn api_enhanced:app --reload --port 8000
    
    # Or run directly
    python api_enhanced.py

Example curl commands:
    # Detailed health check
    curl http://localhost:8000/health/detailed
    
    # Safe reasoning with validation
    curl -X POST http://localhost:8000/reason/safe \\
        -H "Content-Type: application/json" \\
        -d '{"features": [0.1, 0.2, ...], "max_steps": 10}'
    
    # Get metrics
    curl http://localhost:8000/metrics
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import time

import numpy as np

try:
    from fastapi import FastAPI, HTTPException, Query, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
except ImportError as e:
    raise ImportError(
        "FastAPI is required for the web API. "
        "Install with: pip install fastapi uvicorn"
    ) from e

# Core T-RLINKOS imports
from t_rlinkos_trm_fractal_dag import TRLinkosTRM

# Blueprint imports
from blueprints import (
    EnhancedTRLinkosTRM,
    EnhancedTRMConfig,
    GoalDefinition,
    SuccessCriteria,
)

# ============================
#  Configuration
# ============================

DEFAULT_X_DIM = 64
DEFAULT_Y_DIM = 32
DEFAULT_Z_DIM = 64

# Global model instance
enhanced_model: Optional[EnhancedTRLinkosTRM] = None


# ============================
#  Pydantic Models
# ============================


class SafeReasoningRequest(BaseModel):
    """Request model for safe reasoning with validation."""
    
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
        description="Enable backtracking",
    )
    enable_safety: bool = Field(
        default=True,
        description="Enable safety guardrails",
    )
    
    @validator('features')
    def validate_features_length(cls, v):
        """Validate features have correct dimension."""
        if len(v) != DEFAULT_X_DIM:
            raise ValueError(f"Features must have length {DEFAULT_X_DIM}, got {len(v)}")
        return v


class GoalReasoningRequest(BaseModel):
    """Request model for goal-oriented reasoning."""
    
    features: List[float] = Field(
        ...,
        description="Input features vector",
    )
    goal_description: str = Field(
        ...,
        description="Description of the reasoning goal",
    )
    min_score: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable score",
    )
    max_steps: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum reasoning steps",
    )
    max_time: Optional[float] = Field(
        default=None,
        description="Maximum time in seconds",
    )


class SafeReasoningResponse(BaseModel):
    """Response model for safe reasoning."""
    
    success: bool = Field(..., description="Whether reasoning succeeded")
    predictions: Optional[List[float]] = Field(None, description="Output predictions")
    error: Optional[str] = Field(None, description="Error message if failed")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    validation_reports: Dict[str, Any] = Field(default_factory=dict, description="Validation reports")
    dag_stats: Dict[str, Any] = Field(default_factory=dict, description="DAG statistics")


class GoalReasoningResponse(BaseModel):
    """Response model for goal-oriented reasoning."""
    
    success: bool = Field(..., description="Whether goal was achieved")
    predictions: Optional[List[float]] = Field(None, description="Output predictions")
    goal_status: Dict[str, Any] = Field(default_factory=dict, description="Goal progress and status")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")


class HealthResponse(BaseModel):
    """Detailed health response."""
    
    status: str = Field(..., description="Overall status: healthy, degraded, or unhealthy")
    is_healthy: bool = Field(..., description="Boolean health indicator")
    message: str = Field(..., description="Health message")
    issues: List[str] = Field(default_factory=list, description="List of issues")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Health metrics")
    uptime_seconds: float = Field(..., description="System uptime")


# ============================
#  Lifecycle Management
# ============================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI app."""
    # Startup
    global enhanced_model
    
    print("Initializing Enhanced T-RLINKOS TRM++ with Blueprints...")
    
    # Create base model
    base_model = TRLinkosTRM(
        x_dim=DEFAULT_X_DIM,
        y_dim=DEFAULT_Y_DIM,
        z_dim=DEFAULT_Z_DIM,
        hidden_dim=256,
        num_experts=4,
    )
    
    # Wrap with blueprint features
    config = EnhancedTRMConfig(
        enable_safety_guardrails=True,
        enable_observability=True,
        enable_resilient_workflow=True,
        enable_goal_monitoring=True,
    )
    
    enhanced_model = EnhancedTRLinkosTRM(base_model, config)
    
    print("✅ Enhanced model initialized successfully!")
    print(f"   - Safety Guardrails: Enabled")
    print(f"   - Observability: Enabled")
    print(f"   - Resilient Workflow: Enabled")
    print(f"   - Goal Monitoring: Enabled")
    
    yield
    
    # Shutdown
    print("Shutting down Enhanced T-RLINKOS TRM++...")


# ============================
#  FastAPI Application
# ============================


app = FastAPI(
    title="Enhanced T-RLINKOS TRM++ API",
    description="Enterprise-grade API with AI Architecture Blueprints integration",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================
#  Health & Monitoring Endpoints
# ============================


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Enhanced T-RLINKOS TRM++ API with Blueprint Integration",
        "version": "2.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Basic health check."""
    if enhanced_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized",
        )
    
    return {
        "status": "ok",
        "model": "T-RLINKOS TRM++ with Blueprints",
        "features": {
            "safety_guardrails": True,
            "observability": True,
            "resilient_workflow": True,
            "goal_monitoring": True,
        },
    }


@app.get("/health/detailed", response_model=HealthResponse)
async def detailed_health_check():
    """Detailed health check with all metrics."""
    if enhanced_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized",
        )
    
    health = enhanced_model.get_health()
    dashboard = enhanced_model.get_dashboard()
    
    if health is None:
        return HealthResponse(
            status="unknown",
            is_healthy=True,
            message="Health monitoring not available",
            uptime_seconds=dashboard.get("observability", {}).get("uptime_seconds", 0),
        )
    
    status_str = "healthy" if health["is_healthy"] else "unhealthy"
    if health["issues"] and len(health["issues"]) < 3:
        status_str = "degraded"
    
    return HealthResponse(
        status=status_str,
        is_healthy=health["is_healthy"],
        message=health["message"],
        issues=health["issues"],
        metrics=health["metrics"],
        uptime_seconds=dashboard.get("observability", {}).get("uptime_seconds", 0),
    )


@app.get("/metrics", response_model=Dict[str, Any])
async def get_metrics():
    """Get observability metrics."""
    if enhanced_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized",
        )
    
    return enhanced_model.get_metrics()


@app.get("/safety/stats", response_model=Dict[str, Any])
async def get_safety_stats():
    """Get safety statistics."""
    if enhanced_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized",
        )
    
    return enhanced_model.get_safety_stats()


@app.get("/resilience/status", response_model=Dict[str, Any])
async def get_resilience_status():
    """Get resilience status."""
    if enhanced_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized",
        )
    
    return enhanced_model.get_resilience_status()


@app.get("/dashboard", response_model=Dict[str, Any])
async def get_dashboard():
    """Get complete dashboard data."""
    if enhanced_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized",
        )
    
    return enhanced_model.get_dashboard()


# ============================
#  Reasoning Endpoints
# ============================


@app.post("/reason/safe", response_model=SafeReasoningResponse)
async def reason_safe(request: SafeReasoningRequest):
    """
    Safe reasoning with all blueprint features.
    
    This endpoint applies:
    - Input validation and sanitization
    - Resilient execution with retries
    - Output validation
    - Metrics collection
    """
    if enhanced_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized",
        )
    
    # Convert features to numpy array
    x = np.array(request.features).reshape(1, -1)
    
    # Execute safe reasoning
    result = enhanced_model.forward_safe(
        x,
        max_steps=request.max_steps,
        inner_recursions=request.inner_recursions,
        backtrack=request.backtrack,
    )
    
    # Build response
    response = SafeReasoningResponse(
        success=result["success"],
        error=result.get("error"),
        metrics=result.get("metrics", {}),
        validation_reports=result.get("validation_reports", {}),
    )
    
    if result["success"]:
        response.predictions = result["predictions"][0].tolist()
        
        # Add DAG stats
        dag = result.get("dag")
        if dag:
            response.dag_stats = {
                "num_nodes": len(dag.nodes) if hasattr(dag, "nodes") else 0,
                "max_depth": max([n.depth for n in dag.nodes]) if hasattr(dag, "nodes") and dag.nodes else 0,
            }
    
    return response


@app.post("/reason/goal", response_model=GoalReasoningResponse)
async def reason_goal(request: GoalReasoningRequest):
    """
    Goal-oriented reasoning with progress tracking.
    
    This endpoint tracks progress toward a specific reasoning goal
    and provides detailed status updates.
    """
    if enhanced_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized",
        )
    
    # Convert features to numpy array
    x = np.array(request.features).reshape(1, -1)
    
    # Define goal
    def score_threshold(state):
        return state.get("score", 0) >= request.min_score
    
    goal = GoalDefinition(
        goal_id=f"goal_{int(time.time())}",
        description=request.goal_description,
        success_criteria=[
            SuccessCriteria(
                "min_score",
                f"Achieve score >= {request.min_score}",
                score_threshold,
                is_required=True,
            )
        ],
        max_steps=request.max_steps,
        max_time=request.max_time,
        min_score=request.min_score,
    )
    
    # Execute with goal monitoring
    result = enhanced_model.forward_safe(
        x,
        max_steps=request.max_steps,
        goal=goal,
    )
    
    # Build response
    response = GoalReasoningResponse(
        success=result["success"] and result.get("goal_status", {}).get("status") == "achieved",
        metrics=result.get("metrics", {}),
        goal_status=result.get("goal_status", {}),
    )
    
    if result["success"]:
        response.predictions = result["predictions"][0].tolist()
    
    return response


@app.post("/metrics/reset")
async def reset_metrics():
    """Reset all metrics and monitoring state."""
    if enhanced_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not initialized",
        )
    
    enhanced_model.reset_metrics()
    
    return {"status": "ok", "message": "Metrics reset successfully"}


# ============================
#  Main Entry Point
# ============================


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Enhanced T-RLINKOS TRM++ API with Blueprints...")
    print("Access API docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        "api_enhanced:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
