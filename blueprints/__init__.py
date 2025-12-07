"""
AI Architecture Blueprints Integration for T-RLINKOS TRM++

This package implements the AI architecture patterns from THE-BLUEPRINTS.md,
integrating enterprise-grade safety, observability, and resilience into the
T-RLINKOS recursive reasoning system.

Implemented Patterns:
- Safety Guardrails: Input validation and output filtering
- AI Observability: Real-time monitoring and metrics
- Resilient Workflow: Error handling and retry mechanisms
- Goal Monitoring: Progress tracking and success criteria
"""

from .safety_guardrails import SafetyGuardrail, InputValidator, OutputValidator
from .observability import AIObservability, MetricsCollector, HealthMonitor
from .resilient_workflow import ResilientWorkflow, RetryStrategy, ErrorHandler
from .goal_monitoring import GoalMonitor, SuccessCriteria, ProgressTracker

__all__ = [
    "SafetyGuardrail",
    "InputValidator",
    "OutputValidator",
    "AIObservability",
    "MetricsCollector",
    "HealthMonitor",
    "ResilientWorkflow",
    "RetryStrategy",
    "ErrorHandler",
    "GoalMonitor",
    "SuccessCriteria",
    "ProgressTracker",
]

__version__ = "1.0.0"
