"""
Enhanced T-RLINKOS TRM with Blueprint Pattern Integration

This module wraps the core T-RLINKOS TRM++ model with enterprise-grade
features from THE-BLUEPRINTS.md:

- Safety Guardrails: Input/output validation
- AI Observability: Real-time monitoring
- Resilient Workflow: Error handling and retries
- Goal Monitoring: Progress tracking

This provides a production-ready inference system with built-in safety,
observability, and resilience.
"""

import time
import numpy as np
from typing import Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass

from .safety_guardrails import SafetyGuardrail, InputValidator, OutputValidator
from .observability import AIObservability
from .resilient_workflow import ResilientWorkflow, RetryConfig
from .goal_monitoring import GoalMonitor, GoalDefinition, SuccessCriteria


@dataclass
class EnhancedTRMConfig:
    """Configuration for enhanced TRM with blueprint patterns."""
    # Safety features
    enable_safety_guardrails: bool = True
    auto_sanitize_inputs: bool = True
    input_value_range: Tuple[float, float] = (-1e6, 1e6)
    
    # Observability features
    enable_observability: bool = True
    health_check_interval: float = 60.0
    
    # Resilience features
    enable_resilient_workflow: bool = True
    max_retry_attempts: int = 3
    retry_initial_delay: float = 0.1
    
    # Goal monitoring features
    enable_goal_monitoring: bool = False  # Opt-in for specific tasks
    
    # Performance
    enable_metrics: bool = True


class EnhancedTRLinkosTRM:
    """
    Enterprise-grade T-RLINKOS TRM with integrated blueprint patterns.
    
    This wrapper adds safety, observability, resilience, and goal tracking
    to the core T-RLINKOS recursive reasoning model.
    
    Example:
        ```python
        from t_rlinkos_trm_fractal_dag import TRLinkosTRM
        from blueprints import EnhancedTRLinkosTRM, EnhancedTRMConfig
        
        # Create base model
        base_model = TRLinkosTRM(64, 32, 64)
        
        # Wrap with enterprise features
        config = EnhancedTRMConfig(enable_safety_guardrails=True)
        model = EnhancedTRLinkosTRM(base_model, config)
        
        # Use safely
        x = np.random.randn(8, 64)
        result = model.forward_safe(x, max_steps=10)
        
        if result["success"]:
            y_pred = result["predictions"]
            print(f"Predictions: {y_pred.shape}")
        else:
            print(f"Error: {result['error']}")
        
        # Check metrics
        metrics = model.get_metrics()
        print(f"Average latency: {metrics['metrics']['inference_latency_ms']['mean']:.2f}ms")
        ```
    """
    
    def __init__(
        self,
        base_model: Any,
        config: Optional[EnhancedTRMConfig] = None
    ):
        """
        Initialize enhanced TRM.
        
        Args:
            base_model: Base TRLinkosTRM instance
            config: Configuration for enhanced features
        """
        self.base_model = base_model
        self.config = config or EnhancedTRMConfig()
        
        # Initialize blueprint components
        self._init_safety_guardrails()
        self._init_observability()
        self._init_resilient_workflow()
        
        # Goal monitoring is created per-inference when needed
        self.current_goal_monitor: Optional[GoalMonitor] = None
    
    def _init_safety_guardrails(self):
        """Initialize safety guardrails."""
        if not self.config.enable_safety_guardrails:
            self.safety = None
            return
        
        # Create validators
        input_validator = InputValidator(
            expected_shape=(None, self.base_model.x_dim),
            value_range=self.config.input_value_range,
            allow_nan=False,
            allow_inf=False,
        )
        
        output_validator = OutputValidator(
            check_dag_stability=True,
        )
        
        self.safety = SafetyGuardrail(
            input_validator=input_validator,
            output_validator=output_validator,
            auto_sanitize=self.config.auto_sanitize_inputs,
            raise_on_failure=False,
        )
    
    def _init_observability(self):
        """Initialize observability."""
        if not self.config.enable_observability:
            self.observability = None
            return
        
        self.observability = AIObservability(
            enable_metrics=self.config.enable_metrics,
            enable_health_checks=True,
            health_check_interval=self.config.health_check_interval,
        )
    
    def _init_resilient_workflow(self):
        """Initialize resilient workflow."""
        if not self.config.enable_resilient_workflow:
            self.resilience = None
            return
        
        retry_config = RetryConfig(
            max_attempts=self.config.max_retry_attempts,
            initial_delay=self.config.retry_initial_delay,
        )
        
        self.resilience = ResilientWorkflow(
            retry_config=retry_config,
            enable_retry=True,
            enable_circuit_breaker=True,
        )
    
    def forward_safe(
        self,
        x: np.ndarray,
        max_steps: int = 10,
        inner_recursions: int = 3,
        scorer: Optional[Callable] = None,
        backtrack: bool = False,
        backtrack_threshold: float = 0.1,
        goal: Optional[GoalDefinition] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Safe forward pass with all blueprint features.
        
        Args:
            x: Input features [batch_size, x_dim]
            max_steps: Maximum reasoning steps
            inner_recursions: Inner recursions per step
            scorer: Optional scoring function
            backtrack: Enable backtracking
            backtrack_threshold: Backtracking threshold
            goal: Optional goal definition for monitoring
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with:
            - success: bool
            - predictions: np.ndarray (if success)
            - dag: FractalMerkleDAG (if success)
            - error: str (if failure)
            - metrics: Dict[str, Any]
            - validation_reports: Dict[str, Any]
        """
        start_time = time.time()
        result = {
            "success": False,
            "metrics": {},
            "validation_reports": {},
        }
        
        # 1. Input validation (Safety Guardrails)
        if self.safety:
            is_valid, x_clean, report = self.safety.validate_input(x)
            result["validation_reports"]["input"] = {
                "result": report.result.value,
                "message": report.message,
                "details": report.details,
            }
            
            if not is_valid:
                result["error"] = f"Input validation failed: {report.message}"
                return result
            
            x = x_clean  # Use sanitized input
        
        # 2. Setup goal monitoring if requested
        if goal and self.config.enable_goal_monitoring:
            self.current_goal_monitor = GoalMonitor(goal)
            self.current_goal_monitor.start()
        
        # 3. Execute with resilience
        def _inference():
            return self.base_model.forward_recursive(
                x,
                max_steps=max_steps,
                inner_recursions=inner_recursions,
                scorer=scorer,
                backtrack=backtrack,
                backtrack_threshold=backtrack_threshold,
                **kwargs
            )
        
        if self.resilience:
            exec_result = self.resilience.execute(_inference, use_fallback=False)
            
            if not exec_result.success:
                result["error"] = f"Inference failed: {str(exec_result.error)}"
                result["metrics"]["attempts"] = exec_result.attempts
                result["metrics"]["total_time"] = exec_result.total_time
                return result
            
            y_pred, dag = exec_result.value
            result["metrics"]["attempts"] = exec_result.attempts
            result["metrics"]["retry_time"] = exec_result.total_time
        else:
            try:
                y_pred, dag = _inference()
            except Exception as e:
                result["error"] = f"Inference error: {str(e)}"
                return result
        
        # 4. Output validation (Safety Guardrails)
        if self.safety:
            is_valid, output_report = self.safety.validate_output(y_pred, dag)
            result["validation_reports"]["output"] = {
                "result": output_report.result.value,
                "message": output_report.message,
                "details": output_report.details,
            }
            
            if not is_valid:
                result["error"] = f"Output validation failed: {output_report.message}"
                return result
        
        # 5. Record metrics (Observability)
        latency_ms = (time.time() - start_time) * 1000
        num_steps = dag.nodes[-1].step if dag.nodes else 0
        num_nodes = len(dag.nodes)
        
        if self.observability:
            self.observability.record_inference(
                latency_ms=latency_ms,
                num_steps=num_steps,
                dag_nodes=num_nodes,
                success=True,
            )
        
        # 6. Update goal monitoring
        if self.current_goal_monitor:
            best_node = dag.get_best_node()
            score = best_node.score if best_node else 0.0
            
            goal_update = self.current_goal_monitor.update(
                step=num_steps,
                score=score,
                state={
                    "score": score,
                    "step": num_steps,
                    "dag_nodes": num_nodes,
                }
            )
            result["goal_status"] = goal_update
        
        # Success!
        result["success"] = True
        result["predictions"] = y_pred
        result["dag"] = dag
        result["metrics"]["latency_ms"] = latency_ms
        result["metrics"]["num_steps"] = num_steps
        result["metrics"]["dag_nodes"] = num_nodes
        
        return result
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get observability metrics."""
        if not self.observability:
            return {}
        
        return self.observability.get_statistics()
    
    def get_health(self) -> Optional[Dict[str, Any]]:
        """Get health status."""
        if not self.observability:
            return None
        
        status = self.observability.check_health()
        if not status:
            return None
        
        return {
            "is_healthy": status.is_healthy,
            "message": status.message,
            "issues": status.issues,
            "metrics": status.metrics,
        }
    
    def get_safety_stats(self) -> Dict[str, Any]:
        """Get safety statistics."""
        if not self.safety:
            return {}
        
        return self.safety.get_statistics()
    
    def get_resilience_status(self) -> Dict[str, Any]:
        """Get resilience status."""
        if not self.resilience:
            return {}
        
        return self.resilience.get_status()
    
    def get_goal_report(self) -> Optional[Dict[str, Any]]:
        """Get current goal monitoring report."""
        if not self.current_goal_monitor:
            return None
        
        return self.current_goal_monitor.get_status_report()
    
    def get_dashboard(self) -> Dict[str, Any]:
        """
        Get complete dashboard data.
        
        Returns:
            Comprehensive dashboard with all metrics and status
        """
        dashboard = {
            "model_config": {
                "x_dim": self.base_model.x_dim,
                "y_dim": self.base_model.y_dim,
                "z_dim": self.base_model.z_dim,
            },
            "blueprint_features": {
                "safety_guardrails": self.config.enable_safety_guardrails,
                "observability": self.config.enable_observability,
                "resilient_workflow": self.config.enable_resilient_workflow,
                "goal_monitoring": self.config.enable_goal_monitoring,
            },
        }
        
        if self.observability:
            dashboard["observability"] = self.observability.get_dashboard_data()
        
        if self.safety:
            dashboard["safety"] = self.get_safety_stats()
        
        if self.resilience:
            dashboard["resilience"] = self.get_resilience_status()
        
        if self.current_goal_monitor:
            dashboard["goal"] = self.get_goal_report()
        
        return dashboard
    
    def reset_metrics(self):
        """Reset all metrics and monitoring state."""
        if self.observability:
            self.observability.reset()
        
        if self.safety:
            self.safety.reset_statistics()


if __name__ == "__main__":
    # Test enhanced TRM
    print("Testing Enhanced TRM with Blueprints...")
    
    # Import base model
    try:
        from t_rlinkos_trm_fractal_dag import TRLinkosTRM
    except ImportError:
        print("Warning: t_rlinkos_trm_fractal_dag not available, using mock")
        
        # Create a mock model for testing
        class MockTRM:
            def __init__(self, x_dim, y_dim, z_dim):
                self.x_dim = x_dim
                self.y_dim = y_dim
                self.z_dim = z_dim
            
            def forward_recursive(self, x, **kwargs):
                y_pred = np.random.randn(x.shape[0], self.y_dim)
                
                # Mock DAG
                class MockNode:
                    def __init__(self):
                        self.step = 5
                        self.score = 0.8
                
                class MockDAG:
                    def __init__(self):
                        self.nodes = [MockNode()]
                    
                    def get_best_node(self):
                        return self.nodes[0]
                
                return y_pred, MockDAG()
        
        TRLinkosTRM = MockTRM
    
    # Test 1: Basic enhanced TRM
    base_model = TRLinkosTRM(64, 32, 64)
    enhanced_model = EnhancedTRLinkosTRM(base_model)
    
    x = np.random.randn(8, 64)
    result = enhanced_model.forward_safe(x, max_steps=10)
    
    print(f"Test 1 - Basic inference: success={result['success']}")
    if result['success']:
        print(f"         Predictions shape: {result['predictions'].shape}")
        print(f"         Latency: {result['metrics']['latency_ms']:.2f}ms")
    
    # Test 2: With invalid input
    x_nan = x.copy()
    x_nan[0, 0] = np.nan
    result = enhanced_model.forward_safe(x_nan)
    print(f"Test 2 - Invalid input: success={result['success']}")
    print(f"         Validation: {result['validation_reports'].get('input', {}).get('message', 'N/A')}")
    
    # Test 3: Metrics
    metrics = enhanced_model.get_metrics()
    print(f"Test 3 - Metrics collected: {len(metrics.get('metrics', {}))} metric types")
    
    # Test 4: Dashboard
    dashboard = enhanced_model.get_dashboard()
    print(f"Test 4 - Dashboard: {len(dashboard)} sections")
    print(f"         Features enabled: {dashboard['blueprint_features']}")
    
    print("\n✅ Enhanced TRM tests passed!")
