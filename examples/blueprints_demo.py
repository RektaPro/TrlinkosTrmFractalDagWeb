"""
Complete demonstration of AI Architecture Blueprints integration with T-RLINKOS TRM++

This script demonstrates all blueprint patterns:
1. Safety Guardrails
2. AI Observability
3. Resilient Workflow
4. Goal Monitoring

Run with: python examples/blueprints_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time

from t_rlinkos_trm_fractal_dag import TRLinkosTRM
from blueprints import (
    EnhancedTRLinkosTRM,
    EnhancedTRMConfig,
    SafetyGuardrail,
    AIObservability,
    ResilientWorkflow,
    GoalMonitor,
    GoalDefinition,
    SuccessCriteria,
)


def print_section(title):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_safety_guardrails():
    """Demonstrate Safety Guardrails Pattern."""
    print_section("1. Safety Guardrails Pattern")
    
    print("✓ Creating safety guardrail with input validation...")
    guardrail = SafetyGuardrail(auto_sanitize=True)
    
    # Valid input
    print("\nTest 1: Valid input")
    x_valid = np.random.randn(8, 64)
    is_valid, x_clean, report = guardrail.validate_input(x_valid)
    print(f"  Result: {report.result.value}")
    print(f"  Message: {report.message}")
    
    # Invalid input (with NaN)
    print("\nTest 2: Invalid input with NaN")
    x_invalid = x_valid.copy()
    x_invalid[0, 0] = np.nan
    is_valid, x_clean, report = guardrail.validate_input(x_invalid)
    print(f"  Result: {report.result.value}")
    print(f"  Message: {report.message}")
    print(f"  Auto-sanitized: {not np.any(np.isnan(x_clean))}")
    
    # Statistics
    print("\nSafety Statistics:")
    stats = guardrail.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def demo_observability():
    """Demonstrate AI Observability Pattern."""
    print_section("2. AI Observability Pattern")
    
    print("✓ Creating observability system...")
    obs = AIObservability(
        enable_metrics=True,
        enable_health_checks=True,
    )
    
    # Simulate some inferences
    print("\nSimulating 20 inference operations...")
    for i in range(20):
        latency = 10 + np.random.randn() * 2  # 10ms average
        steps = np.random.randint(5, 15)
        nodes = steps * 5
        success = np.random.random() > 0.1  # 90% success rate
        
        obs.record_inference(
            latency_ms=latency,
            num_steps=steps,
            dag_nodes=nodes,
            success=success,
        )
    
    # Get metrics
    print("\nObservability Metrics:")
    metrics = obs.get_statistics()
    if "metrics" in metrics and "inference_latency_ms" in metrics["metrics"]:
        latency_stats = metrics["metrics"]["inference_latency_ms"]
        print(f"  Latency - Mean: {latency_stats['mean']:.2f}ms")
        print(f"  Latency - P95: {latency_stats['p95']:.2f}ms")
        print(f"  Latency - P99: {latency_stats['p99']:.2f}ms")
    
    print(f"\nCounters:")
    counters = metrics.get("counters", {})
    for key, value in counters.items():
        print(f"  {key}: {value}")
    
    # Health check
    print("\nHealth Status:")
    health = obs.check_health()
    if health:
        print(f"  Healthy: {health.is_healthy}")
        print(f"  Message: {health.message}")
        if health.issues:
            print(f"  Issues: {health.issues}")


def demo_resilient_workflow():
    """Demonstrate Resilient Workflow Pattern."""
    print_section("3. Resilient Workflow Pattern")
    
    print("✓ Creating resilient workflow with retry and circuit breaker...")
    workflow = ResilientWorkflow(
        enable_retry=True,
        enable_circuit_breaker=True,
    )
    
    # Test 1: Function that succeeds after retries
    print("\nTest 1: Flaky function (succeeds on 3rd attempt)")
    call_count = [0]
    
    def flaky_func():
        call_count[0] += 1
        if call_count[0] < 3:
            raise ValueError(f"Attempt {call_count[0]} failed")
        return "success"
    
    result = workflow.execute(flaky_func)
    print(f"  Success: {result.success}")
    print(f"  Attempts: {result.attempts}")
    print(f"  Value: {result.value}")
    print(f"  Total time: {result.total_time:.3f}s")
    
    # Test 2: Function that always fails
    print("\nTest 2: Function that always fails")
    
    def always_fails():
        raise RuntimeError("Always fails")
    
    result = workflow.execute(always_fails, use_fallback=False)
    print(f"  Success: {result.success}")
    print(f"  Attempts: {result.attempts}")
    print(f"  Error: {type(result.error).__name__}")
    
    # Status
    print("\nResilience Status:")
    status = workflow.get_status()
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")


def demo_goal_monitoring():
    """Demonstrate Goal and Monitoring Pattern."""
    print_section("4. Goal and Monitoring Pattern")
    
    print("✓ Creating goal with success criteria...")
    
    # Define success criteria
    def score_threshold(state):
        return state.get("score", 0) > 0.8
    
    def efficiency(state):
        return state.get("step", 0) < 50
    
    goal = GoalDefinition(
        goal_id="demo_goal",
        description="Achieve high score efficiently",
        success_criteria=[
            SuccessCriteria(
                "high_score",
                "Score > 0.8",
                score_threshold,
                is_required=True,
                weight=2.0,
            ),
            SuccessCriteria(
                "efficient",
                "Complete in < 50 steps",
                efficiency,
                weight=1.0,
            ),
        ],
        max_steps=100,
        min_score=0.7,
    )
    
    # Monitor progress
    print("\nMonitoring progress toward goal...")
    monitor = GoalMonitor(goal)
    monitor.start()
    
    for step in range(30):
        # Simulate improving score
        score = 0.5 + step * 0.02
        state = {"score": score, "step": step}
        
        result = monitor.update(step, score, state)
        
        # Print progress every 5 steps
        if step % 5 == 0:
            print(f"  Step {step:2d}: score={score:.2f}, status={result['status']}, progress={result.get('progress_score', 0):.2f}")
        
        # Check if goal achieved
        if result["status"] == "achieved":
            print(f"\n✓ Goal achieved at step {step}!")
            break
    
    # Final report
    print("\nFinal Goal Report:")
    report = monitor.get_status_report()
    print(f"  Status: {report['status']}")
    print(f"  Total steps: {report['progress']['total_steps']}")
    print(f"  Score improvement: {report['progress']['score_improvement']:.2f}")
    print(f"\n  Criteria:")
    for criteria in report['criteria']:
        print(f"    - {criteria['name']}: {'✓' if criteria['currently_met'] else '✗'}")
        print(f"      {criteria['description']}")
        print(f"      Met {criteria['met_count']}/{criteria['total_checks']} times")


def demo_enhanced_trm():
    """Demonstrate Enhanced TRM with all blueprints."""
    print_section("5. Enhanced TRM with All Blueprints")
    
    print("✓ Creating enhanced TRM with all blueprint features...")
    
    # Create base model
    base_model = TRLinkosTRM(64, 32, 64, hidden_dim=128, num_experts=4)
    
    # Wrap with blueprints
    config = EnhancedTRMConfig(
        enable_safety_guardrails=True,
        enable_observability=True,
        enable_resilient_workflow=True,
        enable_goal_monitoring=True,
    )
    
    model = EnhancedTRLinkosTRM(base_model, config)
    
    print("  ✓ Safety Guardrails: Enabled")
    print("  ✓ Observability: Enabled")
    print("  ✓ Resilient Workflow: Enabled")
    print("  ✓ Goal Monitoring: Enabled")
    
    # Test inference
    print("\nRunning safe inference...")
    x = np.random.randn(4, 64)
    
    result = model.forward_safe(x, max_steps=10, backtrack=True)
    
    print(f"\nInference Result:")
    print(f"  Success: {result['success']}")
    
    if result['success']:
        print(f"  Predictions shape: {result['predictions'].shape}")
        print(f"  Latency: {result['metrics']['latency_ms']:.2f}ms")
        print(f"  Reasoning steps: {result['metrics']['num_steps']}")
        print(f"  DAG nodes: {result['metrics']['dag_nodes']}")
        
        print(f"\nValidation Reports:")
        for check_type, report in result['validation_reports'].items():
            print(f"  {check_type}: {report['result']} - {report['message']}")
    else:
        print(f"  Error: {result['error']}")
    
    # Run multiple inferences to collect metrics
    print("\nRunning 5 more inferences to collect metrics...")
    for i in range(5):
        x = np.random.randn(2, 64)
        result = model.forward_safe(x, max_steps=8)
    
    print("  ✓ Completed")
    
    # Get comprehensive dashboard
    print("\nComplete Dashboard:")
    dashboard = model.get_dashboard()
    
    print(f"\nModel Configuration:")
    for key, value in dashboard['model_config'].items():
        print(f"  {key}: {value}")
    
    print(f"\nBlueprint Features:")
    for key, value in dashboard['blueprint_features'].items():
        print(f"  {key}: {value}")
    
    if 'observability' in dashboard:
        obs_data = dashboard['observability']
        print(f"\nObservability:")
        print(f"  Uptime: {obs_data.get('uptime_seconds', 0):.1f}s")
        
        if 'metrics' in obs_data:
            metrics = obs_data['metrics']
            if 'counters' in metrics:
                counters = metrics['counters']
                print(f"  Total requests: {counters.get('total_requests', 0)}")
                print(f"  Failed requests: {counters.get('failed_requests', 0)}")
    
    if 'safety' in dashboard:
        safety = dashboard['safety']
        print(f"\nSafety:")
        print(f"  Total inputs: {safety.get('total_inputs', 0)}")
        print(f"  Input failures: {safety.get('input_failures', 0)}")
        print(f"  Auto-sanitizations: {safety.get('auto_sanitizations', 0)}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("  T-RLINKOS TRM++ AI Architecture Blueprints")
    print("  Complete Integration Demonstration")
    print("=" * 70)
    
    try:
        # Run all demos
        demo_safety_guardrails()
        demo_observability()
        demo_resilient_workflow()
        demo_goal_monitoring()
        demo_enhanced_trm()
        
        # Summary
        print_section("Summary")
        print("✅ All blueprint patterns demonstrated successfully!")
        print("\nIntegrated Patterns:")
        print("  1. ✓ Safety Guardrails - Input/output validation")
        print("  2. ✓ AI Observability - Metrics and health monitoring")
        print("  3. ✓ Resilient Workflow - Retry and circuit breakers")
        print("  4. ✓ Goal Monitoring - Progress tracking")
        print("\nThe Enhanced TRM combines all patterns for production-ready AI.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
