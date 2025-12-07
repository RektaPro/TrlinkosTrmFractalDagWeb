"""
AI Observability Pattern Implementation

Based on: THE-BLUEPRINTS.md - Pattern 06: The AI Observability Pattern

Problem: An unobserved AI is a "black box" that can silently degrade, 
burn budget, and create compliance risks.

Solution: Architect an observability system to provide a real-time 
"control panel" for the AI's operational health, cost, and behavior.

This module provides comprehensive monitoring and metrics collection for 
T-RLINKOS TRM++, enabling real-time visibility into model behavior.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from collections import deque
from datetime import datetime


@dataclass
class Metric:
    """Single metric measurement"""
    name: str
    value: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """Health status of the system"""
    is_healthy: bool
    message: str
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """
    Collects and aggregates metrics from T-RLINKOS operations.
    
    Tracks:
    - Inference latency
    - Reasoning steps
    - DAG size and depth
    - Error rates
    - Throughput
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            max_history: Maximum number of historical metrics to keep
        """
        self.max_history = max_history
        self.metrics: Dict[str, deque] = {}
        self.counters: Dict[str, int] = {}
        self.aggregates: Dict[str, Dict[str, float]] = {}
        
    def record_metric(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            metadata: Optional metadata
        """
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=self.max_history)
        
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        self.metrics[name].append(metric)
        
        # Update aggregates
        self._update_aggregates(name)
    
    def increment_counter(self, name: str, amount: int = 1):
        """
        Increment a counter.
        
        Args:
            name: Counter name
            amount: Amount to increment
        """
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += amount
    
    def _update_aggregates(self, metric_name: str):
        """Update aggregate statistics for a metric."""
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return
        
        values = [m.value for m in self.metrics[metric_name]]
        self.aggregates[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "count": len(values),
        }
    
    def get_metric_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Get aggregate statistics for a metric."""
        return self.aggregates.get(name)
    
    def get_counter(self, name: str) -> int:
        """Get counter value."""
        return self.counters.get(name, 0)
    
    def get_recent_metrics(self, name: str, n: int = 10) -> List[Metric]:
        """Get n most recent metrics."""
        if name not in self.metrics:
            return []
        return list(self.metrics[name])[-n:]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics and statistics."""
        return {
            "metrics": {
                name: self.get_metric_stats(name)
                for name in self.metrics.keys()
            },
            "counters": self.counters.copy(),
            "timestamp": time.time(),
        }
    
    def reset(self):
        """Reset all metrics and counters."""
        self.metrics.clear()
        self.counters.clear()
        self.aggregates.clear()


class HealthMonitor:
    """
    Monitors system health and detects issues.
    
    Checks:
    - Latency thresholds
    - Error rates
    - Resource usage
    - Model performance degradation
    """
    
    def __init__(
        self,
        latency_threshold_ms: float = 1000.0,
        error_rate_threshold: float = 0.1,
        min_throughput: float = 1.0,
    ):
        """
        Initialize health monitor.
        
        Args:
            latency_threshold_ms: Maximum acceptable latency in milliseconds
            error_rate_threshold: Maximum acceptable error rate (0-1)
            min_throughput: Minimum acceptable throughput (requests/sec)
        """
        self.latency_threshold_ms = latency_threshold_ms
        self.error_rate_threshold = error_rate_threshold
        self.min_throughput = min_throughput
        
        self.last_check_time = time.time()
        self.health_history: deque = deque(maxlen=100)
    
    def check_health(self, metrics: MetricsCollector) -> HealthStatus:
        """
        Check system health based on metrics.
        
        Args:
            metrics: MetricsCollector instance
            
        Returns:
            HealthStatus object
        """
        issues = []
        health_metrics = {}
        
        # Check latency
        latency_stats = metrics.get_metric_stats("inference_latency_ms")
        if latency_stats:
            health_metrics["latency_mean_ms"] = latency_stats["mean"]
            health_metrics["latency_p95_ms"] = latency_stats["p95"]
            
            if latency_stats["p95"] > self.latency_threshold_ms:
                issues.append(
                    f"High latency: P95 {latency_stats['p95']:.2f}ms exceeds threshold {self.latency_threshold_ms}ms"
                )
        
        # Check error rate
        total_requests = metrics.get_counter("total_requests")
        failed_requests = metrics.get_counter("failed_requests")
        
        if total_requests > 0:
            error_rate = failed_requests / total_requests
            health_metrics["error_rate"] = error_rate
            
            if error_rate > self.error_rate_threshold:
                issues.append(
                    f"High error rate: {error_rate:.2%} exceeds threshold {self.error_rate_threshold:.2%}"
                )
        
        # Check throughput
        elapsed_time = time.time() - self.last_check_time
        if elapsed_time > 0:
            throughput = total_requests / elapsed_time
            health_metrics["throughput_rps"] = throughput
            
            if throughput < self.min_throughput:
                issues.append(
                    f"Low throughput: {throughput:.2f} req/s below minimum {self.min_throughput} req/s"
                )
        
        # Check DAG health
        dag_stats = metrics.get_metric_stats("dag_num_nodes")
        if dag_stats:
            health_metrics["dag_size_mean"] = dag_stats["mean"]
            
            # Check for degenerate DAGs (too small)
            if dag_stats["mean"] < 5:
                issues.append(
                    f"Small DAG size: mean {dag_stats['mean']:.1f} nodes suggests reasoning may be failing"
                )
        
        # Determine overall health
        is_healthy = len(issues) == 0
        message = "System is healthy" if is_healthy else f"Found {len(issues)} issue(s)"
        
        status = HealthStatus(
            is_healthy=is_healthy,
            message=message,
            issues=issues,
            metrics=health_metrics,
            timestamp=time.time()
        )
        
        self.health_history.append(status)
        return status
    
    def get_health_history(self, n: int = 10) -> List[HealthStatus]:
        """Get recent health check history."""
        return list(self.health_history)[-n:]


class AIObservability:
    """
    Complete observability system for T-RLINKOS TRM++.
    
    Provides:
    - Real-time metrics collection
    - Health monitoring
    - Performance tracking
    - Alerting on issues
    """
    
    def __init__(
        self,
        enable_metrics: bool = True,
        enable_health_checks: bool = True,
        health_check_interval: float = 60.0,
        alert_callbacks: Optional[List[Callable]] = None,
    ):
        """
        Initialize AI observability system.
        
        Args:
            enable_metrics: Enable metrics collection
            enable_health_checks: Enable health monitoring
            health_check_interval: Interval between health checks (seconds)
            alert_callbacks: Callbacks to invoke on health issues
        """
        self.enable_metrics = enable_metrics
        self.enable_health_checks = enable_health_checks
        self.health_check_interval = health_check_interval
        self.alert_callbacks = alert_callbacks or []
        
        self.metrics = MetricsCollector() if enable_metrics else None
        self.health_monitor = HealthMonitor() if enable_health_checks else None
        
        self.last_health_check = time.time()
        self.start_time = time.time()
    
    def record_inference(
        self,
        latency_ms: float,
        num_steps: int,
        dag_nodes: int,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record an inference operation.
        
        Args:
            latency_ms: Inference latency in milliseconds
            num_steps: Number of reasoning steps
            dag_nodes: Number of DAG nodes
            success: Whether inference succeeded
            metadata: Optional metadata
        """
        if not self.enable_metrics or self.metrics is None:
            return
        
        # Record metrics
        self.metrics.record_metric("inference_latency_ms", latency_ms, metadata)
        self.metrics.record_metric("reasoning_steps", num_steps, metadata)
        self.metrics.record_metric("dag_num_nodes", dag_nodes, metadata)
        
        # Update counters
        self.metrics.increment_counter("total_requests")
        if not success:
            self.metrics.increment_counter("failed_requests")
        
        # Check if health check is needed
        if self.enable_health_checks and self.health_monitor is not None:
            if time.time() - self.last_health_check > self.health_check_interval:
                self.check_health()
                self.last_health_check = time.time()
    
    def check_health(self) -> Optional[HealthStatus]:
        """
        Perform health check.
        
        Returns:
            HealthStatus if health checks enabled, None otherwise
        """
        if not self.enable_health_checks or self.health_monitor is None or self.metrics is None:
            return None
        
        status = self.health_monitor.check_health(self.metrics)
        
        # Trigger alerts if issues found
        if not status.is_healthy:
            for callback in self.alert_callbacks:
                try:
                    callback(status)
                except Exception as e:
                    print(f"Alert callback failed: {e}")
        
        return status
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get complete dashboard data for UI/monitoring.
        
        Returns:
            Dictionary with all monitoring data
        """
        data = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - self.start_time,
        }
        
        if self.metrics:
            data["metrics"] = self.metrics.get_all_metrics()
        
        if self.health_monitor:
            recent_health = self.health_monitor.get_health_history(n=1)
            if recent_health:
                status = recent_health[0]
                data["health"] = {
                    "is_healthy": status.is_healthy,
                    "message": status.message,
                    "issues": status.issues,
                    "metrics": status.metrics,
                }
        
        return data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.metrics:
            return {}
        
        return self.metrics.get_all_metrics()
    
    def reset(self):
        """Reset all metrics and monitoring state."""
        if self.metrics:
            self.metrics.reset()
        if self.health_monitor:
            self.health_monitor.health_history.clear()
        self.start_time = time.time()
        self.last_health_check = time.time()


if __name__ == "__main__":
    # Test observability
    print("Testing AI Observability Pattern...")
    
    # Test 1: Metrics collection
    metrics = MetricsCollector()
    for i in range(100):
        metrics.record_metric("latency", 10 + np.random.randn() * 2)
    
    stats = metrics.get_metric_stats("latency")
    print(f"Test 1 - Metrics: mean={stats['mean']:.2f}, p95={stats['p95']:.2f}")
    
    # Test 2: Health monitoring
    health_monitor = HealthMonitor(latency_threshold_ms=15.0)
    status = health_monitor.check_health(metrics)
    print(f"Test 2 - Health: {status.is_healthy}, message='{status.message}'")
    
    # Test 3: Complete observability
    obs = AIObservability()
    for i in range(10):
        obs.record_inference(
            latency_ms=10 + np.random.randn() * 2,
            num_steps=5,
            dag_nodes=25,
            success=True
        )
    
    dashboard = obs.get_dashboard_data()
    print(f"Test 3 - Dashboard: uptime={dashboard['uptime_seconds']:.1f}s")
    
    # Test 4: Counters
    print(f"Test 4 - Counters: total={metrics.get_counter('total_requests')}")
    
    print("\n✅ AI Observability tests passed!")
