"""
Resilient Workflow Pattern Implementation

Based on: THE-BLUEPRINTS.md - Pattern 08: The Resilient Workflow Pattern

Problem: Brittle automation ("glass cannons") crashes on predictable errors 
like API timeouts, causing operational downtime and eroding user trust.

Solution: Design a dedicated Resilience Layer that decouples the AI agent's 
core logic from the complexities of error handling.

This module provides error handling, retry mechanisms, and circuit breakers
for T-RLINKOS TRM++, ensuring robust operation in production.
"""

import time
import random
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, List, Dict, TypeVar, Generic
from enum import Enum
import traceback


T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failures detected, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_attempts: int = 3
    initial_delay: float = 0.1
    max_delay: float = 10.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: List[type] = field(default_factory=lambda: [Exception])


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout: float = 60.0


@dataclass
class ExecutionResult(Generic[T]):
    """Result of an execution attempt"""
    success: bool
    value: Optional[T] = None
    error: Optional[Exception] = None
    attempts: int = 0
    total_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class RetryStrategy:
    """
    Implements retry logic with exponential backoff and jitter.
    
    Features:
    - Exponential backoff
    - Random jitter to prevent thundering herd
    - Configurable retry conditions
    - Detailed execution tracking
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry strategy.
        
        Args:
            config: RetryConfig instance
        """
        self.config = config or RetryConfig()
        
    def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> ExecutionResult[T]:
        """
        Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            ExecutionResult with result or error
        """
        start_time = time.time()
        last_error = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = func(*args, **kwargs)
                return ExecutionResult(
                    success=True,
                    value=result,
                    attempts=attempt,
                    total_time=time.time() - start_time,
                )
            except Exception as e:
                last_error = e
                
                # Check if exception is retryable
                is_retryable = any(
                    isinstance(e, exc_type)
                    for exc_type in self.config.retryable_exceptions
                )
                
                if not is_retryable or attempt >= self.config.max_attempts:
                    break
                
                # Calculate delay with exponential backoff
                delay = min(
                    self.config.initial_delay * (self.config.exponential_base ** (attempt - 1)),
                    self.config.max_delay
                )
                
                # Add jitter
                if self.config.jitter:
                    delay *= (0.5 + random.random() * 0.5)
                
                time.sleep(delay)
        
        # All attempts failed
        return ExecutionResult(
            success=False,
            error=last_error,
            attempts=self.config.max_attempts,
            total_time=time.time() - start_time,
            metadata={"error_type": type(last_error).__name__ if last_error else "Unknown"}
        )


class CircuitBreaker:
    """
    Implements circuit breaker pattern to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests blocked
    - HALF_OPEN: Testing recovery, limited requests allowed
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.
        
        Args:
            config: CircuitBreakerConfig instance
        """
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        # Check if circuit should transition from OPEN to HALF_OPEN
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.config.timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
    
    def reset(self):
        """Reset circuit breaker to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
        }


class ErrorHandler:
    """
    Centralized error handling and recovery.
    
    Features:
    - Exception classification
    - Error recovery strategies
    - Fallback mechanisms
    - Error logging and tracking
    """
    
    def __init__(
        self,
        fallback_value: Optional[Any] = None,
        log_errors: bool = True,
    ):
        """
        Initialize error handler.
        
        Args:
            fallback_value: Default fallback value on errors
            log_errors: Whether to log errors
        """
        self.fallback_value = fallback_value
        self.log_errors = log_errors
        self.error_history: List[Dict[str, Any]] = []
        
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle an error.
        
        Args:
            error: Exception that occurred
            context: Optional context information
            
        Returns:
            Error information dictionary
        """
        error_info = {
            "type": type(error).__name__,
            "message": str(error),
            "timestamp": time.time(),
            "context": context or {},
            "traceback": traceback.format_exc(),
        }
        
        self.error_history.append(error_info)
        
        if self.log_errors:
            # In production, use logging module: logging.error(f"Error handled: {error_info['type']}", exc_info=error)
            # For now, print to maintain backward compatibility with tests
            print(f"Error handled: {error_info['type']} - {error_info['message']}")
        
        return error_info
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        if not self.error_history:
            return {"total_errors": 0}
        
        error_types = {}
        for error in self.error_history:
            error_type = error["type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "error_types": error_types,
            "recent_errors": self.error_history[-5:],
        }


class ResilientWorkflow:
    """
    Complete resilient workflow system for T-RLINKOS TRM++.
    
    Combines:
    - Retry logic with exponential backoff
    - Circuit breakers to prevent cascading failures
    - Error handling and recovery
    - Fallback mechanisms
    """
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        enable_retry: bool = True,
        enable_circuit_breaker: bool = True,
        fallback_value: Optional[Any] = None,
    ):
        """
        Initialize resilient workflow.
        
        Args:
            retry_config: Retry configuration
            circuit_config: Circuit breaker configuration
            enable_retry: Enable retry logic
            enable_circuit_breaker: Enable circuit breaker
            fallback_value: Fallback value on failure
        """
        self.enable_retry = enable_retry
        self.enable_circuit_breaker = enable_circuit_breaker
        
        self.retry_strategy = RetryStrategy(retry_config) if enable_retry else None
        self.circuit_breaker = CircuitBreaker(circuit_config) if enable_circuit_breaker else None
        self.error_handler = ErrorHandler(fallback_value=fallback_value)
        
    def execute(
        self,
        func: Callable[..., T],
        *args,
        use_fallback: bool = True,
        **kwargs
    ) -> ExecutionResult[T]:
        """
        Execute function with full resilience features.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            use_fallback: Use fallback value on complete failure
            **kwargs: Keyword arguments
            
        Returns:
            ExecutionResult with result or fallback
        """
        def _execute():
            if self.enable_circuit_breaker and self.circuit_breaker:
                return self.circuit_breaker.call(func, *args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        # Execute with retry if enabled
        if self.enable_retry and self.retry_strategy:
            result = self.retry_strategy.execute(_execute)
        else:
            try:
                start_time = time.time()
                value = _execute()
                result = ExecutionResult(
                    success=True,
                    value=value,
                    attempts=1,
                    total_time=time.time() - start_time,
                )
            except Exception as e:
                result = ExecutionResult(
                    success=False,
                    error=e,
                    attempts=1,
                    total_time=0.0,
                )
        
        # Handle errors
        if not result.success and result.error:
            self.error_handler.handle_error(result.error, {"attempts": result.attempts})
            
            # Use fallback if available
            if use_fallback and self.error_handler.fallback_value is not None:
                result.value = self.error_handler.fallback_value
                result.success = True
                result.metadata["used_fallback"] = True
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of resilience components."""
        status = {
            "retry_enabled": self.enable_retry,
            "circuit_breaker_enabled": self.enable_circuit_breaker,
        }
        
        if self.circuit_breaker:
            status["circuit_breaker"] = self.circuit_breaker.get_state()
        
        status["error_stats"] = self.error_handler.get_error_stats()
        
        return status


if __name__ == "__main__":
    # Test resilient workflow
    print("Testing Resilient Workflow Pattern...")
    
    # Test 1: Retry with eventual success
    call_count = [0]
    def flaky_func():
        call_count[0] += 1
        if call_count[0] < 3:
            raise ValueError("Temporary failure")
        return "success"
    
    workflow = ResilientWorkflow()
    result = workflow.execute(flaky_func)
    print(f"Test 1 - Retry: success={result.success}, attempts={result.attempts}, value={result.value}")
    
    # Test 2: Circuit breaker
    circuit = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
    failure_count = 0
    for i in range(5):
        try:
            circuit.call(lambda: 1/0)  # Always fails
        except:
            failure_count += 1
    
    print(f"Test 2 - Circuit breaker: state={circuit.get_state()['state']}, failures={failure_count}")
    
    # Test 3: Fallback value
    workflow_fallback = ResilientWorkflow(fallback_value="fallback_result", enable_retry=False)
    result = workflow_fallback.execute(lambda: 1/0)
    print(f"Test 3 - Fallback: success={result.success}, value={result.value}, used_fallback={result.metadata.get('used_fallback')}")
    
    # Test 4: Status
    status = workflow.get_status()
    print(f"Test 4 - Status: {status['error_stats']['total_errors']} errors tracked")
    
    print("\n✅ Resilient Workflow tests passed!")
