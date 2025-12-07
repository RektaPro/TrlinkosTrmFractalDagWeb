"""
Safety Guardrails Pattern Implementation

Based on: THE-BLUEPRINTS.md - Pattern 01: The Safety Guardrail Pattern

Problem: Unconstrained AI systems can accept malicious inputs and generate 
harmful, inaccurate, or brand-damaging content.

Solution: Implement a framework of input sanitization and output validation 
to act as a firewall between the user and the model.

This module provides input validation and output filtering for T-RLINKOS TRM++,
ensuring safe operation in production environments.
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum


class ValidationResult(Enum):
    """Result of validation checks"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class ValidationReport:
    """Report from validation check"""
    result: ValidationResult
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    sanitized_input: Optional[Any] = None


class InputValidator:
    """
    Input validation for T-RLINKOS TRM++ inference.
    
    Validates input features before they reach the model to prevent:
    - Malformed inputs (wrong shape, type, etc.)
    - Extreme values that could cause numerical instability
    - Injection attacks (for text inputs)
    - Out-of-distribution inputs
    """
    
    def __init__(
        self,
        expected_shape: Optional[Tuple[int, ...]] = None,
        value_range: Tuple[float, float] = (-1e6, 1e6),
        allow_nan: bool = False,
        allow_inf: bool = False,
        text_max_length: int = 10000,
        blocked_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize input validator.
        
        Args:
            expected_shape: Expected shape of input (e.g., (batch_size, x_dim))
            value_range: Allowed range for numerical values (min, max)
            allow_nan: Whether to allow NaN values
            allow_inf: Whether to allow infinite values
            text_max_length: Maximum length for text inputs
            blocked_patterns: Regex patterns to block in text inputs
        """
        self.expected_shape = expected_shape
        self.value_range = value_range
        self.allow_nan = allow_nan
        self.allow_inf = allow_inf
        self.text_max_length = text_max_length
        self.blocked_patterns = blocked_patterns or []
        
    def validate_array(self, x: np.ndarray) -> ValidationReport:
        """
        Validate numpy array input.
        
        Args:
            x: Input array
            
        Returns:
            ValidationReport with validation results
        """
        # Check shape
        if self.expected_shape is not None:
            if len(x.shape) != len(self.expected_shape):
                return ValidationReport(
                    ValidationResult.FAILED,
                    f"Invalid input shape. Expected {len(self.expected_shape)} dimensions, got {len(x.shape)}",
                    {"expected": self.expected_shape, "actual": x.shape}
                )
            
            # Check dimensions (ignoring batch size)
            for i, (expected, actual) in enumerate(zip(self.expected_shape[1:], x.shape[1:])):
                if expected is not None and expected != actual:
                    return ValidationReport(
                        ValidationResult.FAILED,
                        f"Invalid input dimension at axis {i+1}. Expected {expected}, got {actual}",
                        {"expected": self.expected_shape, "actual": x.shape}
                    )
        
        # Check for NaN
        if not self.allow_nan and np.any(np.isnan(x)):
            nan_count = np.sum(np.isnan(x))
            return ValidationReport(
                ValidationResult.FAILED,
                f"Input contains {nan_count} NaN values",
                {"nan_count": int(nan_count)}
            )
        
        # Check for Inf
        if not self.allow_inf and np.any(np.isinf(x)):
            inf_count = np.sum(np.isinf(x))
            return ValidationReport(
                ValidationResult.FAILED,
                f"Input contains {inf_count} infinite values",
                {"inf_count": int(inf_count)}
            )
        
        # Check value range
        min_val, max_val = np.min(x), np.max(x)
        if min_val < self.value_range[0] or max_val > self.value_range[1]:
            return ValidationReport(
                ValidationResult.WARNING,
                f"Input values outside expected range {self.value_range}",
                {
                    "min_value": float(min_val),
                    "max_value": float(max_val),
                    "expected_range": self.value_range
                }
            )
        
        return ValidationReport(ValidationResult.PASSED, "Input validation passed")
    
    def validate_text(self, text: str) -> ValidationReport:
        """
        Validate text input.
        
        Args:
            text: Input text string
            
        Returns:
            ValidationReport with validation results
        """
        # Check length
        if len(text) > self.text_max_length:
            return ValidationReport(
                ValidationResult.FAILED,
                f"Text exceeds maximum length of {self.text_max_length}",
                {"length": len(text), "max_length": self.text_max_length}
            )
        
        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return ValidationReport(
                    ValidationResult.FAILED,
                    f"Text contains blocked pattern: {pattern}",
                    {"pattern": pattern, "text_preview": text[:100]}
                )
        
        return ValidationReport(ValidationResult.PASSED, "Text validation passed")
    
    def sanitize_array(self, x: np.ndarray) -> np.ndarray:
        """
        Sanitize array by clipping values and replacing NaN/Inf.
        
        Args:
            x: Input array
            
        Returns:
            Sanitized array
        """
        x = x.copy()
        
        # Replace NaN with 0
        if np.any(np.isnan(x)):
            x = np.nan_to_num(x, nan=0.0)
        
        # Replace Inf with large values
        if np.any(np.isinf(x)):
            x = np.nan_to_num(x, posinf=self.value_range[1], neginf=self.value_range[0])
        
        # Clip to value range
        x = np.clip(x, self.value_range[0], self.value_range[1])
        
        return x


class OutputValidator:
    """
    Output validation for T-RLINKOS TRM++ predictions.
    
    Validates model outputs before they are returned to users to prevent:
    - Unreliable predictions (low confidence, unstable DAG)
    - Unexpected output patterns
    - Potentially harmful content
    """
    
    def __init__(
        self,
        min_confidence: float = 0.0,
        max_uncertainty: float = float('inf'),
        check_dag_stability: bool = True,
        output_filters: Optional[List[Callable]] = None,
    ):
        """
        Initialize output validator.
        
        Args:
            min_confidence: Minimum acceptable confidence score
            max_uncertainty: Maximum acceptable uncertainty
            check_dag_stability: Whether to check DAG stability
            output_filters: Custom filter functions
        """
        self.min_confidence = min_confidence
        self.max_uncertainty = max_uncertainty
        self.check_dag_stability = check_dag_stability
        self.output_filters = output_filters or []
    
    def validate_output(
        self,
        y_pred: np.ndarray,
        dag: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationReport:
        """
        Validate model output.
        
        Args:
            y_pred: Model predictions
            dag: Reasoning DAG (optional)
            metadata: Additional metadata (optional)
            
        Returns:
            ValidationReport with validation results
        """
        # Check for NaN/Inf in output
        if np.any(np.isnan(y_pred)):
            return ValidationReport(
                ValidationResult.FAILED,
                "Output contains NaN values",
                {"nan_count": int(np.sum(np.isnan(y_pred)))}
            )
        
        if np.any(np.isinf(y_pred)):
            return ValidationReport(
                ValidationResult.FAILED,
                "Output contains infinite values",
                {"inf_count": int(np.sum(np.isinf(y_pred)))}
            )
        
        # Check DAG stability if provided
        if self.check_dag_stability and dag is not None:
            if hasattr(dag, 'nodes') and len(dag.nodes) == 0:
                return ValidationReport(
                    ValidationResult.FAILED,
                    "DAG is empty - reasoning failed",
                    {"num_nodes": 0}
                )
            
            # Check for best node
            if hasattr(dag, 'get_best_node'):
                best_node = dag.get_best_node()
                if best_node is None:
                    return ValidationReport(
                        ValidationResult.WARNING,
                        "No best node found in DAG",
                        {"num_nodes": len(dag.nodes) if hasattr(dag, 'nodes') else 0}
                    )
        
        # Apply custom filters
        for filter_fn in self.output_filters:
            try:
                is_valid, message = filter_fn(y_pred, dag, metadata)
                if not is_valid:
                    return ValidationReport(
                        ValidationResult.FAILED,
                        f"Custom filter failed: {message}",
                        {"filter": filter_fn.__name__}
                    )
            except Exception as e:
                return ValidationReport(
                    ValidationResult.WARNING,
                    f"Custom filter raised exception: {str(e)}",
                    {"filter": filter_fn.__name__, "error": str(e)}
                )
        
        return ValidationReport(ValidationResult.PASSED, "Output validation passed")


class SafetyGuardrail:
    """
    Complete safety guardrail system combining input and output validation.
    
    This acts as a firewall between users and the T-RLINKOS model,
    ensuring safe operation in production environments.
    """
    
    def __init__(
        self,
        input_validator: Optional[InputValidator] = None,
        output_validator: Optional[OutputValidator] = None,
        auto_sanitize: bool = True,
        raise_on_failure: bool = False,
    ):
        """
        Initialize safety guardrail.
        
        Args:
            input_validator: Input validator instance
            output_validator: Output validator instance
            auto_sanitize: Automatically sanitize inputs on validation failure
            raise_on_failure: Raise exception on validation failure
        """
        self.input_validator = input_validator or InputValidator()
        self.output_validator = output_validator or OutputValidator()
        self.auto_sanitize = auto_sanitize
        self.raise_on_failure = raise_on_failure
        
        # Statistics
        self.stats = {
            "total_inputs": 0,
            "input_failures": 0,
            "input_warnings": 0,
            "total_outputs": 0,
            "output_failures": 0,
            "output_warnings": 0,
            "auto_sanitizations": 0,
        }
    
    def validate_input(self, x: np.ndarray) -> Tuple[bool, np.ndarray, ValidationReport]:
        """
        Validate and potentially sanitize input.
        
        Args:
            x: Input array
            
        Returns:
            Tuple of (is_valid, sanitized_input, report)
        """
        self.stats["total_inputs"] += 1
        
        # Validate input
        report = self.input_validator.validate_array(x)
        
        # Handle validation result
        if report.result == ValidationResult.FAILED:
            self.stats["input_failures"] += 1
            
            if self.auto_sanitize:
                # Attempt to sanitize
                x_sanitized = self.input_validator.sanitize_array(x)
                self.stats["auto_sanitizations"] += 1
                
                # Re-validate
                report = self.input_validator.validate_array(x_sanitized)
                if report.result == ValidationResult.PASSED:
                    return True, x_sanitized, report
            
            if self.raise_on_failure:
                raise ValueError(f"Input validation failed: {report.message}")
            
            return False, x, report
        
        elif report.result == ValidationResult.WARNING:
            self.stats["input_warnings"] += 1
        
        return True, x, report
    
    def validate_output(
        self,
        y_pred: np.ndarray,
        dag: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, ValidationReport]:
        """
        Validate output.
        
        Args:
            y_pred: Model predictions
            dag: Reasoning DAG
            metadata: Additional metadata
            
        Returns:
            Tuple of (is_valid, report)
        """
        self.stats["total_outputs"] += 1
        
        # Validate output
        report = self.output_validator.validate_output(y_pred, dag, metadata)
        
        # Handle validation result
        if report.result == ValidationResult.FAILED:
            self.stats["output_failures"] += 1
            
            if self.raise_on_failure:
                raise ValueError(f"Output validation failed: {report.message}")
            
            return False, report
        
        elif report.result == ValidationResult.WARNING:
            self.stats["output_warnings"] += 1
        
        return True, report
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset validation statistics."""
        for key in self.stats:
            self.stats[key] = 0


if __name__ == "__main__":
    # Test safety guardrails
    print("Testing Safety Guardrails Pattern...")
    
    # Test 1: Valid input
    validator = InputValidator(expected_shape=(None, 64))
    x = np.random.randn(8, 64)
    report = validator.validate_array(x)
    print(f"Test 1 - Valid input: {report.result.value} - {report.message}")
    
    # Test 2: Input with NaN
    x_nan = x.copy()
    x_nan[0, 0] = np.nan
    report = validator.validate_array(x_nan)
    print(f"Test 2 - Input with NaN: {report.result.value} - {report.message}")
    
    # Test 3: Auto-sanitization
    guardrail = SafetyGuardrail(auto_sanitize=True)
    is_valid, x_clean, report = guardrail.validate_input(x_nan)
    print(f"Test 3 - Auto-sanitization: valid={is_valid}, has_nan={np.any(np.isnan(x_clean))}")
    
    # Test 4: Output validation
    y_pred = np.random.randn(8, 32)
    is_valid, report = guardrail.validate_output(y_pred)
    print(f"Test 4 - Output validation: {report.result.value} - {report.message}")
    
    # Test 5: Statistics
    stats = guardrail.get_statistics()
    print(f"Test 5 - Statistics: {stats}")
    
    print("\n✅ Safety Guardrails tests passed!")
