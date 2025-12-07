# 100% Truthfulness & Validation

## Overview

This system enforces **100% truthfulness** in all operations. It is "merciless" in its accuracy - no lies, no hidden errors, no misleading information.

## Core Principles

### 1. **Sans Pitié (Merciless) - Strict Validation**
All inputs are validated strictly before processing:
- Type checking (e.g., must be list, must be string)
- Range validation (e.g., positive values, non-empty)
- Dimension checking (e.g., input size matches model configuration)
- NaN/Infinity detection in numerical inputs

**Example:**
```python
# ✅ Valid input is accepted
result = server.reason_step(x=[0.1] * 64, inner_recursions=3)

# ❌ Invalid input is rejected with clear error
result = server.reason_step(x=[0.1] * 32, inner_recursions=3)
# Returns: {"status": "error", "error": "VALIDATION ERROR: x dimension mismatch. Expected 64, got 32"}
```

### 2. **100% Vrais (100% True) - Truthful Reporting**
All outputs include truthfulness markers:
- `truthful_report`: true - indicates verified truthful output
- `output_verified`: true - indicates output was checked for validity
- `validation_failed`: true - explicitly marks validation failures
- `computation_failed`: true - explicitly marks computation errors

**Example:**
```python
result = server.evaluate_score(y_pred=[0.5]*32, y_target=[0.4]*32, metric="mse")
# Returns:
# {
#   "status": "success",
#   "score": 0.01,
#   "metric": "mse",
#   "truthful_report": true,
#   "score_verified": true
# }
```

### 3. **Ne Me Mentir (Don't Lie to Me) - Never Hide Errors**
The system never hides failures or errors:
- All exceptions are caught and reported
- Failed commands report actual exit codes
- Computation errors are reported with details
- NaN/Infinity results trigger explicit errors

**Example:**
```python
# Command that fails
result = server.execute_command("python -c 'import sys; sys.exit(1)'")
# Returns:
# {
#   "status": "error",  # Not "success"!
#   "return_code": 1,
#   "truthful_report": true
# }
```

## Validation Features

### System Tools

#### `execute_command`
- Validates command is string or list
- Validates command is not empty
- Validates timeout is positive
- Reports true exit status (error if return_code != 0)

#### `list_directory`
- Validates path is a string
- Distinguishes "not found" vs "not a directory"
- Reports exact error types

#### `get_environment_variable`
- Validates variable name is string and non-empty
- Distinguishes "not found" (None) vs "empty" ("")
- Includes `is_empty` flag for transparency

#### `check_command_exists`
- Validates command name is string and non-empty
- Returns exact path if found, None if not
- Never claims false existence

### AI Reasoning Tools

#### `reason_step`
- Validates input types (must be lists)
- Validates dimensions match model configuration
- Validates inner_recursions is positive
- Checks for NaN/Inf in inputs
- Checks for NaN/Inf in outputs
- Catches and reports computation errors

#### `evaluate_score`
- Validates metric name is valid
- Validates inputs are lists
- Validates dimensions match
- Validates arrays are not empty
- Checks for NaN/Inf in inputs
- Verifies computed score is finite

## Testing Truthfulness

Run the comprehensive truthfulness tests:

```bash
# Run all truthfulness validation tests
python tests/test_truthfulness_validation.py

# Run original MCP system tests
python tests/test_mcp_system.py
```

### Test Categories

1. **Input Validation Tests**
   - Invalid types
   - Empty values
   - Invalid ranges
   - Dimension mismatches

2. **Truthful Reporting Tests**
   - Command exit codes
   - Environment variable existence
   - Empty vs not found distinction
   - Count accuracy

3. **AI Validation Tests**
   - Input type checking
   - Dimension validation
   - Computation error handling
   - Output verification

4. **Metric Validation Tests**
   - Metric name validation
   - Dimension matching
   - Empty array detection
   - Score verification

## Error Message Format

All validation errors follow a consistent format:

```
VALIDATION ERROR: <specific description>
```

All computation errors follow:

```
COMPUTATION ERROR: <specific description>
OUTPUT ERROR: <specific description>
ENCODING ERROR: <specific description>
```

## Example Usage

```python
from mcp.server import TRLinkosMCPServer

server = TRLinkosMCPServer(x_dim=64, y_dim=32, z_dim=64)

# Example 1: Valid operation
result = server.evaluate_score(
    y_pred=[0.5] * 32,
    y_target=[0.4] * 32,
    metric="mse"
)
print(result["score"])  # 0.01
print(result["truthful_report"])  # True

# Example 2: Invalid operation - dimension mismatch
result = server.evaluate_score(
    y_pred=[0.5] * 32,
    y_target=[0.4] * 16,  # Wrong size!
    metric="mse"
)
print(result["status"])  # "error"
print(result["validation_failed"])  # True
print(result["error"])  # "VALIDATION ERROR: Dimension mismatch..."

# Example 3: Invalid metric
result = server.evaluate_score(
    y_pred=[0.5] * 32,
    y_target=[0.4] * 32,
    metric="invalid"
)
print(result["status"])  # "error"
print(result["error"])  # "VALIDATION ERROR: Unknown metric 'invalid'..."
```

## Summary

The system is **sans pitié** (merciless) in enforcing truthfulness:

✅ **All inputs validated strictly**
- Type checking
- Range checking
- Dimension checking
- NaN/Inf detection

✅ **All outputs verified**
- Truthful reporting markers
- Output integrity checks
- No silent failures

✅ **All errors reported clearly**
- Explicit error categories
- Detailed error messages
- No hidden failures

✅ **100% accurate metrics**
- Validated computation
- Verified results
- No approximations or lies

This ensures the system is **100% vrais** (100% true) and will **ne me mentir** (never lie to you).
