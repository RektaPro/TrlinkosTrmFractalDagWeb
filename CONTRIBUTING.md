# Contributing to T-RLINKOS TRM++ Fractal DAG

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/RektaPro/TrlinkosTrmFractalDagWeb.git
cd TrlinkosTrmFractalDagWeb
```

2. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

## Development Workflow

### Running Tests

Run all tests:
```bash
make test
# or
pytest tests/ -v
```

Run tests with coverage:
```bash
make test-cov
# or
pytest tests/ -v --cov=. --cov-report=xml --cov-report=term --cov-report=html
```

Run the complete test suite (including optional features):
```bash
python run_all_tests.py
```

### Code Formatting and Linting

We use several tools to maintain code quality:

- **Black**: Code formatter
- **isort**: Import sorter
- **Flake8**: Linter
- **Bandit**: Security linter

#### Auto-format code:
```bash
make format
# or
black .
isort .
```

#### Check code style:
```bash
make lint
# or
flake8 .
black --check .
isort --check-only .
```

### Pre-commit Hooks

Pre-commit hooks automatically run before each commit to ensure code quality:

```bash
# Install hooks (one-time setup)
make pre-commit-install
# or
pre-commit install

# Run hooks manually on all files
make pre-commit-run
# or
pre-commit run --all-files
```

The hooks will:
- Remove trailing whitespace
- Fix end-of-file issues
- Check YAML and JSON syntax
- Format code with Black
- Sort imports with isort
- Run Flake8 linting
- Check for security issues with Bandit

### Code Style Guidelines

- **Line length**: Maximum 100 characters
- **Imports**: Use absolute imports, sorted by isort
- **Docstrings**: Use numpy-style docstrings for functions and classes
- **Type hints**: Encouraged but not required
- **Comments**: Use sparingly, prefer self-documenting code

Example:
```python
import numpy as np
from typing import Optional, Tuple


def compute_reasoning_trace(
    input_data: np.ndarray,
    max_steps: int = 10,
    threshold: float = 0.01
) -> Tuple[np.ndarray, dict]:
    """
    Compute recursive reasoning trace using T-RLINKOS TRM++.

    Parameters
    ----------
    input_data : np.ndarray
        Input data array of shape (batch_size, input_dim)
    max_steps : int, optional
        Maximum reasoning steps, by default 10
    threshold : float, optional
        Convergence threshold, by default 0.01

    Returns
    -------
    Tuple[np.ndarray, dict]
        Output predictions and metadata dictionary
    """
    # Implementation here
    pass
```

## Continuous Integration

The project uses GitHub Actions for CI/CD:

### CI Pipeline

The CI pipeline runs on every push and pull request:

1. **Linting**: Checks code style with Black, isort, and Flake8
2. **Testing**: Runs tests on Python 3.8, 3.9, 3.10, 3.11, and 3.12
3. **Coverage**: Generates code coverage reports
4. **Security**: Runs Bandit and Safety checks
5. **Optional Features**: Tests with PyTorch, HuggingFace, ONNX

### Coverage Reports

Code coverage is automatically tracked and reported:
- Coverage reports are uploaded to Codecov
- HTML reports are available as GitHub Actions artifacts
- Minimum coverage threshold: No strict minimum, but aim for >80%

## Pull Request Guidelines

1. **Branch naming**: Use descriptive names (e.g., `feature/add-reasoning-layer`, `fix/dag-memory-leak`)
2. **Commit messages**: Use clear, concise commit messages
3. **Tests**: Add tests for new features and bug fixes
4. **Documentation**: Update documentation for user-facing changes
5. **Pre-commit checks**: Ensure all pre-commit hooks pass
6. **CI checks**: Ensure all CI checks pass before requesting review

### Pull Request Checklist

- [ ] Code follows the project's style guidelines
- [ ] All tests pass locally
- [ ] New code has test coverage
- [ ] Documentation is updated (if needed)
- [ ] Pre-commit hooks pass
- [ ] CI checks pass
- [ ] No security vulnerabilities introduced

## Issue Reporting

When reporting issues, please include:

1. **Description**: Clear description of the issue
2. **Reproduction steps**: How to reproduce the issue
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**: Python version, OS, dependencies
6. **Logs/Errors**: Relevant error messages or logs

## Questions?

For questions or discussions:
- Open an issue on GitHub
- Check existing documentation in the repository
- Review the README.md and other documentation files

## License

By contributing, you agree that your contributions will be licensed under the BSD-3-Clause License.
