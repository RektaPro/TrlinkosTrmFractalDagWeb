# CI/CD Pipeline Documentation

## Overview

This project uses GitHub Actions for continuous integration and deployment, with comprehensive code quality and testing automation.

## GitHub Actions Workflows

### CI Workflow (`.github/workflows/ci.yml`)

Runs on every push and pull request to `main` and `develop` branches.

#### Jobs

1. **Lint** - Code quality checks
   - Black (code formatter)
   - isort (import sorter)
   - Flake8 (linter)
   - Runs on: Python 3.11

2. **Test** - Test suite across multiple Python versions
   - Python versions: 3.8, 3.9, 3.10, 3.11, 3.12
   - Coverage reporting via Codecov
   - Parallel execution with pytest-xdist
   - HTML coverage reports as artifacts

3. **Test Optional Features** - Tests with all dependencies
   - PyTorch integration
   - HuggingFace transformers
   - ONNX export
   - Runs on: Python 3.11

4. **Security** - Security scanning
   - Bandit (security linter)
   - Safety (dependency vulnerability scanner)
   - Reports uploaded as artifacts

## Pre-commit Hooks

Pre-commit hooks run automatically before each commit to ensure code quality.

### Installation

```bash
pip install pre-commit
pre-commit install
```

### Configured Hooks

- **File checks**: Trailing whitespace, end-of-file, merge conflicts, large files
- **Code formatting**: Black (100 char line length)
- **Import sorting**: isort (black profile)
- **Linting**: Flake8
- **Security**: Bandit (medium-low severity)
- **Syntax checks**: YAML, JSON

### Manual Execution

```bash
# Run on all files
pre-commit run --all-files

# Run on staged files only
pre-commit run
```

## Code Quality Tools

### Black - Code Formatter

Configuration in `pyproject.toml`:
- Line length: 100 characters
- Target: Python 3.8+

```bash
# Check formatting
black --check .

# Auto-format
black .
```

### isort - Import Sorter

Configuration in `pyproject.toml`:
- Profile: black
- Line length: 100

```bash
# Check imports
isort --check-only .

# Auto-sort
isort .
```

### Flake8 - Linter

Configuration in `.flake8`:
- Max line length: 100
- Ignore: E203, E501, W503, E402
- Max complexity: 15

```bash
# Run linter
flake8 .
```

### Coverage - Code Coverage

Configuration in `.coveragerc` and `pyproject.toml`:
- Source: Current directory
- Branch coverage: Enabled
- Formats: XML, HTML, Terminal

```bash
# Run tests with coverage
pytest tests/ --cov=. --cov-report=html --cov-report=term

# View HTML report
open htmlcov/index.html
```

## Make Commands

Convenient shortcuts for development tasks:

```bash
make help                 # Show all commands
make install              # Install core dependencies
make install-dev          # Install development dependencies
make test                 # Run tests (core only)
make test-cov             # Run tests with coverage
make test-all             # Run complete test suite
make lint                 # Run all linters
make format               # Auto-format code
make clean                # Remove build artifacts
make pre-commit-install   # Install pre-commit hooks
make pre-commit-run       # Run pre-commit on all files
```

## CI/CD Status Badges

Add these badges to your README:

```markdown
[![CI](https://github.com/RektaPro/TrlinkosTrmFractalDagWeb/workflows/CI/badge.svg)](https://github.com/RektaPro/TrlinkosTrmFractalDagWeb/actions)
[![codecov](https://codecov.io/gh/RektaPro/TrlinkosTrmFractalDagWeb/branch/main/graph/badge.svg)](https://codecov.io/gh/RektaPro/TrlinkosTrmFractalDagWeb)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
```

## Coverage Reporting

Coverage reports are automatically generated and uploaded to Codecov on every push.

### Local Coverage

```bash
# Generate coverage report
make test-cov

# View HTML report
open htmlcov/index.html
```

### CI Coverage

- Coverage reports are uploaded to Codecov
- HTML reports are available as GitHub Actions artifacts
- Coverage badge is displayed in README

## Troubleshooting

### Pre-commit Hook Failures

If pre-commit hooks fail:

1. Review the error messages
2. Fix the issues manually or run `make format`
3. Stage the changes and try committing again

### Test Failures

If tests fail in CI:

1. Check the GitHub Actions logs
2. Reproduce locally: `make test`
3. Fix the failing tests
4. Ensure all tests pass before pushing

### Optional Dependencies

Some tests require optional dependencies (torch, transformers):

```bash
# Install all dependencies
pip install torch transformers onnx onnxruntime

# Run complete test suite
make test-all
# or
python run_all_tests.py
```

## Best Practices

1. **Always run pre-commit hooks** before committing
2. **Run tests locally** before pushing
3. **Check coverage reports** to ensure new code is tested
4. **Review CI logs** if the pipeline fails
5. **Keep dependencies updated** regularly
6. **Format code** with `make format` before committing

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Pre-commit Documentation](https://pre-commit.com/)
- [Black Documentation](https://black.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Codecov Documentation](https://docs.codecov.com/)
