.PHONY: help install install-dev test test-cov lint format clean pre-commit-install pre-commit-run

help:
	@echo "T-RLINKOS TRM++ Fractal DAG - Development Commands"
	@echo ""
	@echo "  install              Install core dependencies"
	@echo "  install-dev          Install development dependencies"
	@echo "  test                 Run tests"
	@echo "  test-cov             Run tests with coverage report"
	@echo "  lint                 Run all linters (flake8, black, isort)"
	@echo "  format               Auto-format code with black and isort"
	@echo "  clean                Remove build artifacts and cache"
	@echo "  pre-commit-install   Install pre-commit hooks"
	@echo "  pre-commit-run       Run pre-commit on all files"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=. --cov-report=xml --cov-report=term --cov-report=html

lint:
	@echo "Running flake8..."
	flake8 .
	@echo "Running black check..."
	black --check .
	@echo "Running isort check..."
	isort --check-only .

format:
	@echo "Running black..."
	black .
	@echo "Running isort..."
	isort .

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build dist *.egg-info
	rm -rf htmlcov .coverage coverage.xml
	rm -rf .pytest_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

pre-commit-install:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files
