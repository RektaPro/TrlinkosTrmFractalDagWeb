"""Benchmarks package for T-RLINKOS TRM++.

This package provides comprehensive benchmarking tools for measuring
T-RLINKOS performance across different configurations and workloads.
"""

from .formal_benchmarks import BenchmarkSuite, BenchmarkResult

__all__ = ["BenchmarkSuite", "BenchmarkResult"]
__version__ = "1.0.0"
