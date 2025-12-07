"""MCP package for T-RLINKOS TRM++.

This package provides Model Context Protocol (MCP) integration for T-RLINKOS,
enabling seamless integration with LLMs and AI agents through a standardized
tool-based interface.
"""

from .server import TRLinkosMCPServer

__all__ = ["TRLinkosMCPServer"]
__version__ = "1.0.0"
