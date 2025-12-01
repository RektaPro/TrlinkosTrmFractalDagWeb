#!/usr/bin/env python3
"""
T-RLINKOS TRM++ System Server.

This is the main entry point to launch the T-RLINKOS system.
It provides a unified server interface that can run:
- FastAPI REST API (default)
- MCP Server (Model Context Protocol)

Usage:
    # Start FastAPI server (default)
    python server.py

    # Start with custom port
    python server.py --port 8000

    # Start MCP server (stdio mode for LLM integration)
    python server.py --mcp

    # Start MCP server (HTTP mode)
    python server.py --mcp --http --mcp-port 8080

Example:
    # Check health
    curl http://localhost:8000/health

    # Run reasoning
    curl -X POST http://localhost:8000/reason \\
        -H "Content-Type: application/json" \\
        -d '{"features": [0.1, 0.2, ...]}'  # 64 float values
"""

import argparse
import sys


def main() -> None:
    """Main entry point for the T-RLINKOS server."""
    parser = argparse.ArgumentParser(
        description="T-RLINKOS TRM++ System Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Server mode
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Start MCP server instead of FastAPI",
    )

    # FastAPI options
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for FastAPI server (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    # MCP options
    parser.add_argument(
        "--http",
        action="store_true",
        help="Use HTTP transport for MCP (default: stdio)",
    )
    parser.add_argument(
        "--mcp-port",
        type=int,
        default=8080,
        help="Port for MCP HTTP server (default: 8080)",
    )

    # Model configuration
    parser.add_argument(
        "--x-dim",
        type=int,
        default=64,
        help="Input dimension (default: 64)",
    )
    parser.add_argument(
        "--y-dim",
        type=int,
        default=32,
        help="Output dimension (default: 32)",
    )
    parser.add_argument(
        "--z-dim",
        type=int,
        default=64,
        help="Internal state dimension (default: 64)",
    )

    args = parser.parse_args()

    if args.mcp:
        # Start MCP server
        print("Starting T-RLINKOS MCP Server...")
        from mcp.server import TRLinkosMCPServer, handle_stdio, main as mcp_main

        # Build MCP arguments
        mcp_args = [
            f"--x-dim={args.x_dim}",
            f"--y-dim={args.y_dim}",
            f"--z-dim={args.z_dim}",
        ]

        if args.http:
            mcp_args.append("--http")
            mcp_args.append(f"--port={args.mcp_port}")
        else:
            mcp_args.append("--stdio")

        # Override sys.argv for MCP server
        sys.argv = ["mcp/server.py"] + mcp_args
        mcp_main()
    else:
        # Start FastAPI server
        print(f"Starting T-RLINKOS FastAPI Server on {args.host}:{args.port}...")
        print(f"Model config: x_dim={args.x_dim}, y_dim={args.y_dim}, z_dim={args.z_dim}")
        print(f"API docs available at: http://{args.host}:{args.port}/docs")

        try:
            import uvicorn
            from api import app

            uvicorn.run(
                "api:app" if args.reload else app,
                host=args.host,
                port=args.port,
                reload=args.reload,
            )
        except ImportError as e:
            print(f"Error: {e}")
            print("Install FastAPI and uvicorn with: pip install fastapi uvicorn")
            sys.exit(1)


if __name__ == "__main__":
    main()
