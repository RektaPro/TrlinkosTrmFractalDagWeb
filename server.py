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
        _start_mcp_server(args)
    else:
        # Start FastAPI server
        _start_fastapi_server(args)


def _start_mcp_server(args: argparse.Namespace) -> None:
    """Start the MCP server with the given arguments."""
    from mcp.server import TRLinkosMCPServer
    import asyncio

    server = TRLinkosMCPServer(
        x_dim=args.x_dim,
        y_dim=args.y_dim,
        z_dim=args.z_dim,
    )

    if args.http:
        # HTTP mode using FastAPI
        try:
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
            from typing import Any, Dict
            import uvicorn
            from pathlib import Path
            import json

            http_app = FastAPI(title="T-RLINKOS MCP Server")

            class ToolCallRequest(BaseModel):
                name: str
                arguments: Dict[str, Any] = {}

            @http_app.post("/tools/call")
            async def call_tool(request: ToolCallRequest):
                result = server.handle_tool_call(request.name, request.arguments)
                if "error" in result:
                    raise HTTPException(status_code=400, detail=result["error"])
                return result

            @http_app.get("/tools/list")
            async def list_tools():
                manifest_path = Path(__file__).parent / "mcp.json"
                with open(manifest_path) as f:
                    manifest = json.load(f)
                return {"tools": manifest.get("tools", [])}

            @http_app.get("/resources/{resource_id:path}")
            async def read_resource(resource_id: str):
                uri = f"trlinkos://{resource_id}"
                return server.handle_resource_read(uri)

            print(f"MCP HTTP Server running on http://0.0.0.0:{args.mcp_port}")
            uvicorn.run(http_app, host="0.0.0.0", port=args.mcp_port)

        except ImportError:
            print("FastAPI not installed. Install with: pip install fastapi uvicorn")
            sys.exit(1)
    else:
        # Stdio mode
        from mcp.server import handle_stdio
        print("MCP Server running on stdio...")
        asyncio.run(handle_stdio(server))


def _start_fastapi_server(args: argparse.Namespace) -> None:
    """Start the FastAPI server with the given arguments."""
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
