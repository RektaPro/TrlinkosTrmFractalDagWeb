#!/usr/bin/env python3
"""
MCP Server for T-RLINKOS TRM++ Fractal Merkle-DAG.

This server implements the Model Context Protocol (MCP) to expose
T-RLINKOS reasoning capabilities as tools for LLM integration.

Tools exposed:
- reason_step: Single reasoning step
- run_trm_recursive: Complete recursive reasoning
- dag_add_node: Add node to DAG
- dag_best_path: Get best reasoning path
- dag_get_state: Get DAG statistics
- torque_route: Compute routing weights
- dcaap_forward: Execute dCaAP cell
- fractal_branch: Create DAG branch
- evaluate_score: Score predictions
- load_model / save_model: Model persistence
- get_repo_state / write_repo_state: File operations
- execute_command: Execute system commands
- get_system_info: Get system information
- list_directory: List directory contents
- get_environment_variable: Get environment variables
- check_command_exists: Check command availability

Usage:
    python mcp/server.py

For stdio transport (default MCP):
    python mcp/server.py --stdio

For HTTP transport:
    python mcp/server.py --http --port 8080
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from t_rlinkos_trm_fractal_dag import (
    TRLinkosTRM,
    TRLinkosCore,
    FractalMerkleDAG,
    DCaAPCell,
    TorqueRouter,
    save_model as _save_model,
    load_model as _load_model,
    mse_loss,
    cosine_similarity_loss,
)

# Import system tools
from mcp.tools import system as system_tools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("trlinkos-mcp")


class TRLinkosMCPServer:
    """MCP Server for T-RLINKOS TRM++.

    This server manages:
    - TRLinkosTRM model instances
    - Fractal Merkle-DAG states
    - Tool execution for reasoning operations
    """

    def __init__(
        self,
        x_dim: int = 64,
        y_dim: int = 32,
        z_dim: int = 64,
        hidden_dim: int = 256,
        num_experts: int = 4,
    ):
        """Initialize the MCP server.

        Args:
            x_dim: Input dimension
            y_dim: Output dimension
            z_dim: Internal state dimension
            hidden_dim: Hidden layer dimension
            num_experts: Number of dCaAP experts
        """
        self.config = {
            "x_dim": x_dim,
            "y_dim": y_dim,
            "z_dim": z_dim,
            "hidden_dim": hidden_dim,
            "num_experts": num_experts,
        }

        # Initialize model
        self.model = TRLinkosTRM(
            x_dim=x_dim,
            y_dim=y_dim,
            z_dim=z_dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
        )

        # DAG storage
        self.dags: Dict[str, FractalMerkleDAG] = {}
        self.dag_counter = 0

        # Repository root
        self.repo_root = Path(__file__).parent.parent

        logger.info(f"TRLinkosMCPServer initialized with config: {self.config}")

    def _create_dag(self, store_states: bool = True) -> str:
        """Create a new DAG and return its ID."""
        dag_id = f"dag_{self.dag_counter}"
        self.dag_counter += 1
        self.dags[dag_id] = FractalMerkleDAG(store_states=store_states)
        return dag_id

    def _get_dag(self, dag_id: str) -> Optional[FractalMerkleDAG]:
        """Get a DAG by ID."""
        return self.dags.get(dag_id)

    # ==================== Tool Implementations ====================

    def reason_step(
        self,
        x: List[float],
        y: Optional[List[float]] = None,
        z: Optional[List[float]] = None,
        inner_recursions: int = 3,
    ) -> Dict[str, Any]:
        """Execute a single reasoning step.

        Args:
            x: Input feature vector
            y: Current response state (optional)
            z: Current internal state (optional)
            inner_recursions: Number of inner recursions

        Returns:
            Dict with y_next, z_next states
        """
        x_np = np.array(x, dtype=np.float64).reshape(1, -1)

        if y is None:
            y_np = np.zeros((1, self.config["y_dim"]), dtype=np.float64)
        else:
            y_np = np.array(y, dtype=np.float64).reshape(1, -1)

        if z is None:
            z_np = np.zeros((1, self.config["z_dim"]), dtype=np.float64)
        else:
            z_np = np.array(z, dtype=np.float64).reshape(1, -1)

        # Encode x
        x_enc = self.model.x_encoder(x_np)

        # Execute step
        y_next, z_next = self.model.core.step_reasoning(
            x_enc, y_np, z_np, inner_recursions=inner_recursions
        )

        return {
            "y_next": y_next[0].tolist(),
            "z_next": z_next[0].tolist(),
            "inner_recursions": inner_recursions,
        }

    def run_trm_recursive(
        self,
        x: List[float],
        max_steps: int = 10,
        inner_recursions: int = 3,
        backtrack: bool = False,
        fractal_branching: bool = False,
    ) -> Dict[str, Any]:
        """Run complete recursive reasoning.

        Args:
            x: Input feature vector
            max_steps: Maximum reasoning steps
            inner_recursions: Inner recursions per step
            backtrack: Enable backtracking
            fractal_branching: Enable fractal exploration

        Returns:
            Dict with y_pred, dag_stats, dag_id
        """
        x_np = np.array(x, dtype=np.float64).reshape(1, -1)

        if fractal_branching:
            y_pred, dag = self.model.forward_recursive_fractal(
                x_np,
                max_steps=max_steps,
                inner_recursions=inner_recursions,
                backtrack=backtrack,
            )
        else:
            y_pred, dag = self.model.forward_recursive(
                x_np,
                max_steps=max_steps,
                inner_recursions=inner_recursions,
                backtrack=backtrack,
            )

        # Store DAG
        dag_id = f"dag_{self.dag_counter}"
        self.dag_counter += 1
        self.dags[dag_id] = dag

        # Get statistics
        best_node = dag.get_best_node()
        depth_stats = dag.get_depth_statistics()

        return {
            "y_pred": y_pred[0].tolist(),
            "dag_id": dag_id,
            "dag_stats": {
                "num_nodes": len(dag.nodes),
                "max_depth": max(depth_stats.keys()) if depth_stats else 0,
                "depth_statistics": depth_stats,
                "best_node_step": best_node.step if best_node else None,
                "best_node_score": best_node.score if best_node else None,
            },
            "config": {
                "max_steps": max_steps,
                "inner_recursions": inner_recursions,
                "backtrack": backtrack,
                "fractal_branching": fractal_branching,
            },
        }

    def dag_add_node(
        self,
        dag_id: str,
        step: int,
        y: List[float],
        z: List[float],
        parents: Optional[List[str]] = None,
        score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Add a node to the DAG.

        Args:
            dag_id: DAG identifier
            step: Step number
            y: Response state
            z: Internal state
            parents: Parent node IDs
            score: Optional score

        Returns:
            Dict with node_id
        """
        dag = self._get_dag(dag_id)
        if dag is None:
            # Create new DAG
            dag_id = self._create_dag()
            dag = self.dags[dag_id]

        y_np = np.array(y, dtype=np.float64).reshape(1, -1)
        z_np = np.array(z, dtype=np.float64).reshape(1, -1)

        node_id = dag.add_step(
            step=step,
            y=y_np,
            z=z_np,
            parents=parents or [],
            score=score,
        )

        return {
            "dag_id": dag_id,
            "node_id": node_id,
            "step": step,
            "score": score,
        }

    def dag_best_path(self, dag_id: str) -> Dict[str, Any]:
        """Get the best reasoning path in the DAG.

        Args:
            dag_id: DAG identifier

        Returns:
            Dict with path nodes and best score
        """
        dag = self._get_dag(dag_id)
        if dag is None:
            return {"error": f"DAG not found: {dag_id}"}

        best_node = dag.get_best_node()
        if best_node is None:
            return {"error": "No nodes in DAG", "dag_id": dag_id}

        # Get path from root to best node
        path = dag.get_fractal_path(best_node.node_id)

        return {
            "dag_id": dag_id,
            "best_node_id": best_node.node_id,
            "best_score": best_node.score,
            "path_length": len(path),
            "path": [
                {
                    "node_id": node.node_id,
                    "step": node.step,
                    "depth": node.depth,
                    "score": node.score,
                }
                for node in path
            ],
        }

    def dag_get_state(self, dag_id: str) -> Dict[str, Any]:
        """Get the current state of a DAG.

        Args:
            dag_id: DAG identifier

        Returns:
            Dict with DAG statistics
        """
        dag = self._get_dag(dag_id)
        if dag is None:
            return {"error": f"DAG not found: {dag_id}"}

        best_node = dag.get_best_node()
        depth_stats = dag.get_depth_statistics()

        return {
            "dag_id": dag_id,
            "num_nodes": len(dag.nodes),
            "max_depth": dag.max_depth,
            "store_states": dag.store_states,
            "depth_statistics": depth_stats,
            "root_nodes": dag.root_nodes,
            "best_node": {
                "node_id": best_node.node_id,
                "step": best_node.step,
                "depth": best_node.depth,
                "score": best_node.score,
            } if best_node else None,
        }

    def torque_route(
        self,
        x: List[float],
        y: List[float],
        z: List[float],
    ) -> Dict[str, Any]:
        """Compute Torque Clustering routing weights.

        Args:
            x: Input features
            y: Current response
            z: Internal state

        Returns:
            Dict with routing weights per expert
        """
        x_np = np.array(x, dtype=np.float64).reshape(1, -1)
        y_np = np.array(y, dtype=np.float64).reshape(1, -1)
        z_np = np.array(z, dtype=np.float64).reshape(1, -1)

        weights = self.model.core.router.forward(x_np, y_np, z_np)

        return {
            "weights": weights[0].tolist(),
            "num_experts": self.config["num_experts"],
            "top_expert": int(np.argmax(weights[0])),
        }

    def dcaap_forward(
        self,
        x: List[float],
        y: List[float],
        z: List[float],
    ) -> Dict[str, Any]:
        """Execute forward pass through dCaAP cell.

        Args:
            x: External input
            y: Current response
            z: Internal state

        Returns:
            Dict with z_next state
        """
        x_np = np.array(x, dtype=np.float64).reshape(1, -1)
        y_np = np.array(y, dtype=np.float64).reshape(1, -1)
        z_np = np.array(z, dtype=np.float64).reshape(1, -1)

        # Use first expert
        z_next = self.model.core.experts[0].forward(x_np, y_np, z_np)

        return {
            "z_next": z_next[0].tolist(),
            "input_dim": x_np.shape[1] + y_np.shape[1] + z_np.shape[1],
        }

    def fractal_branch(
        self,
        dag_id: str,
        parent_node_id: str,
        y: List[float],
        z: List[float],
        score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Create a fractal branch in the DAG.

        Args:
            dag_id: DAG identifier
            parent_node_id: Parent node to branch from
            y: Initial y state for branch
            z: Initial z state for branch
            score: Optional score

        Returns:
            Dict with branch_node_id
        """
        dag = self._get_dag(dag_id)
        if dag is None:
            return {"error": f"DAG not found: {dag_id}"}

        y_np = np.array(y, dtype=np.float64).reshape(1, -1)
        z_np = np.array(z, dtype=np.float64).reshape(1, -1)

        branch_id = dag.create_branch(
            parent_node_id=parent_node_id,
            y=y_np,
            z=z_np,
            score=score,
        )

        if branch_id is None:
            return {
                "error": "Could not create branch (max depth reached or invalid parent)",
                "dag_id": dag_id,
            }

        return {
            "dag_id": dag_id,
            "branch_node_id": branch_id,
            "parent_node_id": parent_node_id,
            "score": score,
        }

    def evaluate_score(
        self,
        y_pred: List[float],
        y_target: List[float],
        metric: str = "mse",
    ) -> Dict[str, Any]:
        """Evaluate prediction score.

        Args:
            y_pred: Predicted values
            y_target: Target values
            metric: Scoring metric (mse, cosine, mae)

        Returns:
            Dict with score and metric info
        """
        y_pred_np = np.array(y_pred, dtype=np.float64).reshape(1, -1)
        y_target_np = np.array(y_target, dtype=np.float64).reshape(1, -1)

        if metric == "mse":
            score = mse_loss(y_pred_np, y_target_np)
        elif metric == "cosine":
            score = cosine_similarity_loss(y_pred_np, y_target_np)
        elif metric == "mae":
            score = float(np.mean(np.abs(y_pred_np - y_target_np)))
        else:
            return {"error": f"Unknown metric: {metric}"}

        return {
            "score": score,
            "metric": metric,
            "lower_is_better": True,
        }

    def load_model(self, filepath: str) -> Dict[str, Any]:
        """Load a saved model.

        Args:
            filepath: Path to model file

        Returns:
            Dict with status and config
        """
        try:
            self.model = _load_model(filepath)
            self.config = {
                "x_dim": self.model.x_dim,
                "y_dim": self.model.y_dim,
                "z_dim": self.model.z_dim,
                "hidden_dim": self.model.core.answer_dense1.W.shape[0],
                "num_experts": self.model.core.num_experts,
            }
            return {
                "status": "success",
                "filepath": filepath,
                "config": self.config,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def save_model(self, filepath: str) -> Dict[str, Any]:
        """Save the current model.

        Args:
            filepath: Path to save model

        Returns:
            Dict with status
        """
        try:
            _save_model(self.model, filepath)
            return {
                "status": "success",
                "filepath": filepath,
                "config": self.config,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_repo_state(self, path: str) -> Dict[str, Any]:
        """Get repository file/directory state.

        Args:
            path: Path relative to repo root

        Returns:
            Dict with file content or directory listing
        """
        full_path = self.repo_root / path

        if not full_path.exists():
            return {"error": f"Path not found: {path}"}

        if full_path.is_file():
            try:
                content = full_path.read_text(encoding="utf-8")
                return {
                    "type": "file",
                    "path": path,
                    "content": content,
                    "size": len(content),
                }
            except Exception as e:
                return {"error": f"Could not read file: {e}"}
        elif full_path.is_dir():
            entries = []
            for entry in full_path.iterdir():
                entries.append({
                    "name": entry.name,
                    "type": "directory" if entry.is_dir() else "file",
                    "size": entry.stat().st_size if entry.is_file() else None,
                })
            return {
                "type": "directory",
                "path": path,
                "entries": entries,
            }

        return {"error": f"Unknown path type: {path}"}

    def write_repo_state(
        self,
        path: str,
        content: str,
        mode: str = "overwrite",
    ) -> Dict[str, Any]:
        """Write content to a repository file.

        Args:
            path: Path relative to repo root
            content: Content to write
            mode: "overwrite" or "append"

        Returns:
            Dict with status
        """
        full_path = self.repo_root / path

        # Ensure parent directory exists
        full_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if mode == "append":
                with open(full_path, "a", encoding="utf-8") as f:
                    f.write(content)
            else:
                full_path.write_text(content, encoding="utf-8")

            return {
                "status": "success",
                "path": path,
                "mode": mode,
                "size": len(content),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def execute_command(
        self,
        command: str,
        timeout: int = 30,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Execute a system command.

        Args:
            command: Command to execute
            timeout: Timeout in seconds
            cwd: Working directory
            env: Environment variables

        Returns:
            Dict with command output and status
        """
        return system_tools.execute_command(command, timeout, cwd, env)

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information.

        Returns:
            Dict with system information
        """
        return system_tools.get_system_info()

    def list_directory(self, path: str = ".") -> Dict[str, Any]:
        """List directory contents.

        Args:
            path: Directory path

        Returns:
            Dict with directory listing
        """
        return system_tools.list_directory(path)

    def get_environment_variable(self, name: str) -> Dict[str, Any]:
        """Get an environment variable value.

        Args:
            name: Environment variable name

        Returns:
            Dict with variable value
        """
        return system_tools.get_environment_variable(name)

    def check_command_exists(self, command: str) -> Dict[str, Any]:
        """Check if a command exists in the system PATH.

        Args:
            command: Command name to check

        Returns:
            Dict with existence status and path
        """
        return system_tools.check_command_exists(command)

    # ==================== MCP Protocol Methods ====================

    def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a tool call.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments

        Returns:
            Tool result
        """
        tool_map = {
            "reason_step": self.reason_step,
            "run_trm_recursive": self.run_trm_recursive,
            "dag_add_node": self.dag_add_node,
            "dag_best_path": self.dag_best_path,
            "dag_get_state": self.dag_get_state,
            "torque_route": self.torque_route,
            "dcaap_forward": self.dcaap_forward,
            "fractal_branch": self.fractal_branch,
            "evaluate_score": self.evaluate_score,
            "load_model": self.load_model,
            "save_model": self.save_model,
            "get_repo_state": self.get_repo_state,
            "write_repo_state": self.write_repo_state,
            "execute_command": self.execute_command,
            "get_system_info": self.get_system_info,
            "list_directory": self.list_directory,
            "get_environment_variable": self.get_environment_variable,
            "check_command_exists": self.check_command_exists,
        }

        if tool_name not in tool_map:
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            return tool_map[tool_name](**arguments)
        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}")
            return {"error": str(e)}

    def handle_resource_read(self, uri: str) -> Dict[str, Any]:
        """Handle a resource read request.

        Args:
            uri: Resource URI

        Returns:
            Resource content
        """
        if uri == "trlinkos://model/config":
            return {
                "uri": uri,
                "mimeType": "application/json",
                "content": json.dumps(self.config),
            }
        elif uri.startswith("trlinkos://dag/"):
            dag_id = uri.replace("trlinkos://dag/", "")
            state = self.dag_get_state(dag_id)
            return {
                "uri": uri,
                "mimeType": "application/json",
                "content": json.dumps(state),
            }
        elif uri == "trlinkos://benchmark/results":
            # Return placeholder for benchmark results
            return {
                "uri": uri,
                "mimeType": "application/json",
                "content": json.dumps({"status": "no benchmarks run"}),
            }
        else:
            return {"error": f"Unknown resource: {uri}"}


# ==================== MCP Protocol Implementation ====================


async def handle_stdio(server: TRLinkosMCPServer):
    """Handle MCP communication over stdio.

    This implements the MCP protocol over stdin/stdout.
    """
    logger.info("Starting MCP server on stdio")

    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout
    )
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, asyncio.get_event_loop())

    async def read_message() -> Optional[Dict[str, Any]]:
        """Read a JSON-RPC message from stdin."""
        try:
            line = await reader.readline()
            if not line:
                return None
            return json.loads(line.decode("utf-8"))
        except Exception as e:
            logger.error(f"Error reading message: {e}")
            return None

    def write_message(message: Dict[str, Any]):
        """Write a JSON-RPC message to stdout."""
        data = json.dumps(message) + "\n"
        writer.write(data.encode("utf-8"))

    # Send initialization
    write_message({
        "jsonrpc": "2.0",
        "method": "initialized",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {"listChanged": True},
                "resources": {"subscribe": False, "listChanged": True},
            },
            "serverInfo": {
                "name": "trlinkos-trm-mcp",
                "version": "1.0.0",
            },
        },
    })

    while True:
        message = await read_message()
        if message is None:
            break

        method = message.get("method", "")
        params = message.get("params", {})
        msg_id = message.get("id")

        response: Dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id}

        if method == "tools/list":
            # Load tools from mcp.json
            manifest_path = Path(__file__).parent.parent / "mcp.json"
            with open(manifest_path) as f:
                manifest = json.load(f)
            response["result"] = {"tools": manifest.get("tools", [])}

        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            result = server.handle_tool_call(tool_name, arguments)
            response["result"] = {
                "content": [{"type": "text", "text": json.dumps(result)}]
            }

        elif method == "resources/list":
            manifest_path = Path(__file__).parent.parent / "mcp.json"
            with open(manifest_path) as f:
                manifest = json.load(f)
            response["result"] = {"resources": manifest.get("resources", [])}

        elif method == "resources/read":
            uri = params.get("uri", "")
            result = server.handle_resource_read(uri)
            response["result"] = {
                "contents": [result]
            }

        else:
            response["error"] = {
                "code": -32601,
                "message": f"Method not found: {method}",
            }

        if msg_id is not None:
            write_message(response)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="T-RLINKOS MCP Server")
    parser.add_argument("--stdio", action="store_true", help="Use stdio transport")
    parser.add_argument("--http", action="store_true", help="Use HTTP transport")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port")
    parser.add_argument("--x-dim", type=int, default=64, help="Input dimension")
    parser.add_argument("--y-dim", type=int, default=32, help="Output dimension")
    parser.add_argument("--z-dim", type=int, default=64, help="State dimension")
    args = parser.parse_args()

    server = TRLinkosMCPServer(
        x_dim=args.x_dim,
        y_dim=args.y_dim,
        z_dim=args.z_dim,
    )

    if args.http:
        # HTTP mode (using FastAPI)
        try:
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
            import uvicorn

            app = FastAPI(title="T-RLINKOS MCP Server")

            class ToolCallRequest(BaseModel):
                name: str
                arguments: Dict[str, Any] = {}

            @app.post("/tools/call")
            async def call_tool(request: ToolCallRequest):
                result = server.handle_tool_call(request.name, request.arguments)
                if "error" in result:
                    raise HTTPException(status_code=400, detail=result["error"])
                return result

            @app.get("/tools/list")
            async def list_tools():
                manifest_path = Path(__file__).parent.parent / "mcp.json"
                with open(manifest_path) as f:
                    manifest = json.load(f)
                return {"tools": manifest.get("tools", [])}

            @app.get("/resources/{resource_id:path}")
            async def read_resource(resource_id: str):
                uri = f"trlinkos://{resource_id}"
                return server.handle_resource_read(uri)

            uvicorn.run(app, host="0.0.0.0", port=args.port)

        except ImportError:
            logger.error("FastAPI not installed. Install with: pip install fastapi uvicorn")
            sys.exit(1)
    else:
        # Default: stdio mode
        asyncio.run(handle_stdio(server))


if __name__ == "__main__":
    main()
