"""System tools for MCP - System command execution and information."""

import os
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional


def execute_command(
    command: str,
    timeout: int = 30,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Execute a system command safely.
    
    Args:
        command: Command to execute
        timeout: Timeout in seconds (default: 30)
        cwd: Working directory (default: current directory)
        env: Environment variables (default: inherit from parent)
        
    Returns:
        Dict with stdout, stderr, return_code, and status
    """
    import shlex
    
    try:
        # Security: Use shell=False for better security
        # Split command into args using shlex for proper parsing
        if isinstance(command, str):
            cmd_args = shlex.split(command)
        else:
            cmd_args = command
        
        # Prepare environment
        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)
        
        # Execute command
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=exec_env,
            shell=False,
        )
        
        return {
            "status": "success",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
            "command": command,
        }
        
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error": f"Command timed out after {timeout} seconds",
            "command": command,
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "error": "Command not found",
            "command": command,
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "command": command,
        }


def get_system_info() -> Dict[str, Any]:
    """Get system information.
    
    Returns:
        Dict with system information including OS, Python version, etc.
    """
    try:
        return {
            "status": "success",
            "system": {
                "os": platform.system(),
                "os_version": platform.version(),
                "os_release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": sys.version,
                "python_implementation": platform.python_implementation(),
                "hostname": platform.node(),
            },
            "environment": {
                "cwd": os.getcwd(),
                "home": os.path.expanduser("~"),
                "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
            },
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def list_directory(path: str = ".") -> Dict[str, Any]:
    """List directory contents.
    
    Args:
        path: Directory path (default: current directory)
        
    Returns:
        Dict with directory listing
    """
    try:
        full_path = os.path.abspath(path)
        
        if not os.path.exists(full_path):
            return {
                "status": "error",
                "error": f"Path not found: {path}",
            }
        
        if not os.path.isdir(full_path):
            return {
                "status": "error",
                "error": f"Path is not a directory: {path}",
            }
        
        entries = []
        for entry_name in os.listdir(full_path):
            entry_path = os.path.join(full_path, entry_name)
            stat_info = os.stat(entry_path)
            
            entries.append({
                "name": entry_name,
                "type": "directory" if os.path.isdir(entry_path) else "file",
                "size": stat_info.st_size,
                "modified": stat_info.st_mtime,
            })
        
        return {
            "status": "success",
            "path": full_path,
            "entries": entries,
            "count": len(entries),
        }
        
    except PermissionError:
        return {
            "status": "error",
            "error": f"Permission denied: {path}",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


def get_environment_variable(name: str) -> Dict[str, Any]:
    """Get an environment variable value.
    
    Args:
        name: Environment variable name
        
    Returns:
        Dict with variable value
    """
    value = os.environ.get(name)
    
    if value is None:
        return {
            "status": "not_found",
            "name": name,
            "value": None,
        }
    
    return {
        "status": "success",
        "name": name,
        "value": value,
    }


def check_command_exists(command: str) -> Dict[str, Any]:
    """Check if a command exists in the system PATH.
    
    Args:
        command: Command name to check
        
    Returns:
        Dict with existence status and path
    """
    import shutil
    
    path = shutil.which(command)
    
    return {
        "status": "success",
        "command": command,
        "exists": path is not None,
        "path": path,
    }
