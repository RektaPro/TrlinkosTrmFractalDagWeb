"""Repository tools for MCP - File operations."""

from typing import Any, Dict
from pathlib import Path


# Repository root
REPO_ROOT = Path(__file__).parent.parent.parent


def get_repo_state(path: str) -> Dict[str, Any]:
    """Get the state of files in the repository.

    Args:
        path: Path to file or directory relative to repo root

    Returns:
        Dict with file content or directory listing
    """
    full_path = REPO_ROOT / path

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
    path: str,
    content: str,
    mode: str = "overwrite",
) -> Dict[str, Any]:
    """Write content to a file in the repository.

    Args:
        path: Path to file relative to repo root
        content: Content to write
        mode: "overwrite" or "append"

    Returns:
        Dict with status
    """
    full_path = REPO_ROOT / path

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
