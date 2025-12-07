#!/usr/bin/env python3
"""
Test script for MCP system tools.

This script tests all system tools added to the MCP server.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server import TRLinkosMCPServer


def test_system_tools():
    """Test all system tools."""
    print("=" * 60)
    print("Testing MCP System Tools")
    print("=" * 60)
    
    server = TRLinkosMCPServer(x_dim=64, y_dim=32, z_dim=64)
    
    # Test 1: get_system_info
    print("\n1. Testing get_system_info...")
    result = server.get_system_info()
    assert result["status"] == "success", "get_system_info failed"
    print(f"   ✓ OS: {result['system']['os']}")
    print(f"   ✓ Python: {result['system']['python_implementation']}")
    print(f"   ✓ Machine: {result['system']['machine']}")
    
    # Test 2: check_command_exists
    print("\n2. Testing check_command_exists...")
    result = server.check_command_exists("python")
    assert result["status"] == "success", "check_command_exists failed"
    assert result["exists"], "python command not found"
    print(f"   ✓ Command 'python' exists at: {result['path']}")
    
    result = server.check_command_exists("nonexistent_command_xyz")
    assert not result["exists"], "nonexistent command should not exist"
    print(f"   ✓ Nonexistent command correctly reported as not found")
    
    # Test 3: execute_command
    print("\n3. Testing execute_command...")
    result = server.execute_command("python -c \"print('Hello from MCP')\"")
    assert result["status"] == "success", "execute_command failed"
    assert "Hello from MCP" in result["stdout"], "command output incorrect"
    print(f"   ✓ Command output: {result['stdout'].strip()}")
    
    # Test 4: execute_command with timeout
    print("\n4. Testing execute_command with python...")
    result = server.execute_command("python -c \"print('MCP Test')\"")
    assert result["status"] == "success", "python command failed"
    assert "MCP Test" in result["stdout"], "python output incorrect"
    print(f"   ✓ Python command output: {result['stdout'].strip()}")
    
    # Test 5: list_directory
    print("\n5. Testing list_directory...")
    result = server.list_directory(".")
    assert result["status"] == "success", "list_directory failed"
    assert result["count"] > 0, "directory should have entries"
    print(f"   ✓ Found {result['count']} entries in current directory")
    # Check for some expected files
    entry_names = [e["name"] for e in result["entries"]]
    assert "mcp.json" in entry_names, "mcp.json not found"
    assert "README.md" in entry_names, "README.md not found"
    print(f"   ✓ Key files found: mcp.json, README.md")
    
    # Test 6: get_environment_variable
    print("\n6. Testing get_environment_variable...")
    result = server.get_environment_variable("PATH")
    assert result["status"] == "success", "get_environment_variable failed"
    assert result["value"] is not None, "PATH should be set"
    print(f"   ✓ PATH exists (length: {len(result['value'])})")
    
    result = server.get_environment_variable("NONEXISTENT_VAR_XYZ")
    assert result["status"] == "not_found", "nonexistent var should return not_found"
    print(f"   ✓ Nonexistent variable correctly reported as not found")
    
    # Test 7: Error handling - command not found
    print("\n7. Testing error handling...")
    result = server.execute_command("nonexistent_command_xyz")
    assert result["status"] == "error", "should report error for nonexistent command"
    print(f"   ✓ Nonexistent command error: {result['error']}")
    
    # Test 8: Error handling - invalid directory
    result = server.list_directory("/nonexistent/directory/xyz")
    assert result["status"] == "error", "should report error for nonexistent directory"
    print(f"   ✓ Nonexistent directory error: {result['error']}")
    
    print("\n" + "=" * 60)
    print("✓ All system tools tests passed!")
    print("=" * 60)


def test_tool_call_interface():
    """Test that tools work through the handle_tool_call interface."""
    print("\n" + "=" * 60)
    print("Testing handle_tool_call Interface")
    print("=" * 60)
    
    server = TRLinkosMCPServer(x_dim=64, y_dim=32, z_dim=64)
    
    # Test execute_command through handle_tool_call
    print("\n1. Testing execute_command via handle_tool_call...")
    result = server.handle_tool_call("execute_command", {
        "command": "python -c \"print('MCP Tool Call')\""
    })
    assert result["status"] == "success", "tool call failed"
    print(f"   ✓ Output: {result['stdout'].strip()}")
    
    # Test get_system_info through handle_tool_call
    print("\n2. Testing get_system_info via handle_tool_call...")
    result = server.handle_tool_call("get_system_info", {})
    assert result["status"] == "success", "tool call failed"
    print(f"   ✓ OS: {result['system']['os']}")
    
    # Test list_directory through handle_tool_call
    print("\n3. Testing list_directory via handle_tool_call...")
    result = server.handle_tool_call("list_directory", {"path": "."})
    assert result["status"] == "success", "tool call failed"
    print(f"   ✓ Found {result['count']} entries")
    
    print("\n" + "=" * 60)
    print("✓ All handle_tool_call tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_system_tools()
    test_tool_call_interface()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
