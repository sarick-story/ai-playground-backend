#!/usr/bin/env python3
"""
Debug script to inspect MCP tool loading and types.
"""
import asyncio
import os
import sys

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, '/Users/zizhuoliu/Desktop/Story/ai-playground-backend')

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

async def debug_mcp_tools():
    """Load MCP tools and inspect their types and attributes."""
    
    # Get server path (same logic as tools_wrapper.py)
    server_path = os.environ.get("SDK_MCP_SERVER_PATH")
    if not server_path:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate = os.path.join(repo_root, "story-mcp-hub", "story-sdk-mcp", "server.py")
        if os.path.exists(candidate):
            server_path = candidate
        else:
            print("Could not find story-sdk-mcp server.py")
            return
    
    print(f"Using MCP server: {server_path}")
    
    server_params = StdioServerParameters(
        command="python3",
        args=[server_path],
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                
                print(f"\nLoaded {len(tools)} MCP tools:")
                
                for i, tool in enumerate(tools):
                    print(f"\n--- Tool {i+1} ---")
                    print(f"Tool object: {tool}")
                    print(f"Tool type: {type(tool)}")
                    print(f"Tool name: {getattr(tool, 'name', 'NO NAME')}")
                    print(f"Tool description: {getattr(tool, 'description', 'NO DESCRIPTION')}")
                    
                    # Check all attributes
                    attrs = dir(tool)
                    method_attrs = [attr for attr in attrs if not attr.startswith('_')]
                    print(f"Available methods/attrs: {method_attrs}")
                    
                    # Check if it's callable
                    print(f"Is callable: {callable(tool)}")
                    
                    # Check for invoke methods
                    has_invoke = hasattr(tool, 'invoke')
                    has_ainvoke = hasattr(tool, 'ainvoke')
                    print(f"Has invoke: {has_invoke}, Has ainvoke: {has_ainvoke}")
                    
                    if hasattr(tool, 'args_schema'):
                        print(f"Args schema: {tool.args_schema}")
                    
                    # Limit output to first 3 tools for readability
                    if i >= 2:
                        break
                
    except Exception as e:
        print(f"Error loading MCP tools: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_mcp_tools())