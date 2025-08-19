


import os
import asyncio
import inspect
import hashlib
import json
from typing import List, Dict, Any, Optional

from langchain_core.tools import StructuredTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
import logging

logger = logging.getLogger(__name__)

# Global caches
WRAPPED_TOOLS_CACHE: Optional[List[StructuredTool]] = None
ENV_VARS_CACHE: Optional[dict] = None

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 0.5



def create_stateless_tool_wrapper(original_mcp_tool):
    """
    Create a stateless tool wrapper using StructuredTool pattern.
    Creates fresh MCP session on each call with retry logic and concurrency control.
    """
    tool_name = original_mcp_tool.name
    
    async def wrapped_func(**kwargs):
        """Execute tool with fresh MCP session."""
        
        # Create execution ID for logging
        id_source = f"{tool_name}:{json.dumps(kwargs, sort_keys=True)}"
        execution_id = hashlib.md5(id_source.encode()).hexdigest()[:8]
        
        logger.debug(f"🚀 MCP WRAPPER START [{execution_id}] Tool: {tool_name}, Args: {kwargs}")
        
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                # Each MCP session is independent - no locking needed
                server_path = _find_mcp_server_path()
                server_params = StdioServerParameters(
                    command="python3",
                    args=[server_path],
                    env=_get_env_vars()
                )
                
                # Create fresh session for this tool call
                async with stdio_client(server_params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools = await load_mcp_tools(session)
                        
                        # Find and execute the specific tool
                        for t in tools:
                            if t.name == tool_name:
                                logger.debug(f"🔧 [{execution_id}] Executing {tool_name} with args: {kwargs}")
                                
                                # Smart tool invocation pattern from working wrapper
                                if hasattr(t, 'ainvoke'):
                                    logger.debug(f"🔧 [{execution_id}] Using ainvoke for {tool_name}")
                                    result = await t.ainvoke(kwargs)
                                elif hasattr(t, 'invoke'):
                                    logger.debug(f"🔧 [{execution_id}] Using invoke for {tool_name}")
                                    result = t.invoke(kwargs)
                                    if inspect.isawaitable(result):
                                        result = await result
                                else:
                                    logger.debug(f"🔧 [{execution_id}] Using direct call for {tool_name}")
                                    result = await t(**kwargs) if inspect.iscoroutinefunction(t) else t(**kwargs)
                                
                                logger.debug(f"✅ [{execution_id}] Tool {tool_name} completed successfully")
                                return result
                        
                        raise ValueError(f"Tool {tool_name} not found in MCP server")
                            
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Tool {tool_name} failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"❌ [{execution_id}] Tool {tool_name} failed after {MAX_RETRIES} attempts: {e}")
                    raise last_error
    
    # Create StructuredTool with proper schema preservation (like working wrapper)
    return StructuredTool(
        name=original_mcp_tool.name,
        description=original_mcp_tool.description,
        coroutine=wrapped_func,  # Only provide coroutine for async function
        args_schema=original_mcp_tool.args_schema if hasattr(original_mcp_tool, 'args_schema') else None,
    )

async def create_stateless_mcp_tools() -> List:
    """
    Create all MCP tools as stateless wrappers.
    Caches wrapped tools to avoid recreating on every call.
    Main entry point for supervisor system.
    """
    global WRAPPED_TOOLS_CACHE
    
    # Return cached wrapped tools if available
    if WRAPPED_TOOLS_CACHE is not None:
        logger.info(f"Using cached {len(WRAPPED_TOOLS_CACHE)} wrapped MCP tools")
        return WRAPPED_TOOLS_CACHE
    
    try:
        # Load MCP tools directly when cache is empty
        server_path = _find_mcp_server_path()
        server_params = StdioServerParameters(
            command="python3",
            args=[server_path],
            env=_get_env_vars()
        )
        
        # Connect to MCP server and load tools
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tool_instances = await load_mcp_tools(session)
        
        # Create and cache the wrapped tools
        WRAPPED_TOOLS_CACHE = []
        for original_tool in tool_instances:
            wrapper = create_stateless_tool_wrapper(original_tool)
            WRAPPED_TOOLS_CACHE.append(wrapper)
        
        logger.info(f"Created and cached {len(WRAPPED_TOOLS_CACHE)} stateless MCP tools")
        return WRAPPED_TOOLS_CACHE
        
    except Exception as e:
        logger.error(f"Failed to create stateless tools: {e}")
        return []  # Return empty list to allow supervisor to work without tools

def _find_mcp_server_path() -> str:
    """Find MCP server path."""
    server_path = os.environ.get("SDK_MCP_SERVER_PATH")
    if not server_path:
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                         "story-mcp-hub/story-sdk-mcp/server.py"),
            os.path.join(os.getcwd(), "story-mcp-hub/story-sdk-mcp/server.py"),
            os.path.join(os.path.dirname(os.getcwd()), "story-mcp-hub/story-sdk-mcp/server.py"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError("Could not find MCP server at any expected location")
    
    if not os.path.exists(server_path):
        raise FileNotFoundError(f"MCP server not found at: {server_path}")
    
    return server_path

def _get_env_vars() -> dict:
    """Get environment variables for MCP server (cached)."""
    global ENV_VARS_CACHE
    
    if ENV_VARS_CACHE is not None:
        return ENV_VARS_CACHE.copy()
    
    ENV_VARS_CACHE = os.environ.copy()
    
    # Load from .env file if exists
    server_path = _find_mcp_server_path()
    server_dir = os.path.dirname(server_path)
    env_path = os.path.join(server_dir, '.env')
    
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip('"').strip("'")
                    if '#' in value:
                        value = value.split('#')[0].strip()
                    ENV_VARS_CACHE[key] = value
    
    logger.info("Cached environment variables for MCP server")
    return ENV_VARS_CACHE.copy()

def clear_all_caches():
    """
    Clear all caches. Useful for development or when MCP server changes.
    """
    global WRAPPED_TOOLS_CACHE, ENV_VARS_CACHE
    
    WRAPPED_TOOLS_CACHE = None  
    ENV_VARS_CACHE = None
    
    logger.info("Cleared all MCP tool caches")
