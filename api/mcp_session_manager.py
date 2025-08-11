"""
MCP Session Manager for maintaining persistent MCP sessions across supervisor agents.
"""

import os
import logging
from typing import List, Optional, Dict, Any
from langchain_mcp_adapters.client import MultiServerMCPClient
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

logger = logging.getLogger(__name__)


class MCPSessionManager:
    """Manages MCP sessions for supervisor agents."""
    
    def __init__(self):
        self.client = None
        self.tools = None
        self.session = None
        self.read = None
        self.write = None
        
    async def initialize(self):
        """Initialize the MCP client and load tools."""
        try:
            # Check if we should use MultiServerMCPClient
            use_multi_client = os.environ.get("USE_MULTI_MCP_CLIENT", "false").lower() == "true"
            
            if use_multi_client:
                # Use MultiServerMCPClient for better session management
                logger.info("Initializing MultiServerMCPClient")
                
                server_path = self._get_server_path()
                
                self.client = MultiServerMCPClient(
                    {
                        "story_sdk": {
                            "command": "python3",
                            "args": [server_path],
                            "transport": "stdio",
                        }
                    }
                )
                
                # Get tools from the client
                self.tools = await self.client.get_tools()
                logger.info(f"Loaded {len(self.tools)} tools via MultiServerMCPClient")
                
            else:
                # Use traditional stdio client with persistent session
                logger.info("Initializing stdio client with persistent session")
                
                server_path = self._get_server_path()
                server_params = StdioServerParameters(
                    command="python3",
                    args=[server_path],
                )
                
                # Start the stdio client (but don't use context manager)
                self.read, self.write = await stdio_client(server_params).__aenter__()
                
                # Create a persistent session
                self.session = ClientSession(self.read, self.write)
                await self.session.__aenter__()
                await self.session.initialize()
                
                # Load tools with the persistent session
                self.tools = await load_mcp_tools(self.session)
                logger.info(f"Loaded {len(self.tools)} tools with persistent session")
                
        except Exception as e:
            logger.error(f"Failed to initialize MCP session: {str(e)}")
            raise
    
    def _get_server_path(self) -> str:
        """Get the path to the MCP server."""
        server_path = os.environ.get("SDK_MCP_SERVER_PATH")
        if not server_path:
            # Try to find the server path relative to this file
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            candidate = os.path.join(repo_root, "story-mcp-hub", "story-sdk-mcp", "server.py")
            if os.path.exists(candidate):
                server_path = candidate
            else:
                raise FileNotFoundError("Could not find story-sdk-mcp server.py. Set SDK_MCP_SERVER_PATH to override.")
        return server_path
    
    async def get_tools(self) -> List:
        """Get the loaded MCP tools."""
        if self.tools is None:
            await self.initialize()
        return self.tools
    
    async def cleanup(self):
        """Clean up the MCP session."""
        try:
            if self.client:
                # MultiServerMCPClient handles its own cleanup
                logger.info("Cleaning up MultiServerMCPClient")
                # The client will clean up when it goes out of scope
                self.client = None
                
            elif self.session:
                # Clean up persistent session
                logger.info("Cleaning up persistent MCP session")
                await self.session.__aexit__(None, None, None)
                self.session = None
                
                if self.read and self.write:
                    # Clean up stdio client
                    await stdio_client.__aexit__(None, None, None)
                    self.read = None
                    self.write = None
                    
        except Exception as e:
            logger.warning(f"Error during MCP session cleanup: {str(e)}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()


# Global session manager instance
_session_manager: Optional[MCPSessionManager] = None


async def get_mcp_session_manager() -> MCPSessionManager:
    """Get or create the global MCP session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = MCPSessionManager()
        await _session_manager.initialize()
    return _session_manager


async def cleanup_mcp_session():
    """Clean up the global MCP session."""
    global _session_manager
    if _session_manager:
        await _session_manager.cleanup()
        _session_manager = None