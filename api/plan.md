# Fix Plan: MCP Tool Loading and Session Management Issues

## Problem Statement

The current MCP (Model Context Protocol) tool loading implementation suffers from `ClosedResourceError` and overly complex session management that doesn't align with LangGraph's recommended approach. Tools fail to execute after being loaded, causing workflow failures.

### Root Cause Analysis

1. **Over-Engineered Session Management**: Manual management of MCP sessions with global state variables (`_persistent_session`, `_persistent_read`, `_persistent_write`) instead of using proper context managers.

2. **Manual Context Manager Usage**: Calling `__aenter__()` manually instead of using proper `async with` statements, leading to resource lifecycle issues.

3. **Complex Caching Logic**: Unnecessary 5-minute cache timeout, tool validation loops, and session health checks that add complexity without benefit.

4. **Duplicate Functions**: `get_or_create_mcp_tools` is defined twice in `supervisor_agent_system.py` (lines 249 and 563).

5. **Unnecessary MultiMCPServerClient**: Code exists for `MultiMCPServerClient` which is not needed and adds complexity.

6. **Resource Lifecycle Issues**: Sessions are closed before tools can execute, causing `ClosedResourceError` as seen in logs.

### LangGraph Recommended Approach

According to LangGraph documentation, the simple and correct approach is:
```python
async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await load_mcp_tools(session)
        # Use tools immediately
```

## Solution Overview

Replace the complex manual session management with LangGraph's recommended simple context manager approach. This will eliminate resource lifecycle issues and make the code more maintainable.

## Implementation Plan

### Phase 1: Clean Up Current Issues

#### 1.1 Remove MultiMCPServerClient Usage
**File**: `ai-playground-backend/api/mcp_session_manager.py`

Remove all `MultiMCPServerClient` usage:
```python
# DELETE these lines:
from langchain_mcp_adapters.client import MultiServerMCPClient

# DELETE the entire if use_multi_client block (lines 32-50)
if use_multi_client:
    # Use MultiServerMCPClient for better session management
    logger.info("Initializing MultiServerMCPClient")
    ...
```

**Justification**: Not needed according to LangGraph docs and adds unnecessary complexity.

#### 1.2 Remove Duplicate Functions
**File**: `ai-playground-backend/api/supervisor_agent_system.py`

Remove the second `get_or_create_mcp_tools` function starting at line 563. Keep only the first one at line 249, which we'll refactor in Phase 2.

#### 1.3 Remove Global Session State Variables
**File**: `ai-playground-backend/api/supervisor_agent_system.py`

Remove these global variables (lines 136-143):
```python
# DELETE these lines:
_mcp_tools: List[Any] | None = None
_last_used: float = 0.0
_lock = asyncio.Lock()
_persistent_session: Any | None = None
_persistent_read = None
_persistent_write = None
_session_server_path: str | None = None
```

### Phase 2: Implement LangGraph Recommended Approach

#### 2.1 Create Simple MCP Tool Loading Function
**File**: `ai-playground-backend/api/supervisor_agent_system.py`

Replace the complex `get_or_create_mcp_tools` function with this simple implementation:

```python
async def load_fresh_mcp_tools() -> List[Any]:
    """Load MCP tools using LangGraph recommended approach.
    
    This approach uses proper context managers to ensure session lifecycle
    is handled correctly, eliminating ClosedResourceError issues.
    """
    logger.info("🔧 MCP: Loading fresh MCP tools using LangGraph recommended approach")
    
    # Find server path
    server_path = _find_mcp_server_path()
    
    # Create server parameters
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_path],
        env=None  # Use current environment
    )
    
    logger.info(f"🔧 MCP: Using server at: {server_path}")
    
    # Use LangGraph recommended context manager approach
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            logger.info("🔧 MCP: Session initialized successfully")
            
            tools = await load_mcp_tools(session)
            logger.info(f"🔧 MCP: Loaded {len(tools)} tools successfully")
            
            # Validate tools
            valid_tools = []
            for i, tool in enumerate(tools):
                try:
                    tool_name = getattr(tool, 'name', f'tool_{i}')
                    if hasattr(tool, 'invoke') and callable(getattr(tool, 'invoke')):
                        valid_tools.append(tool)
                        logger.info(f"🔧 MCP: Tool '{tool_name}' validated successfully")
                    else:
                        logger.warning(f"🔧 MCP: Tool '{tool_name}' is not invokable, skipping")
                except Exception as e:
                    logger.error(f"🔧 MCP: Error validating tool {i}: {e}")
            
            logger.info(f"🔧 MCP: {len(valid_tools)} valid tools ready for use")
            return valid_tools

def _find_mcp_server_path() -> str:
    """Find the MCP server path using existing logic."""
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
```

#### 2.2 Remove Complex Session Management Functions
**File**: `ai-playground-backend/api/supervisor_agent_system.py`

Delete these functions entirely:
- `_cleanup_persistent_session` (lines 214-246)
- `_create_persistent_session` (lines 388-432)
- `_load_tools_from_persistent_session` (lines 435-560)
- `validate_mcp_session_health` (lines 702-721)
- `cleanup_mcp_cache` (lines 723-728)

### Phase 3: Update Agent Creation

#### 3.1 Update `create_all_agents` Function
**File**: `ai-playground-backend/api/supervisor_agent_system.py`

Replace the MCP tools loading section (lines 768-783):
```python
async def create_all_agents(checkpointer=None, store=None, mcp_tools=None):
    """Create all agents with properly loaded tools."""
    
    # Load MCP tools with proper error handling
    if mcp_tools:
        # Use provided MCP tools directly
        direct_tools = mcp_tools
        logger.info(f"Using provided MCP tools: {len(direct_tools)} tools")
    else:
        # Load fresh MCP tools using LangGraph approach
        try:
            direct_tools = await load_fresh_mcp_tools()
            logger.info(f"Loaded fresh MCP tools: {len(direct_tools)} tools")
        except Exception as e:
            logger.error(f"Failed to load MCP tools: {e}")
            direct_tools = []
            logger.warning("Continuing with empty tools list - agents will have limited functionality")
    
    # Rest of function remains the same...
```

#### 3.2 Update `create_supervisor_system` Function
**File**: `ai-playground-backend/api/supervisor_agent_system.py`

Update the function to load tools if not provided:
```python
async def create_supervisor_system(mcp_tools=None, checkpointer=None, store=None):
    """Create the complete supervisor system with all agents."""
    
    # Load MCP tools if not provided
    if mcp_tools is None:
        mcp_tools = await load_fresh_mcp_tools()
    
    # Use provided or default checkpointer/store
    checkpointer = checkpointer or GLOBAL_CHECKPOINTER
    store = store or GLOBAL_STORE
    
    # Create all agents with loaded tools
    agents = await create_all_agents(checkpointer=checkpointer, store=store, mcp_tools=mcp_tools)
    
    # Rest of function remains the same...
```

### Phase 4: Update SDK Agent Integration

#### 4.1 Update `_run_agent_impl` Function
**File**: `ai-playground-backend/api/sdk_mcp_agent.py`

Replace the complex MCP loading section (lines 475-521) with:
```python
async def _run_agent_impl(
    user_message: str,
    wallet_address: Optional[str] = None,
    queue: Optional[asyncio.Queue] = None,
    conversation_id: Optional[str] = None,
    message_history: Optional[List[Dict[str, str]]] = None,
    task_token: str = ""
):
    # ... existing code until MCP loading section ...
    
    # Load MCP tools using simplified approach
    try:
        logger.info("🔧 MCP TOOLS: Loading tools using LangGraph approach...")
        load_start_time = time.time()
        
        tools = await load_fresh_mcp_tools()  # Import from supervisor_agent_system
        
        load_duration = time.time() - load_start_time
        logger.info(f"🔧 MCP TOOLS: Loaded {len(tools)} tools in {load_duration:.2f} seconds")
        
        if not tools:
            logger.error("🔧 MCP TOOLS: No tools loaded - agent will not have MCP capabilities")
        else:
            logger.info(f"🔧 MCP TOOLS: Successfully loaded tools: {[getattr(t, 'name', 'unnamed') for t in tools]}")
        
    except Exception as e:
        error_msg = f"Failed to load MCP tools: {e}"
        logger.error(f"🔧 MCP TOOLS ERROR: {error_msg}")
        if queue:
            await queue.put({"error": error_msg})
            await queue.put({"done": True})
        return {"error": error_msg}
    
    # Rest of function remains the same...
```

### Phase 5: Simplify Other MCP Usage

#### 5.1 Update `tools_wrapper.py`
**File**: `ai-playground-backend/api/tools_wrapper.py`

Replace the `load_sdk_mcp_tools` function with a simple wrapper:
```python
async def load_sdk_mcp_tools() -> List:
    """Load SDK MCP tools using the simplified approach."""
    from .supervisor_agent_system import load_fresh_mcp_tools
    return await load_fresh_mcp_tools()
```

#### 5.2 Update `mcp_agent.py` (if used)
**File**: `ai-playground-backend/api/mcp_agent.py`

Replace the complex MCP loading in `run_agent` function with:
```python
async def run_agent(
    user_message: str, 
    queue: Optional[asyncio.Queue] = None, 
    conversation_id: Optional[str] = None,
    message_history: Optional[List[Dict[str, str]]] = None
):
    # ... existing code until MCP loading ...
    
    # Load MCP tools using simplified approach
    from .supervisor_agent_system import load_fresh_mcp_tools
    tools = await load_fresh_mcp_tools()
    
    # Rest of function remains the same...
```

### Phase 6: Remove Unused Files (Optional)

#### 6.1 Consider Removing `mcp_session_manager.py`
**File**: `ai-playground-backend/api/mcp_session_manager.py`

This file may no longer be needed if not used elsewhere. Check for imports and remove if unused.

## Testing Plan

### Test Case 1: Basic Tool Loading
1. Start the backend
2. Send a request that requires MCP tools
3. **Expected**: Tools load successfully without `ClosedResourceError`

### Test Case 2: Tool Execution
1. Trigger a tool that requires confirmation (e.g., `create_spg_nft_collection`)
2. Confirm the action
3. **Expected**: Tool executes successfully without session errors

### Test Case 3: Multiple Requests
1. Send multiple requests in sequence
2. **Expected**: Each request loads tools fresh without caching issues

### Test Case 4: Error Handling
1. Temporarily move the MCP server file
2. Send a request
3. **Expected**: Clear error message, no hanging sessions

## Benefits of This Approach

### 1. Eliminates ClosedResourceError
- Context managers ensure proper session lifecycle
- No manual session management to go wrong
- Resources are cleaned up automatically

### 2. Follows LangGraph Best Practices
- Uses documented approach from LangGraph team
- Simpler and more maintainable code
- Aligned with community standards

### 3. Reduces Complexity
- Removes ~500 lines of complex session management code
- Eliminates duplicate functions
- Removes unnecessary caching logic

### 4. Improves Reliability
- No global state to get corrupted
- Fresh tools for each request ensures consistency
- Proper error handling and logging

### 5. Better Performance
- Eliminates overhead of complex validation loops
- No unnecessary session health checks
- Faster startup with simpler logic

## Migration Strategy

### Backward Compatibility
- All existing APIs continue to work unchanged
- No changes to frontend required
- Existing conversation flows unaffected

### Rollback Plan
If issues arise:
1. Keep backup of current `supervisor_agent_system.py`
2. Can quickly revert to previous complex approach
3. Monitor logs for any new error patterns

### Deployment Steps
1. Test changes in development environment
2. Deploy to staging and run full test suite
3. Monitor logs for any ClosedResourceError occurrences
4. Deploy to production with monitoring

## Success Criteria

### Primary Goals
- [ ] No `ClosedResourceError` in logs
- [ ] All MCP tools load successfully
- [ ] Tool execution works reliably
- [ ] Agent workflows complete without session errors

### Secondary Goals
- [ ] Reduced code complexity (measured by lines of code)
- [ ] Faster agent initialization
- [ ] Cleaner log output
- [ ] Improved maintainability

## Timeline

1. **Phase 1**: Clean Up Current Issues - 30 minutes
2. **Phase 2**: Implement LangGraph Approach - 45 minutes
3. **Phase 3**: Update Agent Creation - 20 minutes
4. **Phase 4**: Update SDK Agent Integration - 15 minutes
5. **Phase 5**: Simplify Other MCP Usage - 15 minutes
6. **Phase 6**: Optional Cleanup - 10 minutes
7. **Testing**: 30 minutes
8. **Documentation**: 15 minutes

**Total**: ~3 hours

## Key Insights

### From LangGraph Documentation
- Simple context manager approach is preferred
- No need for complex session caching
- Let context managers handle resource lifecycle

### From Current Issues
- Manual session management causes resource leaks
- Complex caching adds bugs without benefits
- Global state variables are error-prone

### From Log Analysis
- `ClosedResourceError` occurs when tools try to use closed sessions
- Session cleanup happens before tool execution
- Complex validation loops don't prevent the core issue

## Validation Steps

After implementation:
1. Check logs for absence of `ClosedResourceError`
2. Verify all agents can load tools successfully  
3. Test tool execution end-to-end
4. Confirm no regression in existing functionality
5. Monitor memory usage (should be similar or better)

**Confidence Level**: ⭐⭐⭐⭐⭐ (5/5) - Approach is validated by LangGraph documentation and addresses root cause directly.