# Fix Plan: LangGraph Interrupt Resume Issue

## Problem Statement

When a tool triggers an interrupt in the LangGraph workflow (e.g., `create_spg_nft_collection` requiring confirmation), the graph does not resume from the interrupted point after user confirmation. Instead, it appears to create a new agent instance and starts fresh execution.

### Root Cause Analysis

1. **Different Checkpointer Instances**: The initial execution creates a supervisor with a fresh `InMemorySaver`, while the resume execution creates a different supervisor with a different `InMemorySaver`. Since checkpoints are stored in-memory and not shared, the paused state cannot be found during resume.

2. **MCP Session Lifecycle**: The initial graph is created inside an `async with` MCP stdio session. When returning after an interrupt, that session and the `StructuredTool`s bound to it are closed, making those tools unusable even if the same graph instance were reused. **Note**: Our `/langgraph-mcp-agent/` example shows this can be solved by keeping the MCP session open for the entire workflow.

3. **Cache Invalidation**: The `_supervisor_cache` in `supervisor_agent_system.py` is never populated with MCP tools, so `get_supervisor()` always creates tools separately from the SDK agent's flow.

## Solution Overview

Implement shared checkpointer/store instances across initial execution and resume paths, ensuring the graph can find and resume from the correct checkpoint. Additionally, handle MCP tool lifecycle properly by re-creating tools on resume.

## Implementation Plan

### Phase 1: Create Global Checkpointer Infrastructure

#### 1.1 Add Global Checkpointer Instances
**File**: `api/supervisor_agent_system.py`

Add at the top of the file after imports:
```python
# Global checkpointer and store for persistence across requests
GLOBAL_CHECKPOINTER = InMemorySaver()
GLOBAL_STORE = InMemoryStore()

# Note: Consider using MemorySaver instead of InMemorySaver based on 
# successful langgraph-mcp-agent implementation:
# from langgraph.checkpoint.memory import MemorySaver
# GLOBAL_CHECKPOINTER = MemorySaver()
```

#### 1.2 Modify `create_supervisor_system` Function
**File**: `api/supervisor_agent_system.py`

Update function signature to accept optional checkpointer/store:
```python
async def create_supervisor_system(mcp_tools=None, checkpointer=None, store=None):
    """Create the complete supervisor system with all agents.
    
    Args:
        mcp_tools: Optional pre-loaded MCP tools
        checkpointer: Optional checkpointer (defaults to GLOBAL_CHECKPOINTER)
        store: Optional store (defaults to GLOBAL_STORE)
    """
    # Use provided or default to globals
    checkpointer = checkpointer or GLOBAL_CHECKPOINTER
    store = store or GLOBAL_STORE
    
    # Rest of existing implementation...
```

Remove the lines that create new instances:
```python
# DELETE THESE LINES:
# checkpointer = InMemorySaver()
# store = InMemoryStore()
```

### Phase 1.5: Implement Native LangGraph Interrupt Pattern

#### 1.5.1 Replace Tool Wrapper Approach with Native Interrupts
**File**: `api/supervisor_agent_system.py`

Instead of using the current wrapper approach, implement native LangGraph interrupts using `post_model_hook`. This approach is cleaner and more aligned with LangGraph's design patterns.

Add interrupt handling function:
```python
# Define risky tools that require user confirmation
RISKY_TOOLS = {
    "create_spg_nft_collection",
    "mint_and_register_ip_with_terms", 
    "register_derivative",
    "mint_license_tokens",
    "pay_royalty_on_behalf",
    "raise_dispute",
    "deposit_wip",
    "transfer_wip"
}

def halt_on_risky_tools(state):
    """Post-model hook to interrupt on risky tool calls."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        for tc in last.tool_calls:
            if tc.get("name") in RISKY_TOOLS:
                interrupt_data = {
                    "awaiting": tc["name"], 
                    "args": tc.get("args", {}),
                    "interrupt_id": str(uuid.uuid4()),
                    "tool_name": tc["name"],
                    "timestamp": datetime.now().isoformat()
                }
                _ = interrupt(interrupt_data)
    return {}
```

#### 1.5.2 Update Agent Creation to Use Post-Model Hook
**File**: `api/supervisor_agent_system.py`

Update agent creation functions to use the native interrupt pattern:
```python
# Update imports
from datetime import datetime
import uuid
from langgraph.types import interrupt
from langchain_core.messages import AIMessage

# Update agent creation (example for IP_ASSET_AGENT)
IP_ASSET_AGENT = create_react_agent(
    model="openai:gpt-4.1",
    tools=tool_collections["ip_asset_tool"],  # Direct MCP tools, no wrappers
    checkpointer=checkpointer,
    store=store,
    version="v2",  # Use v2 for post_model_hook support
    post_model_hook=halt_on_risky_tools,  # Native interrupt handling
    prompt=(
        "You are an IP Asset Agent specialized in Story Protocol IP management.\n\n"
        "INSTRUCTIONS:\n"
        "- Handle IP asset creation, registration, and metadata operations\n"
        "- Use mint_and_register_ip_with_terms for creating new IP assets\n"
        "- Use register for registering existing NFTs as IP assets\n"
        "- Handle IPFS uploads and metadata creation\n"
        "- After completing tasks, respond to the supervisor with results\n"
        "- Be precise and include transaction details in responses"
    ),
    name="IP_ASSET_AGENT",
)
```

#### 1.5.3 Benefits of Native Interrupt Approach
- **Cleaner Architecture**: No tool wrappers needed, direct MCP tool usage
- **Better Performance**: Eliminates wrapper overhead
- **Native LangGraph Pattern**: Uses built-in interrupt mechanism
- **Easier Debugging**: Interrupts happen at the graph level, not tool level
- **More Reliable**: LangGraph handles interrupt/resume flow natively

### Phase 2: Update SDK Agent to Use Global Checkpointer

#### 2.1 Import Global Instances
**File**: `api/sdk_mcp_agent.py`

Add import at the top:
```python
from .supervisor_agent_system import create_supervisor_system, GLOBAL_CHECKPOINTER, GLOBAL_STORE
```

#### 2.2 Update Supervisor Creation
**File**: `api/sdk_mcp_agent.py`

In `_run_agent_impl` function, around line 514, update the supervisor creation:
```python
supervisor, supervisor_agents = await create_supervisor_system(
    mcp_tools=tools,
    checkpointer=GLOBAL_CHECKPOINTER,
    store=GLOBAL_STORE
)
```

### Phase 3: Fix Resume Path to Use Same Checkpointer

#### 3.1 Update Resume Function
**File**: `api/supervisor_agent_system.py`

In `resume_interrupted_conversation` function, add MCP tools loading and use globals:

```python
async def resume_interrupted_conversation(
    conversation_id: str,
    interrupt_id: str,
    confirmed: bool,
    wallet_address: Optional[str] = None
):
    """Resume an interrupted conversation after user confirmation."""
    
    import logging
    import asyncio
    from langgraph.types import Command
    from .tools_wrapper import load_sdk_mcp_tools  # Add this import
    
    logger = logging.getLogger(__name__)
    
    logger.info(f"Resuming conversation {conversation_id} with confirmation: {confirmed}")
    
    try:
        # Load fresh MCP tools for the resume session
        mcp_tools = await load_sdk_mcp_tools()
        
        # Create supervisor with same globals used in initial run
        supervisor, _ = await create_supervisor_system(
            mcp_tools=mcp_tools,
            checkpointer=GLOBAL_CHECKPOINTER,
            store=GLOBAL_STORE
        )
        
        # Rest of existing implementation...
```

#### 3.2 Remove Cached Supervisor Usage
**File**: `api/supervisor_agent_system.py`

Remove the line that gets cached supervisor:
```python
# DELETE THIS LINE:
# supervisor = await get_supervisor()
```

### Phase 4: Update Cache Management (Optional Enhancement)

#### 4.1 Update `get_supervisor` Function
**File**: `api/supervisor_agent_system.py`

Modify to accept MCP tools and use globals:
```python
async def get_supervisor(mcp_tools=None):
    """Get the supervisor system, creating it if not cached.
    
    Args:
        mcp_tools: Optional pre-loaded MCP tools
    """
    global _supervisor_cache
    
    # If we have MCP tools, always create fresh to ensure tools are alive
    if mcp_tools:
        supervisor, agents = await create_supervisor_system(
            mcp_tools=mcp_tools,
            checkpointer=GLOBAL_CHECKPOINTER,
            store=GLOBAL_STORE
        )
        return supervisor
    
    # Otherwise use cache
    if _supervisor_cache is None:
        supervisor, agents = await create_supervisor_system(
            checkpointer=GLOBAL_CHECKPOINTER,
            store=GLOBAL_STORE
        )
        _supervisor_cache = {
            "supervisor": supervisor,
            "agents": agents
        }
    return _supervisor_cache["supervisor"]
```

## Testing Plan

### Test Case 1: Basic Interrupt Resume
1. Send request: "help me create spg nft contract collection"
2. Provide parameters when prompted
3. Verify interrupt is triggered with confirmation request
4. Send confirmation via `/interrupt/confirm` endpoint
5. **Expected**: Execution resumes in tool wrapper, completes the NFT creation

### Test Case 2: Cancellation Flow
1. Trigger same interrupt as Test Case 1
2. Send `confirmed: false` to `/interrupt/confirm`
3. **Expected**: Tool wrapper receives cancellation, returns "Operation cancelled" message

### Test Case 3: Multiple Interrupts in Sequence
1. Create a flow that triggers multiple interrupts (e.g., create collection then mint tokens)
2. Confirm each interrupt
3. **Expected**: Each interrupt resumes correctly without restarting

### Test Case 4: Concurrent Conversations
1. Start two different conversations with different thread IDs
2. Trigger interrupts in both
3. Resume them in different order
4. **Expected**: Each conversation maintains its own state correctly

## Monitoring and Logging

### Add Debug Logging
1. Log checkpointer instance ID at creation and resume
2. Log thread_id at each checkpoint save/load
3. Log when tools are created vs reused
4. Log the full checkpoint state before interrupt and after resume

### Success Criteria
- Execution continues from the exact point of interrupt
- Tool wrapper receives the resume response
- Original tool executes after confirmation
- No duplicate agent creation or graph restart

## Alternative Approaches (If Needed)

### Option 1: Persistent Checkpointer
If in-memory approach has issues:
```python
from langgraph.checkpoint.sqlite import SqliteSaver
GLOBAL_CHECKPOINTER = SqliteSaver.from_conn_string("file:checkpoints.db")
```

### Option 2: Redis Checkpointer (Production)
For distributed systems:
```python
from langgraph.checkpoint.redis import RedisCheckpointer
GLOBAL_CHECKPOINTER = RedisCheckpointer(redis_url="redis://localhost:6379")
```

## Rollback Plan

If the fix causes issues:
1. Remove global checkpointer references
2. Revert to creating fresh instances
3. Document the limitation that interrupts don't resume properly
4. Consider implementing a different confirmation pattern (e.g., pre-confirmation before tool execution)

## Timeline

1. **Phase 1**: Global Checkpointer Infrastructure - 15 minutes ✅
2. **Phase 1.5**: Native Interrupt Pattern Implementation - 30 minutes
3. **Phase 2**: SDK Agent Global Checkpointer Integration - 20 minutes  
4. **Phase 3**: Resume Path Implementation - 25 minutes
5. **Phase 4**: Cache Management (Optional) - 15 minutes
6. **Testing**: 30 minutes
7. **Documentation**: 15 minutes
8. **Total**: ~2.5 hours

## Validation from Community Sources

### GitHub Discussion Confirmation
Our root cause analysis is validated by [LangGraph Discussion #4341](https://github.com/langchain-ai/langgraph/discussions/4341), where users report identical symptoms:
- "Upon resuming, the entire node is re-executed" 
- Same multi-agent supervisor architecture
- Same checkpointer separation issue (agent vs supervisor using different `InMemorySaver` instances)
- Issue remains unresolved, confirming this is a real production problem

### MCP + Interrupt Combination Rarity
Web search confirms that **MCP tools with LangGraph interrupts are rarely used together**, making this a bleeding-edge implementation challenge. However, our own codebase (`/langgraph-mcp-agent/`) contains a working example that successfully combines both:
- Uses `async with` MCP client session management
- Extracts tools once and reuses throughout workflow
- Implements proper interrupt handling with `Command(resume=...)`
- Maintains checkpointer for state persistence

### Production Experience Validation  
[Medium article](https://generativeai.pub/when-llms-need-humans-managing-langgraph-interrupts-through-fastapi-97d0912fb6af) confirms:
- HITL implementation "isn't straightforward" in production
- FastAPI integration challenges match our backend setup
- Real-world complexity aligns with our experience

### Architecture Pattern Confirmation
[DEV.to article](https://dev.to/sreeni5018/building-multi-agent-systems-with-langgraph-supervisor-138i) validates:
- Multi-agent supervisor patterns match our setup
- State management complexities are well-documented

## Notes

- The fix maintains backward compatibility
- No API changes required
- Frontend continues to work unchanged
- Can be deployed without user impact
- **High confidence**: Multiple community sources confirm identical symptoms and root causes

## Key Insights from langgraph-mcp-agent Implementation

### Working Pattern Analysis
The `/langgraph-mcp-agent/` codebase demonstrates a successful MCP + interrupt implementation:

1. **Checkpointer Usage**: Uses `MemorySaver` (from `langgraph.checkpoint.memory`) instead of `InMemorySaver`
2. **MCP Session Management**: Keeps MCP session open for entire workflow duration
3. **Interrupt Pattern**: Uses `interrupt()` directly within nodes, not in tool wrappers
4. **Resume Pattern**: Uses `Command(resume={...})` with structured data
5. **Single Graph Instance**: Creates graph once with tools, then streams events

### Critical Success Pattern
```python
# Their working pattern:
async with MultiServerMCPClient() as client:
    # Load tools once
    tools = client.get_tools()
    
    # Create graph with checkpointer
    graph = create_workflow_graph(tools, memory=MemorySaver())
    
    # Stream events and handle interrupts
    async for event in graph.astream(input_data, config, stream_mode="updates"):
        if "__interrupt__" in event:
            # Handle interrupt and resume with same graph instance
            await process_events(Command(resume=...))
```

### Key Differences from Our Implementation
1. **Tool Wrapper vs Node Interrupts**: They interrupt in nodes, we interrupt in tool wrappers
2. **Session Lifecycle**: They keep MCP session alive, we recreate on resume
3. **Checkpointer Type**: They use `MemorySaver`, we use `InMemorySaver`
4. **Graph Reuse**: They reuse same graph instance, we recreate

### Validation for Our Approach
Their success validates that:
- Shared checkpointer across interrupt/resume works
- MCP tools can work with interrupts when managed properly
- `Command(resume=...)` is the correct pattern
- Thread-based state persistence is viable

## Additional Considerations

### Thread Safety
- Both `InMemorySaver` and `MemorySaver` are thread-safe for async operations
- Multiple concurrent conversations will maintain separate states via different thread_ids
- Consider adding mutex/lock if switching to file-based checkpointer

### Memory Management
- In-memory checkpointer will grow with usage
- Consider implementing periodic cleanup of old checkpoints
- Monitor memory usage in production
- SQLite/Redis alternatives provide automatic persistence

### Error Handling
- If checkpointer fails to save, the interrupt will still trigger but resume will fail
- Add try-catch around checkpoint operations with appropriate logging
- Consider fallback behavior if checkpoint is corrupted

### Security
- Thread IDs should be validated to prevent checkpoint hijacking
- Consider adding user authentication to checkpoint access
- Encrypt sensitive data in checkpoints if using persistent storage

## Final Recommendations

Based on our analysis and the working langgraph-mcp-agent example:

1. **Proceed with the Plan**: Our approach of shared checkpointers is validated by both community issues and working code
2. **Use Native Interrupts**: Phase 1.5 implements the cleaner post_model_hook approach instead of tool wrappers
3. **Consider MemorySaver**: The working example uses `MemorySaver` instead of `InMemorySaver` - test both
4. **Future Enhancement**: Consider refactoring to keep MCP sessions alive longer (like langgraph-mcp-agent does)
5. **Monitor Performance**: Track checkpoint memory usage and resume latency after implementation
6. **Add Logging**: Implement comprehensive logging for checkpointer operations to aid debugging

**Confidence Level**: ⭐⭐⭐⭐⭐ (5/5) - Plan is thoroughly validated by multiple sources
