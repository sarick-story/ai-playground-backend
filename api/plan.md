# Implementation Plan: Global Supervisor with Stateless Tool Wrappers

## Problem Statement

When LangGraph interrupts are triggered, the MCP session gets closed, causing `ClosedResourceError` when resuming from interrupts. This happens because:

1. MCP sessions are created within `async with` context managers that close when the function returns
2. When an interrupt occurs, the function returns with `{"status": "interrupted"}`
3. The context manager exits, closing the MCP session
4. The cached supervisor still has agents with tools that reference the closed session
5. Resuming fails because tools try to use the closed session

## Solution Overview

Implement stateless MCP tool wrappers with a global supervisor, eliminating session persistence issues while maintaining conversation continuity.

### Architecture Diagram

```
Application Start
      ↓
Initialize Global Supervisor (Once)
      ↓
Load Tool Metadata (Cached)
      ↓
Create Stateless Tool Wrappers
      ↓
Bind Tools to Agents
      ↓
[Ready for Requests]
      ↓
Request → Thread ID → Supervisor → Agent → Tool Wrapper
                                              ↓
                                    [Fresh MCP Session per Call]
                                              ↓
                                        Execute Tool
                                              ↓
                                      [Session Auto-Closes]
```

### Detailed Implementation Plan

#### Phase 1: Create Stateless Tool Infrastructure
**New File: `api/mcp_tools_stateless.py`**

```python
"""
Stateless MCP tool wrappers that reconnect on each call.
Solves session closure issues during interrupts.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from collections import defaultdict
from langchain_core.tools import tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
import logging

logger = logging.getLogger(__name__)

# Global caches
TOOL_METADATA_CACHE: Optional[List[Dict[str, Any]]] = None
ENV_VARS_CACHE: Optional[dict] = None
TOOL_LOCKS = defaultdict(asyncio.Lock)  # Prevent concurrent sessions per tool

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 0.5

async def get_mcp_tool_metadata() -> List[Dict[str, Any]]:
    """
    Get tool metadata without keeping session open.
    Fetches once and caches for application lifetime.
    """
    global TOOL_METADATA_CACHE
    
    if TOOL_METADATA_CACHE is not None:
        return TOOL_METADATA_CACHE
    
    server_path = _find_mcp_server_path()
    server_params = StdioServerParameters(
        command="python3",
        args=[server_path],
        env=_get_env_vars()
    )
    
    # Quick connection to get metadata only
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            
            TOOL_METADATA_CACHE = [
                {
                    'name': t.name,
                    'description': t.description or '',
                    'schema': getattr(t, 'args_schema', {})
                }
                for t in tools
            ]
    
    logger.info(f"Cached metadata for {len(TOOL_METADATA_CACHE)} MCP tools")
    return TOOL_METADATA_CACHE

def create_stateless_tool_wrapper(tool_name: str, tool_description: str):
    """
    Create a stateless tool that reconnects on each call.
    Includes retry logic and concurrency control.
    """
    
    @tool(name=tool_name, description=tool_description)
    async def stateless_tool(**kwargs):
        """Execute tool with fresh MCP session."""
        
        last_error = None
        
        for attempt in range(MAX_RETRIES):
            try:
                # Lock per tool to prevent concurrent stdio sessions
                async with TOOL_LOCKS[tool_name]:
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
                                    logger.debug(f"Executing {tool_name} with args: {kwargs}")
                                    result = await t.ainvoke(kwargs)
                                    logger.debug(f"Tool {tool_name} completed successfully")
                                    return result
                            
                            raise ValueError(f"Tool {tool_name} not found in MCP server")
                            
            except Exception as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Tool {tool_name} failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Tool {tool_name} failed after {MAX_RETRIES} attempts: {e}")
                    raise last_error
    
    return stateless_tool

async def create_stateless_mcp_tools() -> List:
    """
    Create all MCP tools as stateless wrappers.
    Main entry point for supervisor system.
    """
    try:
        metadata = await get_mcp_tool_metadata()
        
        tools = []
        for meta in metadata:
            wrapper = create_stateless_tool_wrapper(
                meta['name'], 
                meta['description']
            )
            tools.append(wrapper)
        
        logger.info(f"Created {len(tools)} stateless MCP tools")
        return tools
        
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
```

#### Phase 2: Modify Global Supervisor System
**File: `api/supervisor_agent_system.py`**

**Key Changes:**

1. **Add import at top of file:**
```python
from .mcp_tools_stateless import create_stateless_mcp_tools
```

2. **Remove these functions entirely:**
```python
# DELETE THESE FUNCTIONS:
# - load_fresh_mcp_tools()
# - get_or_create_mcp_tools() 
# - _find_mcp_server_path() (now in mcp_tools_stateless.py)
```

3. **Replace `create_supervisor_system()` with:**
```python
async def create_supervisor_system(mcp_tools=None):
    """Create the complete supervisor system with all agents."""
    global GLOBAL_SUPERVISOR_SYSTEM
    
    # Skip if already initialized (singleton pattern)
    if GLOBAL_SUPERVISOR_SYSTEM is not None:
        logger.info("Using existing global supervisor system")
        return GLOBAL_SUPERVISOR_SYSTEM
    
    logger.info("Creating new global supervisor system")
    
    # Create checkpointer and store (shared across all conversations)
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    
    # Use provided tools or create stateless ones
    if mcp_tools is None:
        logger.info("Creating stateless MCP tools")
        mcp_tools = await create_stateless_mcp_tools()
        
        if not mcp_tools:
            logger.warning("No MCP tools loaded, supervisor will have limited functionality")
    
    # Create all agents with loaded tools
    agents = await create_all_agents(mcp_tools)
    
    # Create supervisor with all agents
    supervisor = create_supervisor(
        model=init_chat_model("openai:gpt-4.1"),
        agents=list(agents.values()),
        prompt=(
            "You are a supervisor managing Story Protocol specialized agents:\n\n"
            "AVAILABLE AGENTS:\n"
            "- IP_ASSET_AGENT: Creates, registers IP assets, handles metadata and IPFS\n"
            "- IP_ACCOUNT_AGENT: Manages ERC20 tokens, checks balances, mints test tokens\n"
            "- IP_LICENSE_AGENT: Handles licensing, mints license tokens, attaches terms\n"
            "- NFT_CLIENT_AGENT: Creates SPG collections, manages contract fees\n"
            "- DISPUTE_AGENT: Raises disputes, handles dispute processes\n"
            "- ROYALTY_AGENT: Processes royalty payments, claims revenue\n"
            "- WIP_AGENT: Handles WIP token operations (wrapping, transfers)\n"
            "- GROUP_AGENT: Group operations (limited functionality)\n"
            "- PERMISSION_AGENT: Permission management (limited functionality)\n\n"
            "INSTRUCTIONS:\n"
            "- Analyze user requests and route to the most appropriate specialist agent\n"
            "- Assign work to ONE agent at a time, do not call agents in parallel\n"
            "- For complex workflows, coordinate between agents sequentially\n"
            "- Do not perform any Story Protocol operations yourself\n"
            "- If one agent is not able to handle the request, hand off to the next agent\n"
            "- The user is not aware of the different specialized agent assistants, so do not mention them"
        ),
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile(checkpointer=checkpointer, store=store)

    GLOBAL_SUPERVISOR_SYSTEM = supervisor
    logger.info("Global supervisor system initialized successfully")
    
    return supervisor
```

4. **Update `get_supervisor_from_cache()`:**
```python
async def get_supervisor_from_cache():
    """Get or create the global supervisor system."""
    global GLOBAL_SUPERVISOR_SYSTEM
    
    if GLOBAL_SUPERVISOR_SYSTEM is None:
        logger.info("Global supervisor not found, creating new one")
        await create_supervisor_system()
    
    return GLOBAL_SUPERVISOR_SYSTEM
```

5. **Simplify `resume_interrupted_conversation()`:**
```python
async def resume_interrupted_conversation(
    conversation_id: str,
    interrupt_id: str,
    confirmed: bool,
    wallet_address: Optional[str] = None
):
    """Resume an interrupted conversation after user confirmation."""
    
    logger.info(f"🔍 RESUME: Starting resume for conversation {conversation_id} with confirmation: {confirmed}")
    
    try:
        # Get the global supervisor (same instance for all)
        supervisor = await get_supervisor_from_cache()
        
        # Create thread config to resume from checkpoint
        thread_config = {
            "configurable": {
                "thread_id": conversation_id,
                "wallet_address": wallet_address
            }
        }
        logger.info(f"🔍 RESUME: Thread config: {thread_config}")
        
        # Resume with Command
        result = await asyncio.wait_for(
            supervisor.ainvoke(
                Command(resume=confirmed),
                config=thread_config
            ),
            timeout=30.0
        )
        
        logger.info(f"🔄 RESUME COMPLETED for {conversation_id}")
        
        # Extract last AI message content
        last_ai_content = None
        if isinstance(result, dict) and 'messages' in result:
            messages = result['messages']
            for msg in reversed(messages):
                if hasattr(msg, '__class__') and 'AI' in msg.__class__.__name__:
                    if hasattr(msg, 'content'):
                        last_ai_content = msg.content
                    break
        
        if confirmed:
            logger.info(f"Conversation resumed successfully: {conversation_id}")
            return {
                "status": "completed", 
                "message": last_ai_content,
                "conversation_id": conversation_id
            }
        else:
            logger.info(f"Conversation cancelled by user: {conversation_id}")
            return {
                "status": "cancelled", 
                "message": last_ai_content,
                "conversation_id": conversation_id
            }
            
    except asyncio.TimeoutError:
        logger.error(f"Timeout resuming conversation {conversation_id} after 30 seconds")
        return {"status": "error", "error": "Resume operation timed out"}
    except Exception as e:
        logger.error(f"Error resuming conversation {conversation_id}: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {"status": "error", "error": str(e)}
```

#### Phase 3: Simplify SDK MCP Agent
**File: `api/sdk_mcp_agent.py`**

**Major Simplification - Remove most MCP session code:**

1. **Remove these functions entirely:**
```python
# DELETE THESE FUNCTIONS:
# - _run_agent_impl()
# - close_mcp_session()
# - cleanup_session()
# - All MCP session management in _run_agent_impl()
```

2. **Replace `run_agent()` with simplified version:**
```python
async def run_agent(
    user_message: str,
    wallet_address: Optional[str] = None,
    queue: Optional[asyncio.Queue] = None,
    conversation_id: Optional[str] = None,
    message_history: Optional[List[Dict[str, str]]] = None
):
    """Run the SDK MCP agent using global supervisor."""
    logger.info(f"Starting SDK MCP agent with message: {user_message}")
    
    # Add validation for wallet address
    if wallet_address == "none" or wallet_address == "null" or wallet_address == "undefined":
        logger.warning(f"Invalid wallet address '{wallet_address}' detected, setting to None")
        wallet_address = None
    
    logger.info(f"Using wallet address: {wallet_address or 'None'}")
    
    # Handle direct transaction parsing (keep existing logic)
    if queue and wallet_address and ("send" in user_message.lower() and "ip to" in user_message.lower()):
        logger.info("Detected direct transaction command, attempting to parse")
        
        try:
            import re
            pattern = r"send\s+([0-9.]+)\s+ip\s+to\s+(0x[a-fA-F0-9]+)"
            match = re.search(pattern, user_message, re.IGNORECASE)
            
            if not match:
                pattern = r"send\s+([0-9.]+)\s+ip\s+to\s+(.*?)($|\s|\.)"
                match = re.search(pattern, user_message, re.IGNORECASE)
            
            if match:
                amount = match.group(1)
                to_address = match.group(2).strip()
                
                try:
                    to_address = validate_and_format_address(to_address)
                    logger.info(f"Validated address: {to_address}")
                    
                    try:
                        amount_float = float(amount)
                        if amount_float <= 0:
                            await queue.put("Error: Amount must be greater than zero.")
                            return {"error": "Amount must be greater than zero"}
                    except ValueError:
                        await queue.put(f"Error: Invalid amount format: {amount}")
                        return {"error": f"Invalid amount format: {amount}"}
                
                    logger.info(f"Parsed direct transaction: {amount} IP to {to_address}")
                    
                    # Try direct transaction
                    await queue.put(f"Creating transaction to send {amount} IP to {to_address}...")
                    success = await create_transaction_request(to_address, amount, queue)
                    
                    if success:
                        logger.info("Direct transaction request sent successfully")
                        return {"success": "Transaction request sent"}
                    else:
                        logger.warning("Failed to send direct transaction, attempting fallback to agent")
                        
                except ValueError as e:
                    error_msg = str(e)
                    logger.error(f"Address validation failed: {error_msg}")
                    await queue.put(f"Error: {error_msg}")
                    return {"error": error_msg}
        except Exception as e:
            logger.error(f"Error in direct transaction handling: {str(e)}")
            # Continue with agent flow as fallback
    
    # Use global supervisor for all other requests
    try:
        from .supervisor_agent_system import get_supervisor_from_cache
        
        # Get the global supervisor
        supervisor = await get_supervisor_from_cache()
        
        # Prepare messages
        messages = message_history or [{"role": "user", "content": user_message}]
        
        # Create config with conversation ID as thread ID
        config = {
            "configurable": {
                "thread_id": conversation_id or str(uuid.uuid4()),
                "wallet_address": wallet_address
            }
        }
        
        if queue:
            # Stream the response
            try:
                async for chunk in supervisor.astream(messages, config):
                    # Handle different chunk types
                    if isinstance(chunk, dict):
                        # Handle interrupt status
                        if chunk.get("status") == "interrupted":
                            logger.info(f"Agent returned interrupt status")
                            interrupt_info = chunk.get("interrupt_data", {})
                            interrupt_message = f"__INTERRUPT_START__{json.dumps(interrupt_info)}__INTERRUPT_END__"
                            await queue.put(interrupt_message)
                            return chunk
                        
                        # Handle regular messages in chunk
                        if "messages" in chunk:
                            for msg in chunk["messages"]:
                                if hasattr(msg, 'content') and msg.content:
                                    await queue.put(msg.content)
                    elif isinstance(chunk, str) and chunk.strip():
                        await queue.put(chunk)
                
                # Signal completion
                await queue.put({"done": True})
                logger.info("Streaming completed successfully")
                return {"success": "Streaming completed"}
                
            except Exception as e:
                logger.error(f"Error during streaming: {str(e)}")
                await queue.put(f"\nError: {str(e)}\n")
                await queue.put({"done": True})
                raise
        else:
            # Non-streaming response
            result = await supervisor.ainvoke(messages, config)
            logger.info("Non-streaming response completed")
            return result
            
    except Exception as e:
        logger.error(f"Error in run_agent: {str(e)}")
        logger.error(traceback.format_exc())
        if queue:
            try:
                await queue.put(f"\nError: {str(e)}\n")
                await queue.put({"done": True})
            except Exception as e2:
                logger.error(f"Error sending error notification: {str(e2)}")
        raise
```

3. **Keep these functions unchanged:**
```python
# KEEP AS-IS:
# - create_transaction_request()
# - is_valid_ethereum_address()
# - validate_and_format_address()
# - system_prompt (at module level)
```

#### Phase 4: Update Chat Handler
**File: `api/chat.py`**

**Minor Updates - Simplify to use global supervisor:**

1. **Update `stream_agent_response()` function:**
```python
async def stream_agent_response(
    messages: List[Message], 
    conversation_id: Optional[str] = None,
    mcp_type: str = "storyscan",
    wallet_address: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """Stream the agent's response using global supervisor for both MCP types."""
    
    # Get the last user message
    last_message = messages[-1].content if messages and messages[-1].role == "user" else ""
    
    logger.info(f"Processing user message with {mcp_type} MCP: {last_message[:50]}...")
    
    if not last_message:
        logger.warning("No user message found")
        yield "No user message found.\n"
        return
    
    # Create a queue for streaming
    queue = asyncio.Queue()
    
    # Start the agent task
    async def run_agent_task():
        try:
            # Pass all messages to maintain conversation context
            formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            
            # Both SDK and storyscan now use the same global supervisor
            if mcp_type == "sdk":
                logger.info(f"Using global supervisor for SDK with wallet: {wallet_address}")
                # Use SDK agent for transaction parsing, but it will call global supervisor
                result = await run_sdk_agent(
                    last_message, 
                    wallet_address=wallet_address,
                    queue=queue, 
                    conversation_id=conversation_id, 
                    message_history=formatted_messages
                )
            else:
                logger.info("Using global supervisor for storyscan")
                # Directly use global supervisor for storyscan
                from .supervisor_agent_system import get_supervisor_from_cache
                supervisor = await get_supervisor_from_cache()
                
                config = {
                    "configurable": {
                        "thread_id": conversation_id,
                        "wallet_address": wallet_address,
                        "mcp_type": mcp_type
                    }
                }
                
                # Stream using supervisor
                async for chunk in supervisor.astream(formatted_messages, config):
                    if isinstance(chunk, dict):
                        if chunk.get("status") == "interrupted":
                            interrupt_info = chunk.get("interrupt_data", {})
                            interrupt_message = f"__INTERRUPT_START__{json.dumps(interrupt_info)}__INTERRUPT_END__"
                            await queue.put(interrupt_message)
                            return
                        
                        if "messages" in chunk:
                            for msg in chunk["messages"]:
                                if hasattr(msg, 'content') and msg.content:
                                    await queue.put(msg.content)
                    elif isinstance(chunk, str) and chunk.strip():
                        await queue.put(chunk)
                
                result = {"success": "Streaming completed"}
            
            # Handle interrupt responses properly
            if isinstance(result, dict) and result.get("status") == "interrupted":
                logger.info(f"Agent returned interrupt status: {result.get('interrupt_data', {}).get('interrupt_id', 'unknown')}")
                # Send interrupt information to frontend
                interrupt_info = result.get("interrupt_data", {})
                interrupt_message = f"__INTERRUPT_START__{json.dumps(interrupt_info)}__INTERRUPT_END__"
                await queue.put(interrupt_message)
                # Don't send done=True - wait for frontend confirmation
                return
                
        except Exception as e:
            logger.error(f"Error in run_agent_task: {str(e)}")
            await queue.put({"error": str(e)})
        finally:
            await queue.put(None)
    
    agent_task = asyncio.create_task(run_agent_task())
    
    try:
        while True:
            item = await queue.get()
            
            if item is None:
                logger.info("End of stream")
                break
                
            if isinstance(item, dict) and "error" in item:
                yield f"Error: {item['error']}"
                break
            
            if isinstance(item, str):
                # Skip any JSON-like structures, but allow normal text through
                if not (item.startswith('{') or item.startswith('e:{')):
                    yield item
    
    except Exception as e:
        logger.error(f"Error in stream_agent_response: {str(e)}")
        yield f"Error: {str(e)}"
            
    finally:
        if not agent_task.done():
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass
```

2. **Keep `handle_interrupt_confirmation()` unchanged:**
```python
# NO CHANGES NEEDED - function will work better now since:
# - Uses same global supervisor 
# - No MCP session errors to handle
# - resume_interrupted_conversation() is simplified
```

3. **Keep other functions unchanged:**
```python
# KEEP AS-IS:
# - handle_chat()
# - handle_transaction() 
# - All helper functions (is_transaction_intent, parse_transaction_details)
```

### Key Design Decisions

1. **Why Global Supervisor?**
   - Single checkpointer maintains all states
   - Thread isolation via thread_id
   - Memory efficient
   - Simpler than per-conversation caching

2. **Why Stateless Tools?**
   - No session state to lose
   - Interrupts can't break sessions
   - Simple error recovery
   - Thread-safe by design

3. **Why Thread ID per Conversation?**
   - LangGraph built-in isolation
   - Each conversation has separate history
   - Concurrent users don't conflict
   - Standard LangGraph pattern

### Implementation Safety Features

1. **Concurrency Control**
   - Per-tool locks prevent stdio conflicts
   - Thread isolation via thread_id
   - Async-safe caching

2. **Error Handling**
   - 3-retry attempts with backoff
   - Graceful degradation without tools
   - Comprehensive logging

3. **Performance Optimization**
   - Metadata cached once
   - Environment variables cached
   - Supervisor created once

### Expected Outcomes

✅ **Interrupts Work Correctly**
- No persistent sessions to break
- Resume uses same supervisor/checkpointer

✅ **Multi-User Safety**
- Thread isolation per conversation
- No shared mutable state
- Concurrent requests handled properly

✅ **Resource Efficiency**
- One supervisor for all users
- Minimal memory footprint
- No session leak potential

✅ **Maintainability**
- Simpler architecture
- Less code to maintain
- Clear separation of concerns

### Performance Considerations

**Overhead per Tool Call:**
- Session creation: ~50-100ms
- Tool lookup: ~10-20ms
- Total: ~100-300ms acceptable for Story Protocol operations

**Mitigation Strategies:**
- Metadata caching reduces lookups
- Environment caching reduces I/O
- Lock granularity minimizes contention

### Testing Strategy

1. **Unit Tests:**
   - Stateless tool wrapper creation
   - Retry logic verification
   - Lock behavior validation

2. **Integration Tests:**
   - Basic tool execution
   - Interrupt → Confirm → Resume
   - Interrupt → Cancel → Continue

3. **Concurrency Tests:**
   - Multiple users simultaneously
   - Rapid sequential requests
   - Thread isolation verification

4. **Failure Tests:**
   - MCP server unavailable
   - Tool execution errors
   - Network interruptions

5. **Performance Tests:**
   - Measure tool call latency
   - Monitor memory usage
   - Check for resource leaks



