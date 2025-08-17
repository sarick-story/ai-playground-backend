from langgraph.prebuilt import create_react_agent
from .tools_wrapper import create_wrapped_tool_collections, create_wrapped_tool_collections_from_tools
from .interrupt_handler import create_transaction_interrupt, send_standard_interrupt

from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from typing import Optional, Tuple, List, Any, Dict
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import interrupt
from datetime import datetime
import uuid
import os
import sys
import json
import asyncio
import time
import traceback
from asyncio import Queue, Lock
import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Conversation-specific checkpointers for proper isolation
_conversation_checkpointers: Dict[str, InMemorySaver] = {}
_conversation_stores: Dict[str, InMemoryStore] = {}

# Global supervisor system cache
GLOBAL_SUPERVISOR_SYSTEM = None

def get_conversation_checkpointer(conversation_id: str) -> InMemorySaver:
    """Get or create a checkpointer for a specific conversation."""
    if conversation_id not in _conversation_checkpointers:
        logger.info(f"🔄 Creating new checkpointer for conversation: {conversation_id}")
        _conversation_checkpointers[conversation_id] = InMemorySaver()
    else:
        logger.info(f"🔄 Reusing existing checkpointer for conversation: {conversation_id}")
    return _conversation_checkpointers[conversation_id]

def get_conversation_store(conversation_id: str) -> InMemoryStore:
    """Get or create a store for a specific conversation."""
    if conversation_id not in _conversation_stores:
        logger.info(f"🔄 Creating new store for conversation: {conversation_id}")
        _conversation_stores[conversation_id] = InMemoryStore()
    else:
        logger.info(f"🔄 Reusing existing store for conversation: {conversation_id}")
    return _conversation_stores[conversation_id]

def cleanup_conversation_state(conversation_id: str):
    """Clean up checkpointer and store for a conversation."""
    if conversation_id in _conversation_checkpointers:
        logger.info(f"🧹 Cleaning up checkpointer for conversation: {conversation_id}")
        del _conversation_checkpointers[conversation_id]
    if conversation_id in _conversation_stores:
        logger.info(f"🧹 Cleaning up store for conversation: {conversation_id}")
        del _conversation_stores[conversation_id]

# Legacy global references (DEPRECATED - use conversation-specific ones)
GLOBAL_CHECKPOINTER = InMemorySaver()  # Keep for backward compatibility
GLOBAL_STORE = InMemoryStore()  # Keep for backward compatibility

# Note: Consider using MemorySaver instead of InMemorySaver based on 
# successful langgraph-mcp-agent implementation:
# from langgraph.checkpoint.memory import MemorySaver
# GLOBAL_CHECKPOINTER = MemorySaver()

@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers together.

    Args:
        a: The first number to multiply
        b: The second number to multiply

    Returns:
        The product of the two numbers
    """
    return a * b

RISKY_TOOLS_MATH = {
    "multiply"
}

def halt_on_risky_tools_math(state: Dict[str, Any]) -> Dict[str, Any]:
    """Post-model hook to interrupt on risky tool calls using standardized format."""
    messages = state.get("messages", [])
    logger.info(f"🔍 POST_HOOK: Called with {len(messages)} messages in state")
    logger.info(f"🔍 POST_HOOK: State keys: {list(state.keys())}")
    
    if not messages:
        logger.info("🔍 POST_HOOK: No messages, returning empty")
        return {}
        
    last = messages[-1]
    logger.info(f"🔍 POST_HOOK: Last message type: {type(last).__name__}")
    
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        logger.info(f"🔍 POST_HOOK: Found {len(last.tool_calls)} tool calls")
        for tc in last.tool_calls:
            tool_name = tc.get("name", "")
            logger.info(f"🔍 POST_HOOK: Processing tool call: {tool_name}")
            if tool_name in RISKY_TOOLS_MATH:
                # Get tool arguments
                tool_args = tc.get("args", {})
                
                logger.info(f"🔍 PRE-INTERRUPT: State has {len(messages)} messages before interrupt")
                logger.info(f"🔍 PRE-INTERRUPT: Tool args: {tool_args}")
                logger.info(f"🔍 PRE-INTERRUPT: Tool call ID: {tc.get('id', 'unknown')}")
                
                # Create standardized interrupt using interrupt_handler
                interrupt_msg = create_transaction_interrupt(
                    tool_name=tool_name,
                    operation=f"Execute {tool_name}",
                    parameters=tool_args,
                    network="Story Protocol"
                )
                
                # Send interrupt with proper format
                logger.info(f"fuck interrupt")
                response = send_standard_interrupt(interrupt_msg)

                logger.info(f"love resume")
                logger.info(f"🔍 POST-INTERRUPT: Resume response: {response}")

                if response:
                    logger.info(f"confirmed = True")
                    logger.info(f"🔍 CONFIRMED: Tool {tool_name} approved, allowing execution")
                    return {}
                logger.info(f"confirmed = False")
                logger.info(f"🔍 CANCELLED: Tool {tool_name} cancelled, injecting cancel message")
                tool_messages = ToolMessage(
                    content="Cancelled by human. Continue without executing that tool and provide next step.",
                    tool_call_id=tc["id"],
                    name=tool_name
                )
                
                return {"messages": [tool_messages]}
    
    logger.info("🔍 POST_HOOK: No risky tools found, continuing")
    return {}


def create_math_agent(conversation_id: str):
    """Create a Math agent with conversation-specific checkpointer."""
    checkpointer = get_conversation_checkpointer(conversation_id)
    store = get_conversation_store(conversation_id)
    
    logger.info(f"🔄 Creating Math_agent for conversation: {conversation_id}")
    logger.info(f"🔄 Math_agent checkpointer type: {type(checkpointer).__name__}")
    logger.info(f"🔄 Math_agent checkpointer instance ID: {id(checkpointer)}")
    logger.info(f"🔄 Math_agent store type: {type(store).__name__}")
    logger.info(f"🔄 Math_agent store instance ID: {id(store)}")
    
    agent = create_react_agent(
        model="openai:gpt-4.1",
        prompt= ("Please only call one tool at a time. Do not call multiple tools at once."),
        tools=[multiply],
        checkpointer=checkpointer,
        store=store,
        post_model_hook=halt_on_risky_tools_math,
        version="v2",
    )
    
    return agent

# Legacy global Math_agent (DEPRECATED - use create_math_agent instead)
Math_agent = create_react_agent(
    model="openai:gpt-4.1",
    tools=[multiply],
    checkpointer=GLOBAL_CHECKPOINTER,
    post_model_hook=halt_on_risky_tools_math,
    version="v2",
)







# Global variables for MCP tools caching
_mcp_tools: List[Any] | None = None
_last_used: float = 0.0
_lock = asyncio.Lock()





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


def halt_on_risky_tools(state: Dict[str, Any]) -> Dict[str, Any]:
    """Post-model hook to interrupt on risky tool calls using standardized format."""
    messages = state.get("messages", [])
    if not messages:
        return {}
        
    last = messages[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        for tc in last.tool_calls:
            tool_name = tc.get("name", "")
            if tool_name in RISKY_TOOLS:
                # Get tool arguments
                tool_args = tc.get("args", {})
                
                # Create standardized interrupt using interrupt_handler
                interrupt_msg = create_transaction_interrupt(
                    tool_name=tool_name,
                    operation=f"Execute {tool_name}",
                    parameters=tool_args,
                    network="Story Protocol"
                )
                
                # Send interrupt with proper format
                logger.info(f"fuck interrupt")
                response = send_standard_interrupt(interrupt_msg)

                logger.info(f"love resume")

                if response:
                    logger.info(f"confirmed = True")
                    return {}
                logger.info(f"confirmed = False")
                tool_messages = ToolMessage(
                    content="Cancelled by human. Continue without executing this tool.",
                    tool_call_id=tc["id"],
                    name=tool_name
                )
                
                return {"messages": [tool_messages]}
    
    return {}


async def get_or_create_mcp_tools() -> List[Any]:
    """Get or create MCP tools using proper MCP client library with caching.
    
    Returns cached tools if available, otherwise loads fresh tools from MCP server.
    """
    global _mcp_tools, _last_used
    
    async with _lock:
        current_time = time.time()
        
        # Reuse existing tools if they were loaded recently (within 5 minutes)
        if _mcp_tools is not None and current_time - _last_used < 300:
            logger.info("Reusing existing MCP tools")
            _last_used = current_time
            return _mcp_tools
            
        # Find the server path
        server_path = os.environ.get("SDK_MCP_SERVER_PATH")
        if not server_path:
            # Try multiple possible locations for server.py
            possible_paths = [
                # Relative path from parent directory
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                             "story-mcp-hub/story-sdk-mcp/server.py"),
                # Direct path based on workspace structure
                "/Users/sarickshah/Documents/story/story-mcp-hub/story-sdk-mcp/server.py",
                # Current directory
                os.path.join(os.getcwd(), "story-mcp-hub/story-sdk-mcp/server.py"),
                # One level up
                os.path.join(os.path.dirname(os.getcwd()), "story-mcp-hub/story-sdk-mcp/server.py"),
            ]
            
            # Try each path
            for path in possible_paths:
                logger.info(f"Checking for server.py at: {path}")
                if os.path.exists(path):
                    server_path = path
                    logger.info(f"Found server.py at: {server_path}")
                    break
            
            if not server_path:
                error_msg = "Could not find server.py in any of the expected locations"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
        
        logger.info(f"SDK MCP Server path: {server_path}")
        
        if not os.path.exists(server_path):
            error_msg = f"SDK MCP server file not found at {server_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Clear existing cached data to load fresh tools
        if _mcp_tools is not None:
            logger.info("Clearing cached MCP tools to load fresh ones")
            _mcp_tools = None
        
        try:
            logger.info(f"Creating MCP session with server: {server_path}")
            
            # Use standard MCP approach with stdio_client (same as tools_wrapper.py)
            server_params = StdioServerParameters(
                command=sys.executable,
                args=[server_path]
            )
            
            # Use stdio_client context manager to handle session lifecycle properly
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    logger.info("MCP session initialized successfully")
                    
                    # Load tools using langchain_mcp_adapters (no custom wrapper needed)
                    tools = await load_mcp_tools(session)
                    logger.info(f"Loaded {len(tools)} tools from MCP server")
                    
                    # Store tools for reuse (sessions are context-managed)
                    _mcp_tools = tools
                    _last_used = time.time()
                    
                    return tools
            
        except Exception as e:
            error_msg = f"Error creating MCP session: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # Clean up cached data
            _mcp_tools = None
            
            raise RuntimeError(f"Failed to create MCP session: {str(e)}")

async def create_all_agents(checkpointer=None, store=None, mcp_tools=None):
    """Create all agents with properly loaded tools.
    
    Args:
        checkpointer: Optional checkpointer for state persistence
        store: Optional store for cross-thread memory
        mcp_tools: Optional pre-loaded MCP tools to use instead of loading separately
    """
    # For Phase 1.5: Use direct MCP tools instead of wrapped tool collections
    # This enables native interrupt handling via post_model_hook
    if mcp_tools:
        # Use provided MCP tools directly - no wrapper needed
        direct_tools = mcp_tools
    else:
        # Load MCP tools directly using the caching method
        direct_tools = await get_or_create_mcp_tools()
    
    # Create tool collections using direct tools (maintain categorization for now)
    tool_collections = {
        "ip_asset_tool": direct_tools,      # All agents get access to all tools
        "ip_account_tool": direct_tools,    # This allows flexible routing and
        "license_tool": direct_tools,       # eliminates the need for complex
        "nft_client_tool": direct_tools,    # tool categorization logic
        "dispute_tool": direct_tools,
        "group_tool": direct_tools,
        "permission_tool": direct_tools,
        "royalty_tool": direct_tools,
        "wip_tool": direct_tools
    }
    
    # Create IP Asset Agent
    IP_ASSET_AGENT = create_react_agent(
        model="openai:gpt-4.1",
        tools=tool_collections["ip_asset_tool"],
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

    # Create IP Account Agent  
    IP_ACCOUNT_AGENT = create_react_agent(
        model="openai:gpt-4.1",
        tools=tool_collections["ip_account_tool"],
        checkpointer=checkpointer,
        store=store,
        version="v2",  # Use v2 for post_model_hook support
        post_model_hook=halt_on_risky_tools,  # Native interrupt handling
        prompt=(
            "You are an IP Account Agent for Story Protocol account management.\n\n"
            "INSTRUCTIONS:\n"
            "- Handle ERC20 token operations and account balances\n"
            "- Check token balances and mint test tokens when needed\n"
            "- Provide account status and token information\n"
            "- After completing tasks, respond to the supervisor with results\n"
            "- Include specific balance amounts and token addresses"
        ),
        name="IP_ACCOUNT_AGENT",
    )

    # Create License Agent
    IP_LICENSE_AGENT = create_react_agent(
        model="openai:gpt-4.1",
        tools=tool_collections["license_tool"],
        checkpointer=checkpointer,
        store=store,
        version="v2",  # Use v2 for post_model_hook support
        post_model_hook=halt_on_risky_tools,  # Native interrupt handling
        prompt=(
            "You are a License Agent for Story Protocol licensing operations.\n\n"
            "INSTRUCTIONS:\n"
            "- Handle license terms retrieval and license token minting\n"
            "- Attach license terms to IP assets\n"
            "- Check license fees and revenue sharing before minting\n"
            "- Explain licensing implications to users\n"
            "- After completing tasks, respond to the supervisor with results\n"
            "- Include license terms details and costs in responses"
        ),
        name="IP_LICENSE_AGENT",
    )

    # Create NFT Client Agent
    NFT_CLIENT_AGENT = create_react_agent(
        model="openai:gpt-4.1",
        tools=tool_collections["nft_client_tool"],
        checkpointer=checkpointer,
        store=store,
        version="v2",  # Use v2 for post_model_hook support
        post_model_hook=halt_on_risky_tools,  # Native interrupt handling
        prompt=(
            "You are an NFT Client Agent for Story Protocol NFT collection management.\n\n"
            "INSTRUCTIONS:\n"
            "- Create SPG NFT collections with custom settings\n"
            "- Check SPG contract minting fees and tokens\n"
            "- Configure collection parameters (mint fees, tokens, settings)\n"
            "- After completing tasks, respond to the supervisor with results\n"
            "- Include collection addresses and fee information"
        ),
        name="NFT_CLIENT_AGENT",
    )

    # Create Dispute Agent
    DISPUTE_AGENT = create_react_agent(
        model="openai:gpt-4.1",
        tools=tool_collections["dispute_tool"],
        checkpointer=checkpointer,
        store=store,
        version="v2",  # Use v2 for post_model_hook support
        post_model_hook=halt_on_risky_tools,  # Native interrupt handling
        prompt=(
            "You are a Dispute Agent for Story Protocol dispute management.\n\n"
            "INSTRUCTIONS:\n"
            "- Handle dispute raising against IP assets\n"
            "- Explain dispute processes and requirements\n"
            "- Process dispute bonds and evidence\n"
            "- After completing tasks, respond to the supervisor with results\n"
            "- Include dispute IDs and bond amounts in responses"
        ),
        name="DISPUTE_AGENT",
    )

    # Create Group Agent (placeholder for future group operations)
    GROUP_AGENT = create_react_agent(
        model="openai:gpt-4.1",
        tools=tool_collections["group_tool"],
        checkpointer=checkpointer,
        store=store,
        version="v2",  # Use v2 for post_model_hook support
        post_model_hook=halt_on_risky_tools,  # Native interrupt handling
        prompt=(
            "You are a Group Agent for Story Protocol group operations.\n\n"
            "INSTRUCTIONS:\n"
            "- Handle group-related operations (currently no tools available)\n"
            "- Inform users about group functionality status\n"
            "- After completing tasks, respond to the supervisor with results"
        ),
        name="GROUP_AGENT",
    )

    # Create Permission Agent (placeholder for future permission operations)
    PERMISSION_AGENT = create_react_agent(
        model="openai:gpt-4.1",
        tools=tool_collections["permission_tool"],
        checkpointer=checkpointer,
        store=store,
        version="v2",  # Use v2 for post_model_hook support
        post_model_hook=halt_on_risky_tools,  # Native interrupt handling
        prompt=(
            "You are a Permission Agent for Story Protocol permission management.\n\n"
            "INSTRUCTIONS:\n"
            "- Handle permission-related operations (currently no tools available)\n"
            "- Inform users about permission functionality status\n"
            "- After completing tasks, respond to the supervisor with results"
        ),
        name="PERMISSION_AGENT",
    )

    # Create Royalty Agent
    ROYALTY_AGENT = create_react_agent(
        model="openai:gpt-4.1",
        tools=tool_collections["royalty_tool"],
        checkpointer=checkpointer,
        store=store,
        version="v2",  # Use v2 for post_model_hook support
        post_model_hook=halt_on_risky_tools,  # Native interrupt handling
        prompt=(
            "You are a Royalty Agent for Story Protocol royalty management.\n\n"
            "INSTRUCTIONS:\n"
            "- Handle royalty payments on behalf of IP assets\n"
            "- Claim revenue from child IPs for ancestor IPs\n"
            "- Calculate and process royalty distributions\n"
            "- After completing tasks, respond to the supervisor with results\n"
            "- Include payment amounts and recipient details"
        ),
        name="ROYALTY_AGENT",
    )

    # Create WIP Agent  
    WIP_AGENT = create_react_agent(
        model="openai:gpt-4.1",
        tools=tool_collections["wip_tool"],
        checkpointer=checkpointer,
        store=store,
        version="v2",  # Use v2 for post_model_hook support
        post_model_hook=halt_on_risky_tools,  # Native interrupt handling
        prompt=(
            "You are a WIP Token Agent for Story Protocol wrapped IP token operations.\n\n"
            "INSTRUCTIONS:\n"
            "- Handle WIP token deposits (wrapping IP to WIP)\n"
            "- Transfer WIP tokens between addresses\n"
            "- Explain WIP token mechanics and benefits\n"
            "- After completing tasks, respond to the supervisor with results\n"
            "- Include transaction hashes and token amounts"
        ),
        name="WIP_AGENT",
    )

    return {
        "IP_ASSET_AGENT": IP_ASSET_AGENT,
        "IP_ACCOUNT_AGENT": IP_ACCOUNT_AGENT, 
        "IP_LICENSE_AGENT": IP_LICENSE_AGENT,
        "NFT_CLIENT_AGENT": NFT_CLIENT_AGENT,
        "DISPUTE_AGENT": DISPUTE_AGENT,
        "GROUP_AGENT": GROUP_AGENT,
        "PERMISSION_AGENT": PERMISSION_AGENT,
        "ROYALTY_AGENT": ROYALTY_AGENT,
        "WIP_AGENT": WIP_AGENT
    }


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
    
    # Create all agents with loaded tools
    agents = await create_all_agents(checkpointer=checkpointer, store=store, mcp_tools=mcp_tools)
    
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
            "- The user is not aware of the different specialized agent assistants, so do not mention them\n"
        ),
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile(checkpointer=checkpointer, store=store)
    
    return supervisor, agents


async def get_supervisor_or_create_supervisor():
    """Get the supervisor system, creating it if not cached."""
    global GLOBAL_SUPERVISOR_SYSTEM
    if GLOBAL_SUPERVISOR_SYSTEM is None:
        supervisor, agents = await create_supervisor_system()
        GLOBAL_SUPERVISOR_SYSTEM = {
            "supervisor": supervisor,
            "agents": agents
        }
    return GLOBAL_SUPERVISOR_SYSTEM["supervisor"]

async def get_agents_or_create_agents():
    """Get all agents, creating them if not cached."""
    global GLOBAL_SUPERVISOR_SYSTEM
    if GLOBAL_SUPERVISOR_SYSTEM is None:
        supervisor, agents = await create_supervisor_system()
        GLOBAL_SUPERVISOR_SYSTEM = {
            "supervisor": supervisor,
            "agents": agents
        }
    return GLOBAL_SUPERVISOR_SYSTEM["agents"]

# Track interrupted conversations
_interrupted_conversations = {}

def _serialize_langchain_objects(obj):
    """Recursively serialize LangChain objects and other complex objects to JSON-serializable format."""
    import json
    from langchain_core.messages import BaseMessage
    
    # Handle None
    if obj is None:
        return None
        
    # Handle primitive types
    if isinstance(obj, (str, int, float, bool)):
        return obj
        
    # Handle LangChain messages
    if isinstance(obj, BaseMessage):
        return {
            "type": obj.__class__.__name__,
            "content": obj.content,
            "additional_kwargs": _serialize_langchain_objects(getattr(obj, 'additional_kwargs', {})),
            "response_metadata": _serialize_langchain_objects(getattr(obj, 'response_metadata', {})),
            "tool_calls": _serialize_langchain_objects(getattr(obj, 'tool_calls', [])),
            "usage_metadata": _serialize_langchain_objects(getattr(obj, 'usage_metadata', {})),
        }
        
    # Handle LangGraph interrupts - extract the value as per LangGraph best practices
    if hasattr(obj, '__class__') and 'Interrupt' in obj.__class__.__name__:
        # Extract the interrupt value (this should be JSON-serializable)
        if hasattr(obj, 'value'):
            return _serialize_langchain_objects(obj.value)
        else:
            return {"type": "interrupt", "details": str(obj)}
            
    # Handle dictionaries
    if isinstance(obj, dict):
        # Special handling for __interrupt__ key as per LangGraph docs
        if "__interrupt__" in obj:
            interrupts = obj["__interrupt__"]
            if interrupts and len(interrupts) > 0:
                # Extract just the value from the first interrupt
                interrupt_value = interrupts[0].value if hasattr(interrupts[0], 'value') else str(interrupts[0])
                result = obj.copy()
                result["__interrupt__"] = interrupt_value
                return {key: _serialize_langchain_objects(value) if key != "__interrupt__" else value for key, value in result.items()}
        return {key: _serialize_langchain_objects(value) for key, value in obj.items()}
        
    # Handle lists/tuples
    if isinstance(obj, (list, tuple)):
        return [_serialize_langchain_objects(item) for item in obj]
        
    # Handle sets
    if isinstance(obj, set):
        return list(obj)
        
    # Try to use object's dict if it has one
    if hasattr(obj, '__dict__'):
        try:
            return _serialize_langchain_objects(obj.__dict__)
        except:
            pass
            
    # Try JSON serialization test
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        # If all else fails, convert to string
        return str(obj)


async def resume_interrupted_conversation(
    conversation_id: str,
    interrupt_id: str,
    confirmed: bool,
    wallet_address: Optional[str] = None
):
    """Resume an interrupted conversation after user confirmation."""
    
    import logging
    import asyncio
    import traceback
    from langgraph.types import Command
    
    logger = logging.getLogger(__name__)
    
    logger.info(f"🔍 RESUME: Starting resume for conversation {conversation_id} with confirmation: {confirmed}")
    
    try:
        # Get the supervisor system with conversation-specific checkpointer
        supervisor = create_math_agent(conversation_id)
        
        # Log supervisor details
        logger.info(f"🔍 RESUME: Using Math_agent supervisor")
        logger.info(f"🔍 RESUME: Checkpointer type: {type(supervisor.checkpointer).__name__}")
        logger.info(f"🔍 RESUME: Checkpointer instance ID: {id(supervisor.checkpointer)}")
        logger.info(f"🔍 RESUME: Store type: {type(supervisor.store).__name__}")
        logger.info(f"🔍 RESUME: Store instance ID: {id(supervisor.store)}")
        
        # Create thread config to resume from checkpoint
        thread_config = {
            "configurable": {
                "thread_id": conversation_id,
                "wallet_address": wallet_address
            }
        }
        logger.info(f"🔍 RESUME: Thread config: {thread_config}")
        
        # **CRITICAL STEP**: Check if checkpoint state exists before attempting resume
        try:
            logger.info(f"🔍 RESUME: Checking checkpoint state before resume...")
            state = await supervisor.aget_state(thread_config)
            logger.info(f"🔍 RESUME: Checkpoint state found: {state is not None}")
            
            if state:
                logger.info(f"🔍 RESUME: State object type: {type(state)}")
                if hasattr(state, 'values'):
                    logger.info(f"🔍 RESUME: State values keys: {list(state.values.keys())}")
                    if 'messages' in state.values:
                        messages = state.values['messages']
                        logger.info(f"🔍 RESUME: Found {len(messages)} messages in checkpoint")
                        # Log last few messages for context
                        for i, msg in enumerate(messages[-2:], 1):
                            msg_type = type(msg).__name__
                            content_preview = str(msg.content)[:50] if hasattr(msg, 'content') else 'no content'
                            logger.info(f"🔍 RESUME: Message {i}: {msg_type} - {content_preview}...")
                    else:
                        logger.error(f"🔍 RESUME: ❌ CRITICAL ERROR: No 'messages' key in checkpoint state!")
                        logger.info(f"🔍 RESUME: Available keys: {list(state.values.keys())}")
                else:
                    logger.error(f"🔍 RESUME: ❌ CRITICAL ERROR: State has no 'values' attribute!")
                    logger.info(f"🔍 RESUME: State attributes: {dir(state)}")
                    
                # Also check for interrupts in the state
                if hasattr(state, 'values') and '__interrupt__' in state.values:
                    interrupts = state.values['__interrupt__']
                    logger.info(f"🔍 RESUME: Found {len(interrupts) if interrupts else 0} interrupts in checkpoint")
                else:
                    logger.info(f"🔍 RESUME: No interrupts found in checkpoint state")
                    
            else:
                logger.error(f"🔍 RESUME: ❌ CRITICAL ERROR: No checkpoint state found for thread {conversation_id}!")
                return {"status": "error", "error": f"No checkpoint state found for conversation {conversation_id}"}
                
        except Exception as checkpoint_error:
            logger.error(f"🔍 RESUME: ❌ Error getting checkpoint state: {checkpoint_error}")
            logger.error(f"🔍 RESUME: Checkpoint error traceback: {traceback.format_exc()}")
        
        # Resume the graph execution using proper Command pattern with timeout
        logger.info(f"🔄 RESUME START: Command(resume={confirmed}) for {conversation_id}")
        logger.info(f"🔄 About to call supervisor.ainvoke with Command(resume={confirmed})")
        
        result = await asyncio.wait_for(
            supervisor.ainvoke(
                Command(resume=confirmed),  # Use Command with resume parameter
                config=thread_config
            ),
            timeout=30.0  # 30 second timeout
        )
        
        logger.info(f"🔄 RESUME COMPLETED for {conversation_id}")
        logger.info(f"🔄 Raw result type: {type(result)}")
        logger.info(f"🔄 Raw result keys: {result.keys() if isinstance(result, dict) else 'N/A'}")
        
        # Log the messages if they exist
        if isinstance(result, dict) and 'messages' in result:
            messages = result['messages']
            logger.info(f"🔄 Found {len(messages)} messages in result")
            for i, msg in enumerate(messages[-3:]):  # Log last 3 messages
                logger.info(f"🔄 Message {i}: {type(msg)} - {str(msg)[:100]}...")
        else:
            logger.info(f"🔄 No messages found in result: {str(result)[:200]}...")
        
        # Serialize the result to handle AIMessage and other LangChain objects
        serialized_result = _serialize_langchain_objects(result)
        
        if confirmed:
            logger.info(f"Conversation resumed successfully: {conversation_id}")
            logger.info(f"🔍 RETURN: Returning to frontend - status: completed, result keys: {list(serialized_result.keys()) if isinstance(serialized_result, dict) else type(serialized_result)}")
            return {"status": "completed", "result": serialized_result}
        else:
            logger.info(f"Conversation cancelled by user: {conversation_id}")
            return {"status": "cancelled", "result": serialized_result}
            
    except asyncio.TimeoutError:
        logger.error(f"Timeout resuming conversation {conversation_id} after 30 seconds")
        return {"status": "error", "error": "Resume operation timed out"}
    except Exception as e:
        logger.error(f"Error resuming conversation {conversation_id}: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {"status": "error", "error": str(e)}