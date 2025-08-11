from langgraph.prebuilt import create_react_agent
from .tools_wrapper import create_wrapped_tool_collections, create_wrapped_tool_collections_from_tools

from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from typing import Optional


async def create_all_agents(checkpointer=None, store=None, mcp_tools=None):
    """Create all agents with properly loaded tools.
    
    Args:
        checkpointer: Optional checkpointer for state persistence
        store: Optional store for cross-thread memory
        mcp_tools: Optional pre-loaded MCP tools to use instead of loading separately
    """
    # Load tool collections - either from provided MCP tools or load separately
    if mcp_tools:
        tool_collections = await create_wrapped_tool_collections_from_tools(mcp_tools)
    else:
        tool_collections = await create_wrapped_tool_collections()
    
    # Create IP Asset Agent
    IP_ASSET_AGENT = create_react_agent(
        model="openai:gpt-4.1",
        tools=tool_collections["ip_asset_tool"],
        checkpointer=checkpointer,
        store=store,
        # version="v2" removed - focusing on resume step instead
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
        # version="v2" removed - focusing on resume step instead
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
        # version="v2" removed - focusing on resume step instead
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
        # version="v2" removed - focusing on resume step instead
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
        # version="v2" removed - focusing on resume step instead
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
        # version="v2" removed - focusing on resume step instead
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
        # version="v2" removed - focusing on resume step instead
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
        # version="v2" removed - focusing on resume step instead
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
        # version="v2" removed - focusing on resume step instead
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


async def create_supervisor_system(mcp_tools=None):
    """Create the complete supervisor system with all agents."""
    # Create checkpointer and store for persistence
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    
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


# Cache for the supervisor system
_supervisor_cache = None

async def get_supervisor():
    """Get the supervisor system, creating it if not cached."""
    global _supervisor_cache
    if _supervisor_cache is None:
        supervisor, agents = await create_supervisor_system()
        _supervisor_cache = {
            "supervisor": supervisor,
            "agents": agents
        }
    return _supervisor_cache["supervisor"]

async def get_agents():
    """Get all agents, creating them if not cached."""
    global _supervisor_cache
    if _supervisor_cache is None:
        supervisor, agents = await create_supervisor_system()
        _supervisor_cache = {
            "supervisor": supervisor,
            "agents": agents
        }
    return _supervisor_cache["agents"]

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
    from langgraph.types import Command
    
    logger = logging.getLogger(__name__)
    
    logger.info(f"Resuming conversation {conversation_id} with confirmation: {confirmed}")
    
    try:
        # Get the supervisor system
        supervisor = await get_supervisor()
        
        # Create thread config to resume from checkpoint
        thread_config = {
            "configurable": {
                "thread_id": conversation_id,
                "checkpoint_ns": "",
                "wallet_address": wallet_address
            }
        }
        
        # Resume the graph execution using proper Command pattern with timeout
        # This is the correct way to resume interrupts in LangGraph 2025
        
        # Create structured resume value as per LangGraph tutorial
        if confirmed:
            resume_value = {"type": "accept"}
        else:
            resume_value = {"type": "reject"}
        
        logger.info(f"🔄 RESUME START: Command(resume={resume_value}) for {conversation_id}")
        
        # Add timeout to prevent hanging - 30 seconds should be enough
        logger.info(f"🔄 About to call supervisor.ainvoke with Command(resume={resume_value})")
        
        result = await asyncio.wait_for(
            supervisor.ainvoke(
                Command(resume=resume_value),  # Use structured resume value format
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