import asyncio
import json
import logging
import os
import sys
import time
import traceback
import uuid
from asyncio import Queue, Lock
from datetime import datetime
from typing import Optional, Tuple, List, Any, Dict

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.types import interrupt, Command
from langgraph_supervisor import create_supervisor

from .interrupt_handler import create_transaction_interrupt, send_standard_interrupt
from .tool_categories import categorize_tools

logger = logging.getLogger(__name__)

# Global supervisor systems cache (dict with conversation_id as key)
GLOBAL_SUPERVISOR_SYSTEMS = {}

# Define risky tools that require user confirmation
RISKY_TOOLS = {
    "create_spg_nft_collection",
    "mint_and_register_ip_with_terms", 
    "register_derivative",
    "mint_license_tokens",
    "pay_royalty_on_behalf",
    "raise_dispute",
    "deposit_wip",
    "transfer_wip",
    "upload_image_to_ipfs",
    "create_ip_metadata"
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
            tool_args = tc.get("args", {})
            
            if tool_name in RISKY_TOOLS:
                # Create standardized interrupt using interrupt_handler
                interrupt_msg = create_transaction_interrupt(
                    tool_name=tool_name,
                    operation=f"Execute {tool_name}",
                    parameters=tool_args,
                    network="Story Testnet"
                )
                
                # Send interrupt with proper format
                response = send_standard_interrupt(interrupt_msg)

                if response:
                    return {}
                tool_messages = ToolMessage(
                    content="Cancelled by human. Continue without executing this tool.",
                    tool_call_id=tc["id"],
                    name=tool_name
                )
                
                return {"messages": [tool_messages]}
    
    return {}








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



async def create_all_agents(mcp_tools):
    """Create all agents with properly loaded tools.
    
    Args:
        mcp_tools: pre-loaded MCP tools
    """

    direct_tools = mcp_tools
    
    # Create tool collections using direct tools (maintain categorization for now)

    tool_collections = categorize_tools(mcp_tools)
    

    
    # Create IP Asset Agent
    IP_ASSET_AGENT = create_react_agent(
        model="openai:gpt-4.1",
        tools=tool_collections["ip_asset_tool"],
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
            "- Please always show the exact result of the tool return, do not summarize or miss any information\n"
            "- Always call one tool at a time\n"
            "- Please always confirm the parameters of the tool call with the user before calling the tool\n"
            "- If you don't have available tools, say so and ask the supervisor to assign a different agent\n\n"
            
            "**IMPORTANT WORKFLOW FOR mint_and_register_ip_with_terms:**\n"
            "Before calling mint_and_register_ip_with_terms, you MUST follow this workflow:\n\n"
            
            "1. **Check SPG Contract Minting Parameters:**\n"
            "   - First call get_spg_nft_contract_minting_fee_and_token to get the contract's minting fee and token\n"
            "   - This will return the minting_fee and minting_token for the specified SPG NFT contract\n\n"
            
            "2. **Set Hidden Parameters Based on Contract Type:**\n\n"
            "   **Option A: If user provides a specific spg_nft_contract address:**\n"
            "   - Set spg_nft_contract_max_minting_fee = the minting_fee from step 1\n"
            "   - Set spg_nft_contract_mint_fee_token = the minting_token from step 1\n\n"
            
            "   **Option B: If user leaves spg_nft_contract blank (using default):**\n"
            "   - Set spg_nft_contract_max_minting_fee = 0\n"
            "   - Set spg_nft_contract_mint_fee_token = \"0x1514000000000000000000000000000000000000\"\n\n"
            
            "3. **Execute mint_and_register_ip_with_terms:**\n"
            "   - Use the exact values obtained/set in step 2 for the hidden parameters\n"
            "   - These parameters ensure proper fee handling during the minting process\n\n"
            
            "**Example Workflow:**\n"
            "User: \"Mint an IP asset with SPG contract 0x123...\"\n"
            "1. Call: get_spg_nft_contract_minting_fee_and_token(spg_nft_contract=\"0x123...\")\n"
            "2. Get result: {minting_fee: 100000, minting_token: \"0xABC...\"}\n"
            "3. Call: mint_and_register_ip_with_terms(..., spg_nft_contract_max_minting_fee=100000, spg_nft_contract_mint_fee_token=\"0xABC...\")\n\n"
            
            "User: \"Mint an IP asset\" (no specific contract = use default)\n"
            "1. Call: mint_and_register_ip_with_terms(..., spg_nft_contract_max_minting_fee=0, spg_nft_contract_mint_fee_token=\"0x1514000000000000000000000000000000000000\")"
            
        ),
        name="IP_ASSET_AGENT",
    )

    # Create IP Account Agent  
    IP_ACCOUNT_AGENT = create_react_agent(
        model="openai:gpt-4.1",
        tools=tool_collections["ip_account_tool"],
        version="v2",  # Use v2 for post_model_hook support
        post_model_hook=halt_on_risky_tools,  # Native interrupt handling
        prompt=(
            "You are an IP Account Agent for Story Protocol account management.\n\n"
            "INSTRUCTIONS:\n"
            "- Handle ERC20 token operations and account balances\n"
            "- Check token balances and mint test tokens when needed\n"
            "- Provide account status and token information\n"
            "- After completing tasks, respond to the supervisor with results\n"
            "- Please always show the exact result of the tool return, do not summarize or miss any information\n"
            "- Always call one tool at a time\n"
            "- Please always confirm the parameters of the tool call with the user before calling the tool\n"
            "- If you don't have available tools, say so and ask the supervisor to assign a different agent"
        ),
        name="IP_ACCOUNT_AGENT",
    )

    # Create License Agent
    IP_LICENSE_AGENT = create_react_agent(
        model="openai:gpt-4.1",
        tools=tool_collections["license_tool"],
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
            "- Please always show the exact result of the tool return, do not summarize or miss any information\n"
            "- Always call one tool at a time\n"
            "- Please always confirm the parameters of the tool call with the user before calling the tool\n"
            "- If you don't have available tools, say so and ask the supervisor to assign a different agent"
        ),
        name="IP_LICENSE_AGENT",
    )

    # Create NFT Client Agent
    NFT_CLIENT_AGENT = create_react_agent(
        model="openai:gpt-4.1",
        tools=tool_collections["nft_client_tool"],
        version="v2",  # Use v2 for post_model_hook support
        post_model_hook=halt_on_risky_tools,  # Native interrupt handling
        prompt=(
            "You are an NFT Client Agent for Story Protocol NFT collection management.\n\n"
            "INSTRUCTIONS:\n"
            "- Create SPG NFT collections with custom settings\n"
            "- Check SPG contract minting fees and tokens\n"
            "- Configure collection parameters (mint fees, tokens, settings)\n"
            "- After completing tasks, respond to the supervisor with results\n"
            "- Please always show the exact result of the tool return, do not summarize or miss any information\n"
            "- Always call one tool at a time\n"
            "- Please always confirm the parameters of the tool call with the user before calling the tool\n"
            "- If you don't have available tools, say so and ask the supervisor to assign a different agent"
        ),
        name="NFT_CLIENT_AGENT",
    )

    # Create Dispute Agent
    DISPUTE_AGENT = create_react_agent(
        model="openai:gpt-4.1",
        tools=tool_collections["dispute_tool"],
        version="v2",  # Use v2 for post_model_hook support
        post_model_hook=halt_on_risky_tools,  # Native interrupt handling
        prompt=(
            "You are a Dispute Agent for Story Protocol dispute management.\n\n"
            "INSTRUCTIONS:\n"
            "- Handle dispute raising against IP assets\n"
            "- Explain dispute processes and requirements\n"
            "- Process dispute bonds and evidence\n"
            "- After completing tasks, respond to the supervisor with results\n"
            "- Please always show the exact result of the tool return, do not summarize or miss any information\n"
            "- Always call one tool at a time\n"
            "- Please always confirm the parameters of the tool call with the user before calling the tool\n"
            "- If you don't have available tools, say so and ask the supervisor to assign a different agent"
        ),
        name="DISPUTE_AGENT",
    )

    # Create Group Agent (placeholder for future group operations)
    GROUP_AGENT = create_react_agent(
        model="openai:gpt-4.1",
        tools=tool_collections["group_tool"],
        version="v2",  # Use v2 for post_model_hook support
        post_model_hook=halt_on_risky_tools,  # Native interrupt handling
        prompt=(
            "You are a Group Agent for Story Protocol group operations.\n\n"
            "INSTRUCTIONS:\n"
            "- Handle group-related operations (currently no tools available)\n"
            "- Inform users about group functionality status\n"
            "- After completing tasks, respond to the supervisor with results\n"
            "- Please always show the exact result of the tool return, do not summarize or miss any information\n"
            "- Always call one tool at a time\n"
            "- Please always confirm the parameters of the tool call with the user before calling the tool\n"
            "- If you don't have available tools, say so and ask the supervisor to assign a different agent"
        ),
        name="GROUP_AGENT",
    )

    # Create Permission Agent (placeholder for future permission operations)
    PERMISSION_AGENT = create_react_agent(
        model="openai:gpt-4.1",
        tools=tool_collections["permission_tool"],
        version="v2",  # Use v2 for post_model_hook support
        post_model_hook=halt_on_risky_tools,  # Native interrupt handling
        prompt=(
            "You are a Permission Agent for Story Protocol permission management.\n\n"
            "INSTRUCTIONS:\n"
            "- Handle permission-related operations (currently no tools available)\n"
            "- Inform users about permission functionality status\n"
            "- After completing tasks, respond to the supervisor with results\n"
            "- Please always show the exact result of the tool return, do not summarize or miss any information\n"
            "- Always call one tool at a time\n"
            "- Please always confirm the parameters of the tool call with the user before calling the tool\n"
            "- If you don't have available tools, say so and ask the supervisor to assign a different agent"
        ),
        name="PERMISSION_AGENT",
    )

    # Create Royalty Agent
    ROYALTY_AGENT = create_react_agent(
        model="openai:gpt-4.1",
        tools=tool_collections["royalty_tool"],
        version="v2",  # Use v2 for post_model_hook support
        post_model_hook=halt_on_risky_tools,  # Native interrupt handling
        prompt=(
            "You are a Royalty Agent for Story Protocol royalty management.\n\n"
            "INSTRUCTIONS:\n"
            "- Handle royalty payments on behalf of IP assets\n"
            "- Claim revenue from child IPs for ancestor IPs\n"
            "- Calculate and process royalty distributions\n"
            "- After completing tasks, respond to the supervisor with results\n"
            "- Please always show the exact result of the tool return, do not summarize or miss any information\n"
            "- Always call one tool at a time\n"
            "- Please always confirm the parameters of the tool call with the user before calling the tool\n"
            "- If you don't have available tools, say so and ask the supervisor to assign a different agent"
        ),
        name="ROYALTY_AGENT",
    )

    # Create WIP Agent  
    WIP_AGENT = create_react_agent(
        model="openai:gpt-4.1",
        tools=tool_collections["wip_tool"],
        version="v2",  # Use v2 for post_model_hook support
        post_model_hook=halt_on_risky_tools,  # Native interrupt handling
        prompt=(
            "You are a WIP Token Agent for Story Protocol wrapped IP token operations.\n\n"
            "INSTRUCTIONS:\n"
            "- Handle WIP token deposits (wrapping IP to WIP)\n"
            "- Transfer WIP tokens between addresses\n"
            "- Explain WIP token mechanics and benefits\n"
            "- After completing tasks, respond to the supervisor with results\n"
            "- Please always show the exact result of the tool return, do not summarize or miss any information\n"
            "- Always call one tool at a time\n"
            "- Please always confirm the parameters of the tool call with the user before calling the tool\n"
            "- If you don't have available tools, say so and ask the supervisor to assign a different agent"
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


async def get_or_create_supervisor_system(mcp_tools, conversation_id: str):
    """Create the complete supervisor system with all agents.
    
    Args:
        mcp_tools: pre-loaded MCP tools
        conversation_id: unique identifier for the conversation
    """
    global GLOBAL_SUPERVISOR_SYSTEMS
    
    # Check if supervisor already exists for this conversation_id
    if conversation_id in GLOBAL_SUPERVISOR_SYSTEMS:
        return GLOBAL_SUPERVISOR_SYSTEMS[conversation_id]
    
    checkpointer = InMemorySaver()
    store = InMemoryStore()
    
    # Create all agents with loaded tools
    agents = await create_all_agents(mcp_tools)
    
    # Create supervisor with all agents
    supervisor = create_supervisor(
        model=init_chat_model("openai:gpt-4.1"),
        agents=list(agents.values()),
        prompt=(
            """
            You are a supervisor managing Story Protocol specialized agents.    
            Only handle actions or questions related to Story Protocol IP management, NFT operations, royalties, disputes, IPFS & Metadata, permissions, the tools provided, and blockchain related questions.

            
            **🚨 CRITICAL RULE - SPECIALIST AGENT RESPONSES**:
            When a specialist agent returns ANY response (especially tool results):
            
            1. **NEVER MODIFY, SUMMARIZE, OR PARAPHRASE** the specialist agent's response
            2. **PASS THROUGH EXACTLY** what the specialist agent said
            3. **PRESERVE ALL DETAILS** including:
               - Transaction hashes, addresses, IDs
               - Error messages and technical details  
               - Numerical values, fees, amounts
               - JSON objects, metadata, timestamps
               - Formatting, spacing, and structure
            
            4. **FORBIDDEN ACTIONS**:
               - ❌ "The agent successfully created..." (NO SUMMARIZING)
               - ❌ "Here's a summary of the result..." (NO SUMMARIES)
               - ❌ "The operation completed with..." (NO PARAPHRASING)
               - ❌ Removing any part of the specialist's response
               - ❌ Reformatting or restructuring the specialist's output
            
            5. **CORRECT APPROACH**:
               - ✅ Copy the ENTIRE specialist agent response verbatim
               - ✅ You may add brief introductory context BEFORE the response
               - ✅ You may suggest next steps AFTER the complete response
               - ✅ But the specialist's actual response must be 100% intact

            
            **❗ IMPORTANT BEHAVIORAL RULES**:
            - The user is not aware of the different specialized agent assistants, so do not mention them by name
            - Act as a unified interface - route requests but present results as if you performed them
            - When specialists complete tasks, present their results directly without "The agent did X" language

            
            **🔧 AVAILABLE AGENTS**:
            - IP_ASSET_AGENT: Creates, registers IP assets, handles metadata and IPFS
            - IP_ACCOUNT_AGENT: Manages ERC20 tokens, checks balances, mints test tokens
            - IP_LICENSE_AGENT: Handles licensing, mints license tokens, attaches terms
            - NFT_CLIENT_AGENT: Creates SPG collections, manages contract fees
            - DISPUTE_AGENT: Raises disputes, handles dispute processes
            - ROYALTY_AGENT: Processes royalty payments, claims revenue
            - WIP_AGENT: Handles WIP token operations (wrapping, transfers)
            - GROUP_AGENT: Group operations (limited functionality)
            - PERMISSION_AGENT: Permission management (limited functionality)

            
            **🌐 NETWORK INFORMATION:**
            Remember the native token is $IP and wrapped version is WIP for transactions.
            - Story Testnet (Aeneid): Chain ID 1315
            - Explorer: https://aeneid.explorer.story.foundation

            **💰 TOKEN ADDRESSES (ALWAYS use exact addresses, never token names):**
            - WIP (Wrapped IP): 0x1514000000000000000000000000000000000000
            - MERC20 (Test Token): 0xF2104833d386a2734a4eB3B8ad6FC6812F29E38E
            IMPORTANT: When users say "WIP", "MERC20", or token names, always convert to the full address above.
            
            
            **🔧 INSTRUCTIONS:**
            - Analyze user requests and route to the most appropriate specialist agent
            - Assign work to ONE agent at a time, do not call agents in parallel
            - For complex workflows, coordinate between agents sequentially
            - Do not perform any Story Protocol operations yourself
            - If one agent is not able to handle the request, hand off to the next agent
            - When presenting results, show the specialist's complete response without modification
            """
        ),
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile(checkpointer=checkpointer, store=store)

    # Cache the supervisor for this conversation_id
    GLOBAL_SUPERVISOR_SYSTEMS[conversation_id] = supervisor
    return supervisor


async def get_supervisor_from_cache(conversation_id: str):
    """Get the supervisor system from cache for a specific conversation"""
    global GLOBAL_SUPERVISOR_SYSTEMS
    
    if conversation_id in GLOBAL_SUPERVISOR_SYSTEMS:
        return GLOBAL_SUPERVISOR_SYSTEMS[conversation_id]
    else:
        return None


def _serialize_langchain_objects(obj):
    """Recursively serialize LangChain objects and other complex objects to JSON-serializable format."""
    
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
    
    try:
        # Use the SAME supervisor instance that was interrupted
        # Creating a new supervisor would lose the interrupted state!
        supervisor = await get_supervisor_from_cache(conversation_id)
        
        
        # Create thread config to resume from checkpoint
        thread_config = {
            "configurable": {
                "thread_id": conversation_id,
                "wallet_address": wallet_address
            }
        }
        
        result = await asyncio.wait_for(
            supervisor.ainvoke(
                Command(resume=confirmed),  # Use Command with resume parameter
                config=thread_config
            ),
            timeout=30.0  # 30 second timeout
        )
        
        # Extract only the last AI message content
        last_ai_content = None
        if isinstance(result, dict) and 'messages' in result:
            messages = result['messages']
            
            # Find the last AI message
            for msg in reversed(messages):
                if hasattr(msg, '__class__') and 'AI' in msg.__class__.__name__:
                    if hasattr(msg, 'content'):
                        last_ai_content = msg.content
                    break
        
        if confirmed:
            return {
                "status": "completed", 
                "message": last_ai_content,
                "conversation_id": conversation_id
            }
        else:
            return {
                "status": "cancelled", 
                "message": last_ai_content,
                "conversation_id": conversation_id
            }
            
    except asyncio.TimeoutError:
        return {"status": "error", "error": "Resume operation timed out"}
    except Exception as e:
        return {"status": "error", "error": str(e)}
    

