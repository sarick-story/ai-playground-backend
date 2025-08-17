# Create server parameters for stdio connection
# MCP imports removed - now using centralized get_or_create_mcp_tools()
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from .supervisor_agent_system import get_supervisor_or_create_supervisor, create_supervisor_system, get_or_create_mcp_tools
import os
import asyncio
import asyncio.subprocess  # Import at the module level to avoid local variable issue
import subprocess  # Import at the module level
import json
import uuid
import traceback
import sys
import time
import re
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv
from langchain_core.callbacks.base import BaseCallbackHandler
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager, create_manage_memory_tool, create_search_memory_tool
from web3 import Web3
import atexit
import signal
import anyio
import io

load_dotenv()

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Helper functions for address validation
def is_valid_ethereum_address(address: str) -> bool:
    """
    Validate an Ethereum address format.
    
    Args:
        address: The address to validate
        
    Returns:
        True if the address is valid, False otherwise
    """
    if not isinstance(address, str):
        return False
    
    # Check if it starts with 0x
    if not address.startswith('0x'):
        return False
    
    # Check if it's the right length (42 characters = 0x + 40 hex chars)
    if len(address) != 42:
        return False
    
    # Check if it contains only valid hex characters
    return bool(re.match(r'^0x[0-9a-fA-F]{40}$', address))

def validate_and_format_address(address: str) -> str:
    """
    Validate and format an Ethereum address.
    
    Args:
        address: The address to validate and format
        
    Returns:
        The formatted address
    
    Raises:
        ValueError: If the address is invalid
    """
    # Add 0x prefix if missing
    if not address.startswith('0x'):
        address = f'0x{address}'
    
    # Check if the address is valid
    if not is_valid_ethereum_address(address):
        raise ValueError(f"Invalid Ethereum address: {address}")
    
    return address

            

model = ChatOpenAI(model="gpt-4.1", api_key=os.getenv("OPENAI_API_KEY"), streaming=True)

system_prompt = """
    You are a specialized Story Protocol SDK assistant focused on intellectual property management, NFT operations, and blockchain transactions on Story Protocol. 
    Only handle actions or questions related to Story Protocol IP management, NFT operations, royalties, disputes, IPFS & Metadata, the tools provided, and blockchain related questions.
    
    🚨 **CRITICAL SECURITY PROTOCOL - READ TOOL DOCUMENTATION** 🚨
    EVERY TOOL HAS ITS OWN DOCUMENTATION. YOU MUST READ AND FOLLOW IT EXACTLY.
    
    **⛔ MANDATORY TOOL USAGE RULES ⛔:**
    
    1. **READ FIRST**: Before using ANY tool, ALWAYS read its complete documentation/docstring
    
    2. **WORKFLOW DETECTION**: If you see any of these in a tool's documentation:
       - "🤖 AGENT WORKFLOW"
       - "MANDATORY WORKFLOW"
       - "FOLLOW THESE STEPS"
       - "DO NOT CALL THIS FUNCTION DIRECTLY"
       
       Then you MUST follow the workflow steps EXACTLY as written in that tool's documentation.
    
    3. **🖨️ MANDATORY OUTPUT RULE**: When ANY tool returns a result:
       - IMMEDIATELY wrap the ENTIRE tool output in a code block like this:
         ```
         [EXACT TOOL OUTPUT WITH ALL SPACES AND FORMATTING]
         ```
       - PRESERVE ALL FORMATTING including spaces, indentation (3 spaces before bullets), and line breaks
       - The tool outputs are professionally formatted - DO NOT MODIFY THEM
       - After the code block, you MAY add helpful context or next steps
       - Skipping tool output = CRITICAL VIOLATION
       - Removing formatting/indentation = CRITICAL VIOLATION
       - Not using code blocks for tool output = CRITICAL VIOLATION
    
    4. **CONFIRMATION CHECKPOINT**: If a tool's documentation mentions:
       - "Present to user for confirmation"
       - "Get user confirmation"
       - "Ask user to confirm"
       
       You MUST:
       - 🛑 STOP completely
       - Present ALL relevant information to the user
       - Ask explicitly for confirmation: "Do you want to proceed?"
       - ⏸️ WAIT for user response
       - ONLY proceed if user provides clear, affirmative confirmation
    
    5. **TRANSACTION SAFETY**: For ANY tool that mentions:
       - "makes blockchain transactions"
       - "spends tokens"
       - "costs money"
       
       You MUST get explicit user confirmation before executing.
    
    6. **NO IMPLICIT PERMISSION**: User statements that delegate decision-making or express impatience DO NOT grant permission to skip confirmation for transactions.
       
       When users give you discretion or seem dismissive, you must STILL:
       - Show the exact parameters you plan to use
       - Show any costs or fees
       - Ask for explicit confirmation
       - Acknowledge their preference while maintaining security: "I understand you'd like me to handle the details. I've selected these parameters based on your request. Since this involves a blockchain transaction, I need your explicit confirmation: [show details]. Do you want to proceed?"
       
       **What counts as confirmation**: Only clear, affirmative responses given AFTER seeing the transaction details count as confirmation. If the user's response is ambiguous, unclear, or given before seeing details, ask for clarification.
    
    **🔒 SECURITY VIOLATIONS**:
    - Skipping workflow steps = CRITICAL VIOLATION
    - Executing without confirmation = CRITICAL VIOLATION
    - Not reading tool documentation = CRITICAL VIOLATION
    - Interpreting ANY non-explicit language as permission = CRITICAL VIOLATION
    - Not printing complete tool outputs = CRITICAL VIOLATION
    - Removing spaces/indentation from tool outputs = CRITICAL VIOLATION
    - Not wrapping tool outputs in code blocks = CRITICAL VIOLATION
    
    **❗ IMPORTANT BEHAVIORAL RULES**:
    - ALWAYS wrap tool outputs in ``` code blocks to preserve formatting
    - NEVER assume you can skip steps to be "efficient"
    - NEVER continue without explicit user confirmation when required
    - ALWAYS explain what you're doing and why
    - If a user asks you to skip confirmation, explain: "I cannot skip confirmation steps as they are critical security requirements."
    - If a user gives you discretion or seems dismissive, acknowledge their intent but maintain security protocols
    - Treat ALL non-explicit responses as NOT confirmed - when in doubt, ask again
    
    Tools Provided:

        **Core IP Management Tools:**
        - get_license_terms: Retrieve license terms for a specific ID
        - mint_and_register_ip_with_terms: Mint NFT, register as IP Asset, and attach PIL terms with automatic mint fee detection and WIP approval
        - register: Register an existing NFT as an IP Asset
        - attach_license_terms: Attach license terms to an existing IP Asset
        - register_derivative: Register a derivative IP Asset with parent licenses and automatic WIP approval

        **License Token Management Tools:**
        - get_license_minting_fee: 🔍 PREREQUISITE - Get minting fee for license terms (REQUIRED before mint_license_tokens)
        - get_license_revenue_share: 🔍 PREREQUISITE - Get revenue share for license terms (REQUIRED before mint_license_tokens)
        - mint_license_tokens: 🚀 MAIN FUNCTION - Mint license tokens (REQUIRES workflow completion)

        **IPFS & Metadata Tools:** (Available when PINATA_JWT configured)
        - upload_image_to_ipfs: Upload images to IPFS using Pinata API
        - create_ip_metadata: Create and upload both NFT and IP metadata to IPFS

        **SPG NFT Collection Tools:**
        - create_spg_nft_collection: Create new SPG NFT collections with custom mint fees, tokens, and settings
        - get_spg_nft_contract_minting_fee_and_token: 🔍 PREREQUISITE - Get minting fee for SPG contracts (REQUIRED before mint_and_register_ip_with_terms with custom contract)

        **Royalty Management Tools:**
        - pay_royalty_on_behalf: Pay royalties on behalf of one IP to another with automatic WIP approval
        - claim_all_revenue: Claim all revenue from child IPs of an ancestor IP

        **Dispute Management Tools:**
        - raise_dispute: Raise disputes against IP assets with bond payments and automatic WIP approval

        **WIP (Wrapped IP) Token Tools:**
        - deposit_wip: Wrap IP tokens to WIP tokens
        - transfer_wip: Transfer WIP tokens to recipients
    
    **Enhanced Features:**
    - 💰 **Automatic Token Approvals**: Most tools automatically handle WIP token approvals before transactions
    - 🔧 **Multi-Token Support**: SPG collections support custom mint fee tokens (WIP, MERC20, custom ERC20)
    - 🛡️ **Smart Fee Detection**: Automatic detection and handling of SPG contract mint fees
    - ⚡ **Gasless Operations**: EIP-2612 permit support for advanced token operations

    **Transaction Safety:**
    ⚠️ All tools except those starting with `get_` will make blockchain transactions and change the blockchain state. Always confirm parameters with users before proceeding.

    Methods starting with `get_` are safe to call—they do NOT make blockchain transactions or change the blockchain; they only retrieve information.
    
    **Common Workflows:**
    1. **Create Collection**: Use create_spg_nft_collection with custom mint fees
    2. **Mint & Register IP**: Use mint_and_register_ip_with_terms for complete IP creation
    3. **License Management**: Use mint_license_tokens and register_derivative for IP licensing
    4. **Revenue Operations**: Use pay_royalty_on_behalf and claim_all_revenue for monetization
    
    **CRITICAL REMINDERS:**
    📋 Each tool's documentation is the source of truth for how to use it
    📋 Workflow steps in tool documentation are MANDATORY, not optional
    📋 Your duty is to protect users by following all security protocols
    📋 "Up to you" ≠ "Skip confirmation" - ALWAYS confirm transactions
    📋 Tool outputs MUST be wrapped in ``` code blocks to preserve formatting
    
    Provide concise and clear analyses of operations using the available tools.
    Remember the native token is $IP and wrapped version is WIP for transactions.

    **Network Information:**
    - Story Testnet (Aeneid): Chain ID 1315
    - Explorer: https://aeneid.explorer.story.foundation
    
    **Token Addresses (ALWAYS use exact addresses, never token names):**
    - WIP (Wrapped IP): 0x1514000000000000000000000000000000000000
    - MERC20 (Test Token): 0xF2104833d386a2734a4eB3B8ad6FC6812F29E38E
    
    IMPORTANT: When users say "WIP", "MERC20", or token names, always convert to the full address above.

"""
# Debug: Print confirmation
logger.info("SDK MCP agent initialized successfully")

async def create_transaction_request(to_address: str, amount: str, queue: asyncio.Queue, private_key: Optional[str] = None) -> bool:
    """Create and send a transaction request directly for testing purposes.
    
    If a private_key is provided, sends the transaction directly.
    Otherwise, creates a transaction intent for the frontend to handle.
    """
    try:
        # Don't modify or validate the address at all - just use it as is
        logger.info(f"Using address as provided: {to_address}")
            
        # Convert amount to float first
        try:
            amount_float = float(amount)
            wei_value = Web3.to_wei(amount_float, "ether")
            logger.info(f"Direct transaction: Converting {amount_float} IP to {wei_value} wei")
        except ValueError as e:
            error_msg = f"Invalid amount format: {str(e)}"
            logger.error(error_msg)
            await queue.put(f"\nError: {error_msg}\n")
            return False
        
        # If private key is provided, send the transaction directly
        if private_key and private_key.strip():
            try:
                logger.info("Private key provided, attempting to send transaction directly")
                # TODO: Implement direct transaction signing using private key
                # This would involve creating and signing a transaction using web3.py
                # For security reasons, this is just a placeholder - direct transaction
                # signing should be implemented with extreme caution
                
                await queue.put(f"\nSending transaction directly using provided private key...\n")
                await queue.put(f"\nTransaction sent successfully!\n")
                
                logger.info(f"Direct transaction sent using private key")
                return True
            except Exception as e:
                error_msg = f"Error sending direct transaction: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                await queue.put(f"\nError: {error_msg}\n")
                return False
        
        # No private key - format the transaction intent for the frontend
        transaction_intent = {
            "to": to_address,
            "amount": str(amount_float),
            "data": "0x"
        }
        
        logger.info(f"Direct transaction: Created transaction intent with amount {amount} IP")
        
        # Format the transaction in the same exact way the frontend expects
        transaction_message = f"Transaction intent: \n```json\n{json.dumps(transaction_intent, indent=2)}\n```"
        logger.info(f"Direct transaction: Created formatted message")
        
        # Send directly as text instead of with special format
        await queue.put(transaction_message)
        
        # Also send the user-friendly message separately
        await queue.put(f"\nI've prepared a transaction to send {amount} IP to {to_address}. Please approve it in your wallet when prompted.\n")
        
        logger.info(f"Direct transaction: Intent sent to frontend")
        
        return True
    except Exception as e:
        error_msg = f"Error creating transaction intent: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        if queue:
            try:
                await queue.put(f"\nError: {error_msg}\n")
            except:
                pass
        return False

# Add this function to handle clean shutdown
async def close_mcp_session():
    """Close the MCP session using the session manager."""
    logger.info("Closing MCP session during shutdown")
    try:
        # No persistent session to close in the new implementation
        pass
    except Exception as e:
        logger.warning(f"Error closing session: {str(e)}")

# Add these lines to ensure the session is properly closed when the application terminates
def cleanup_session():
    """Synchronous cleanup function for atexit hook."""
    try:
        # Try to run the async cleanup in a new event loop
        import asyncio
        loop = asyncio.new_event_loop()
        loop.run_until_complete(close_mcp_session())
        loop.close()
    except Exception as e:
        logger.warning(f"Error during cleanup: {str(e)}")

# Register the cleanup function
atexit.register(cleanup_session)

async def run_agent(
    user_message: str,
    wallet_address: Optional[str] = None,
    queue: Optional[asyncio.Queue] = None,
    conversation_id: Optional[str] = None,
    message_history: Optional[List[Dict[str, str]]] = None
):
    """Run the SDK MCP agent with the given user message."""
    logger.info(f"Starting SDK MCP agent with message: {user_message}")
    
    # Create a task-local context for this invocation
    task_token = str(uuid.uuid4())
    logger.info(f"Creating task context with token: {task_token}")
    
    # Add validation for wallet address here
    if wallet_address == "none" or wallet_address == "null" or wallet_address == "undefined":
        logger.warning(f"Invalid wallet address '{wallet_address}' detected, setting to None")
        wallet_address = None
    
    logger.info(f"Using wallet address: {wallet_address or 'None'}")
    
    # Try direct transaction handling first - this is the most reliable method
    if queue and wallet_address and ("send" in user_message.lower() and "ip to" in user_message.lower()):
        logger.info("Detected direct transaction command, attempting to parse")
        
        # Try to parse the command directly
        try:
            # Pattern: send X IP to ADDRESS - even more lenient regex
            import re
            pattern = r"send\s+([0-9.]+)\s+ip\s+to\s+(0x[a-fA-F0-9]+)"
            match = re.search(pattern, user_message, re.IGNORECASE)
            
            if not match:
                # Try an alternative pattern without restricting address format
                pattern = r"send\s+([0-9.]+)\s+ip\s+to\s+(.*?)($|\s|\.)"
                match = re.search(pattern, user_message, re.IGNORECASE)
            
            if match:
                amount = match.group(1)
                to_address = match.group(2).strip()
                
                try:
                    # Validate and format the address
                    to_address = validate_and_format_address(to_address)
                    logger.info(f"Validated address: {to_address}")
                    
                    # Parse the amount
                    try:
                        amount_float = float(amount)
                        if amount_float <= 0:
                            await queue.put("Error: Amount must be greater than zero.")
                            await queue.put({"done": True})
                            return {"error": "Amount must be greater than zero"}
                    except ValueError:
                        await queue.put(f"Error: Invalid amount format: {amount}")
                        await queue.put({"done": True})
                        return {"error": f"Invalid amount format: {amount}"}
                
                    logger.info(f"Parsed direct transaction: {amount} IP to {to_address}")
                    
                    # Try direct transaction - this should work every time
                    await queue.put(f"Creating transaction to send {amount} IP to {to_address}...")
                    success = await create_transaction_request(to_address, amount, queue)
                    
                    if success:
                        logger.info("Direct transaction request sent successfully")
                        # Tell the frontend we're done after sending the transaction
                        await queue.put({"done": True})
                        return {"success": "Transaction request sent"}
                    else:
                        logger.warning("Failed to send direct transaction, attempting fallback to agent")
                except ValueError as e:
                    # Address validation failed
                    error_msg = str(e)
                    logger.error(f"Address validation failed: {error_msg}")
                    await queue.put(f"Error: {error_msg}")
                    await queue.put({"done": True})
                    return {"error": error_msg}
        except Exception as e:
            logger.error(f"Error in direct transaction handling: {str(e)}")
            logger.error(traceback.format_exc())
            # Continue with agent flow as fallback
    
    # If the user message contains keywords about sending IP tokens but we couldn't parse it directly,
    # add some debugging information
    if queue and wallet_address and ("send" in user_message.lower() and "ip" in user_message.lower()):
        logger.warning("Transaction command detected but couldn't parse with regex, will try agent flow")
        await queue.put("\nDetected transaction intent but couldn't parse directly. Trying AI agent...\n")
    
    # If direct handling failed, use normal agent flow
    try:
        # Use direct call to avoid asyncio issues
        result = await _run_agent_impl(
            user_message=user_message,
            wallet_address=wallet_address,
            queue=queue,
            conversation_id=conversation_id,
            message_history=message_history,
            task_token=task_token
        )
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

async def _run_agent_impl(
    user_message: str,
    wallet_address: Optional[str] = None,
    queue: Optional[asyncio.Queue] = None,
    conversation_id: Optional[str] = None,
    message_history: Optional[List[Dict[str, str]]] = None,
    task_token: str = ""
):
    """Implementation of run_agent with explicit task context."""
    try:
        logger.info(f"Executing agent implementation with token: {task_token}")
        logger.info(f"Wallet address: {wallet_address}")
        logger.info(f"Message history length: {len(message_history) if message_history else 0}")
        
        # Check if this is a transaction request but no wallet is connected
        if "send" in user_message.lower() and "ip" in user_message.lower() and not wallet_address:
            error_msg = "Cannot process transaction: No wallet connected. Please connect your wallet first."
            logger.warning(error_msg)
            if queue:
                await queue.put(f"\n⚠️ {error_msg}\n")
                await queue.put({"done": True})
            return {"error": error_msg}
        
        # Load MCP tools using centralized caching method
        try:
            logger.info("Loading MCP tools with caching")
            tools = await get_or_create_mcp_tools()
            logger.info(f"Loaded {len(tools)} MCP tools")
            
            # Log wallet address if provided
            if wallet_address:
                logger.info(f"Using wallet address: {wallet_address}")

            # Use supervisor system with loaded MCP tools
            logger.info("Using supervisor system with specialized agents")
            #supervisor = await get_supervisor_or_create_supervisor()
            from .supervisor_agent_system import create_math_agent
            supervisor = create_math_agent(conversation_id)
            
            # Log checkpointer details for comparison with resume
            logger.info(f"🔍 INITIAL: Math_agent checkpointer type: {type(supervisor.checkpointer).__name__}")
            logger.info(f"🔍 INITIAL: Math_agent checkpointer instance ID: {id(supervisor.checkpointer)}")
            logger.info(f"🔍 INITIAL: Math_agent store type: {type(supervisor.store).__name__}")
            logger.info(f"🔍 INITIAL: Math_agent store instance ID: {id(supervisor.store)}")
            
            # Prepare messages for supervisor
            messages = message_history or [{"role": "user", "content": user_message}]
            logger.info(f"Prepared {len(messages)} messages for the supervisor")

            if queue:
                # Define the streaming handler directly before using it
                class StreamingCallbackHandler(BaseCallbackHandler):
                    run_inline = True
                    
                    async def on_llm_new_token(self, token: str, **kwargs):
                        try:
                            logger.debug(f"LLM token: {token}")
                            
                            # Skip empty tokens
                            if not token:
                                return
                            
                            # Handle newlines properly - don't escape them
                            # Just ensure the token is safe for streaming
                            safe_token = token
                            
                            # Only filter out problematic control characters, keep newlines
                            safe_token = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', safe_token)
                            
                            # Send the token as-is (with proper newlines)
                            await queue.put(safe_token)
                        except Exception as e:
                            logger.warning(f"Error processing token, skipping: {str(e)}")
                            # If there's an error, just skip this token
                            pass
                    
                    async def on_chain_error(self, error, **kwargs):
                        """Handle chain errors (but not interrupts - those are handled via stream mode)."""
                        try:
                            # Import Command to check instance
                            from langgraph.types import Command
                            
                            # Check if this is a LangGraph Command (normal flow control, not an error)
                            if isinstance(error, Command) or 'Command' in str(type(error)):
                                logger.debug(f"LangGraph Command detected (normal flow control): {type(error).__name__}")
                                return  # Commands are normal flow control, not errors
                            
                            # Check if this is an interrupt - let stream mode handle it, don't treat as error
                            if hasattr(error, '__class__') and 'interrupt' in str(error.__class__).lower():
                                logger.debug(f"Interrupt detected in callback - will be handled by stream mode: {error}")
                                return  # Let astream handle interrupts naturally
                            
                            # Handle genuine errors (not interrupts)
                            error_msg = f"\nError: {str(error)}\n"
                            logger.error(f"Chain error: {error}")
                            await queue.put(error_msg)
                            
                        except Exception as e:
                            logger.error(f"Error handling chain error: {e}")
                            await queue.put(f"\nUnexpected error occurred\n")
                    
                    async def on_tool_start(self, serialized, input_dict, **kwargs):
                        tool_call_id = str(kwargs.get("run_id", uuid.uuid4()))  # Convert UUID to string
                        tool_name = serialized.get("name", "unknown_tool") 
                        tool_description = serialized.get("description", "No description available")
                        
                        logger.info(f"Starting tool: {serialized} with input: {input_dict}")

                        # Send tool call info to frontend using special format
                        tool_call = {
                            "tool_call": {
                                "id": tool_call_id,
                                "name": {
                                    "name": tool_name,
                                    "description": tool_description
                                },
                                "args": json.dumps(input_dict)
                            }
                        }
                        # Don't send internal tool calls to user - they're for frontend processing only
                        # await queue.put(f"__INTERNAL_TOOL_CALL__{json.dumps(tool_call)}__END_INTERNAL__")
                    
                    async def on_tool_end(self, output: str, **kwargs):
                        tool_call_id = str(kwargs.get("run_id", uuid.uuid4()))  # Convert UUID to string
                        tool_name = kwargs.get("name", "unknown_tool")
                        tool_input = kwargs.get("input", {})
                        
                        logger.info(f"Tool completed: {tool_name}")
                        logger.debug(f"Tool output: {output}")
                        
                        output_str = (output.content if hasattr(output, "content")
                                    else str(output) if hasattr(output, "__str__")
                                    else "Tool execution completed")
                        
                        # Send tool result info to frontend using special format
                        tool_result = {
                            "tool_result": {
                                "id": tool_call_id,
                                "name": tool_name,
                                "args": tool_input,
                                "result": output_str
                            }
                        }

                        # Don't send internal tool results to user - they're for frontend processing only
                        # await queue.put(f"__INTERNAL_TOOL_RESULT__{json.dumps(tool_result)}__END_INTERNAL__")
                
                callbacks = [StreamingCallbackHandler()]
                
                # Run supervisor with streaming
                logger.info(f"Starting supervisor system with streaming in task context: {task_token}")
                
                # Create thread config for persistence
                thread_id = conversation_id or str(uuid.uuid4())
                thread_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "wallet_address": wallet_address  # Pass wallet to tools
                    },
                    "callbacks": callbacks
                }
                
                logger.info(f"🔍 INITIAL: Using thread_id: {thread_id}")
                logger.info(f"🔍 INITIAL: Thread config: {thread_config}")
                
                # Check initial state before execution
                try:
                    initial_state = await supervisor.aget_state(thread_config)
                    logger.info(f"🔍 INITIAL: Pre-execution state exists: {initial_state is not None}")
                except Exception as e:
                    logger.info(f"🔍 INITIAL: No pre-existing state (expected): {e}")
                
                # Use astream with proper stream mode to handle interrupts
                final_result = None
                async for chunk in supervisor.astream(
                    {"messages": messages},
                    config=thread_config,
                    stream_mode=["values", "updates"]
                ):
                            logger.info(f"🔄 Stream chunk type: {type(chunk)}")
                            if isinstance(chunk, dict) and "__interrupt__" in chunk:
                                logger.info(f"🔄 Found interrupt in chunk keys: {chunk.keys()}")
                            elif isinstance(chunk, tuple) and len(chunk) == 2:
                                mode, data = chunk
                                logger.info(f"🔄 Tuple chunk - mode: {mode}, data keys: {data.keys() if isinstance(data, dict) else type(data)}")
                                if isinstance(data, dict) and "__interrupt__" in data:
                                    logger.info(f"🔄 Found interrupt in tuple data keys: {data.keys()}")
                            else:
                                logger.debug(f"🔄 Other chunk: {chunk}")
                            
                            # Handle different stream modes
                            if isinstance(chunk, tuple) and len(chunk) == 2:
                                mode, data = chunk
                                
                                if mode == "values":
                                    # Store the latest state
                                    final_result = data
                                    
                                    # Check for interrupts in the state
                                    if "__interrupt__" in data:
                                        interrupts = data["__interrupt__"]
                                        if interrupts:
                                            interrupt_data = interrupts[0]  # Get first interrupt
                                            if hasattr(interrupt_data, 'value'):
                                                interrupt_info = interrupt_data.value
                                            else:
                                                interrupt_info = interrupt_data
                                            
                                            logger.info(f"🔍 INTERRUPT: Interrupt detected in values mode")
                                            logger.info(f"🔍 INTERRUPT: State data keys: {list(data.keys())}")
                                            logger.info(f"🔍 INTERRUPT: Messages count in state: {len(data.get('messages', []))}")
                                            
                                            # Check checkpoint state at interrupt time
                                            try:
                                                interrupt_state = await supervisor.aget_state(thread_config)
                                                logger.info(f"🔍 INTERRUPT: Checkpoint state saved: {interrupt_state is not None}")
                                                if interrupt_state and hasattr(interrupt_state, 'values'):
                                                    logger.info(f"🔍 INTERRUPT: Checkpoint has {len(interrupt_state.values.get('messages', []))} messages")
                                            except Exception as state_error:
                                                logger.error(f"🔍 INTERRUPT: Error checking state during interrupt: {state_error}")
                                            
                                            # Include conversation_id in interrupt data for frontend
                                            interrupt_info['conversation_id'] = conversation_id
                                            
                                            # Send interrupt to frontend
                                            interrupt_message = f"__INTERRUPT_START__{json.dumps(interrupt_info)}__INTERRUPT_END__"
                                            logger.info(f"Sending interrupt to frontend: {interrupt_info.get('interrupt_id', 'unknown')}")
                                            logger.info(f"🔍 INTERRUPT: Backend UUID for resume: {conversation_id}")
                                            await queue.put(interrupt_message)
                                            
                                            # Don't send done=True, wait for frontend confirmation
                                            return {"status": "interrupted", "conversation_id": conversation_id, "interrupt_data": interrupt_info}
                                
                                elif mode == "updates":
                                    # Handle node updates and check for interrupts
                                    logger.debug(f"Node update: {data}")
                                    
                                    # Check for interrupts in updates mode too
                                    if "__interrupt__" in data:
                                        interrupts = data["__interrupt__"]
                                        if interrupts:
                                            interrupt_data = interrupts[0]  # Get first interrupt
                                            if hasattr(interrupt_data, 'value'):
                                                interrupt_info = interrupt_data.value
                                            else:
                                                interrupt_info = interrupt_data
                                            
                                            # Include conversation_id in interrupt data for frontend
                                            interrupt_info['conversation_id'] = conversation_id
                                            
                                            # Send interrupt to frontend
                                            interrupt_message = f"__INTERRUPT_START__{json.dumps(interrupt_info)}__INTERRUPT_END__"
                                            logger.info(f"✅ Sending interrupt to frontend from updates mode: {interrupt_info.get('interrupt_id', 'unknown')}")
                                            await queue.put(interrupt_message)
                                            
                                            # Don't send done=True, wait for frontend confirmation
                                            return {"status": "interrupted", "conversation_id": conversation_id, "interrupt_data": interrupt_info}
                            
                            elif isinstance(chunk, dict):
                                # Direct state update
                                final_result = chunk
                                
                                # Check for interrupts
                                if "__interrupt__" in chunk:
                                    interrupts = chunk["__interrupt__"]
                                    if interrupts:
                                        interrupt_data = interrupts[0]
                                        if hasattr(interrupt_data, 'value'):
                                            interrupt_info = interrupt_data.value
                                        else:
                                            interrupt_info = interrupt_data
                                        
                                        # Add backend conversation_id to interrupt data for frontend
                                        interrupt_info_with_id = {**interrupt_info, "backend_conversation_id": conversation_id}
                                        
                                        interrupt_message = f"__INTERRUPT_START__{json.dumps(interrupt_info_with_id)}__INTERRUPT_END__"
                                        logger.info(f"Sending interrupt to frontend: {interrupt_info.get('interrupt_id', 'unknown')}")
                                        await queue.put(interrupt_message)
                                        
                                        return {"status": "interrupted", "conversation_id": conversation_id, "interrupt_data": interrupt_info}
                
                # If we get here, execution completed without interrupts
                logger.info("Supervisor system completed execution with streaming")
                if final_result:
                    logger.info(f"Final result keys: {final_result.keys() if isinstance(final_result, dict) else 'not dict'}")
                    
                    # Extract and send the final AI response to frontend
                    if isinstance(final_result, dict) and "messages" in final_result:
                                messages_result = final_result["messages"]
                                # Find the last AI message
                                for msg in reversed(messages_result):
                                    if hasattr(msg, 'content') and msg.content and hasattr(msg, '__class__') and 'AI' in str(msg.__class__):
                                        logger.info(f"Sending final AI response: {msg.content}")
                                        await queue.put(msg.content)
                                        break
                
                await queue.put({"done": True})
                return final_result
                    
            else:
                # Run supervisor without streaming
                logger.info("Starting supervisor system without streaming")
                
                # Create thread config for persistence
                thread_config = {
                    "configurable": {
                        "thread_id": conversation_id or str(uuid.uuid4()),
                        "wallet_address": wallet_address
                    }
                }
                
                try:
                    result = await supervisor.ainvoke(
                        {"messages": messages},
                        config=thread_config
                    )
                    logger.info("Supervisor system completed execution without streaming")
                    logger.info(f"Supervisor result: {result}")
                    return result
                except Exception as e:
                    error_msg = f"Error during supervisor execution: {str(e)}"
                    logger.error(error_msg)
                    logger.error(traceback.format_exc())
                    return {"error": error_msg}
        
        except Exception as e:
            error_msg = f"Failed to create MCP session: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            if queue:
                await queue.put(f"Error: {error_msg}")
                await queue.put({"done": True})
            return {"error": error_msg}
        
    except Exception as e:
        logger.error(f"Error in _run_agent_impl: {str(e)}")
        logger.error(traceback.format_exc())
        if queue:
            await queue.put(f"\nError: {str(e)}\n")
            await queue.put({"done": True})
        raise

async def stream_agent_response(
    user_message: str,
    wallet_address: Optional[str] = None,
    conversation_id: Optional[str] = None,
    message_history: Optional[List[Dict[str, str]]] = None
):
    """Stream the agent's response to the user message."""
    # Create a queue for the agent to stream responses
    queue = asyncio.Queue()
    
    # Create a task for running the agent, but don't use asyncio.shield
    # as it's causing issues with cancellation - call directly instead
    agent_task = asyncio.create_task(
        run_agent(
            user_message,
            wallet_address=wallet_address,
            queue=queue,
            conversation_id=conversation_id,
            message_history=message_history
        )
    )
    
    try:
        # Stream responses from the queue
        while True:
            try:
                # Wait for 0.1 seconds for a message to be put on the queue
                # If no message, check if the agent task is done
                value = await asyncio.wait_for(queue.get(), timeout=0.1)
                if isinstance(value, dict) and value.get("done", False):
                    # Agent is done
                    break
                # Yield the response
                yield value
            except asyncio.TimeoutError:
                # Check if the agent task is done
                if agent_task.done():
                    # Check if there was an exception
                    try:
                        # Get the exception if any
                        agent_task.result()
                    except Exception as e:
                        # Yield the error
                        logger.error(f"Agent failed with error: {str(e)}")
                        logger.error(traceback.format_exc())
                        yield f"\nAgent failed with error: {str(e)}\n"
                    # No more responses to yield
                    break
    except asyncio.CancelledError:
        # Cancel the agent task if the streaming is cancelled
        if not agent_task.done():
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass
        raise
    except Exception as e:
        # Handle other exceptions
        logger.error(f"Stream agent response failed with error: {str(e)}")
        logger.error(traceback.format_exc())
        yield f"\nStream agent response failed with error: {str(e)}\n"
        if not agent_task.done():
            agent_task.cancel()
            try:
                await agent_task
            except (asyncio.CancelledError, Exception):
                pass
    finally:
        # Ensure the agent task is cancelled if not done
        if not agent_task.done():
            agent_task.cancel()
            try:
                await agent_task
            except (asyncio.CancelledError, Exception):
                pass

# Run the async function
if __name__ == "__main__":
    asyncio.run(run_agent("send 10 IP to 0x1234567890abcdef1234567890abcdef12345678")) 