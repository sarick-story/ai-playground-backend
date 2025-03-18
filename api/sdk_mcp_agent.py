# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools # type: ignore
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import os
import asyncio
import json
import uuid
import traceback
import sys
import time
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv
from langchain_core.callbacks.base import BaseCallbackHandler
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager, create_manage_memory_tool, create_search_memory_tool
from web3 import Web3
import atexit
import signal

load_dotenv()

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables to cache MCP client and tools
_mcp_session = None
_mcp_tools = None
_last_used = 0
_lock = asyncio.Lock()

# Import chain IDs from the utils directory
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from story_mcp_hub.utils.contract_addresses import CHAIN_IDS
    logger.info(f"Successfully imported CHAIN_IDS: {CHAIN_IDS}")
except ImportError as e:
    logger.warning(f"Failed to import CHAIN_IDS from utils.contract_addresses: {e}")
    # Fallback values if import fails
    CHAIN_IDS = {
        "sepolia": 11155111,
        "aeneid": 1315,
        "mainnet": 1514,
    }
    logger.info(f"Using fallback CHAIN_IDS: {CHAIN_IDS}")

async def get_or_create_mcp_session() -> Tuple[ClientSession, List[Any]]:
    """Get or create MCP session, reusing existing one if available."""
    global _mcp_session, _mcp_tools, _last_used
    
    async with _lock:
        current_time = time.time()
        
        # Reuse existing session if it exists and was used recently (within 5 minutes)
        if _mcp_session is not None and current_time - _last_used < 300:
            logger.info("Reusing existing MCP session")
            _last_used = current_time
            return _mcp_session, _mcp_tools
            
        # Find the server path
        server_path = os.environ.get("SDK_MCP_SERVER_PATH")
        if not server_path:
            # Local development path (relative to parent directory)
            server_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                "story-mcp-hub/story-sdk-mcp/server.py")
        
        logger.info(f"SDK MCP Server path: {server_path}")
        
        if not os.path.exists(server_path):
            error_msg = f"SDK MCP server file not found at {server_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Close existing session if any
        if _mcp_session is not None:
            logger.info("Closing existing MCP session")
            try:
                await _mcp_session.aclose()
            except Exception as e:
                logger.warning(f"Error closing existing session: {str(e)}")
                
        # Define the server params outside the try block
        server_params = StdioServerParameters(
            command="python",
            args=[server_path],
        )
        
        # Create a task in the current task context to avoid context switching issues
        current_task = asyncio.current_task()
        logger.info(f"Creating MCP session in task: {current_task.get_name() if current_task else 'unknown'}")
        
        # Use a simplified approach that avoids nested context managers
        try:
            # Start a subprocess directly instead of using the client context manager
            import subprocess
            import asyncio.subprocess
            
            logger.info(f"Starting subprocess: {server_params.command} {' '.join(server_params.args)}")
            process = await asyncio.create_subprocess_exec(
                server_params.command, 
                *server_params.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            if not process:
                raise RuntimeError("Failed to start subprocess")
                
            logger.info(f"Subprocess started with PID: {process.pid}")
            
            # Create streams from the process pipes
            from mcp.client.streams import AsyncStdioStream
            read_stream = AsyncStdioStream(process.stdout)
            write_stream = AsyncStdioStream(process.stdin)
            
            # Create session without using context managers that might cause issues
            logger.info("Creating session with direct streams")
            session = ClientSession(read_stream, write_stream)
            
            # Initialize the session
            logger.info("Initializing session")
            await session.initialize()
            logger.info("Session initialized successfully")
            
            # Load tools
            logger.info("Loading tools")
            tools = await load_mcp_tools(session)
            logger.info(f"Loaded {len(tools)} tools successfully")
            
            # Store the process for cleanup later
            session._process = process  # Store for cleanup
            
            # Cache the session and tools
            _mcp_session = session
            _mcp_tools = tools
            _last_used = current_time
            
            return session, tools
            
        except Exception as e:
            logger.error(f"Error creating MCP session: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Clean up if needed
            if 'process' in locals() and process:
                try:
                    process.terminate()
                    await process.wait()
                except Exception as e2:
                    logger.error(f"Error terminating process: {str(e2)}")
            
            raise

model = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.9,
)

system_prompt = """You are StoryAI, a helpful AI assistant for the Story Protocol ecosystem.

The user wants to send IP tokens to other addresses. Your main job is to help with this.

IMPORTANT: When a user messages you with ANY request that looks like sending tokens:
1. YOU MUST EXECUTE THE SEND_IP TOOL IMMEDIATELY - no exceptions
2. The format is usually "send X IP to ADDRESS" and you should call send_ip tool right away
3. Never respond with text first - call the tool immediately
4. Never ask for confirmation before executing the transaction

The send_ip tool takes these parameters:
- to_address: The recipient's Ethereum address (must start with 0x)
- amount: The amount of IP tokens to send (a number)

Example: If the user says "send 0.1 IP to 0x123...", immediately call the send_ip tool with those parameters.

For any request related to sending tokens, DO NOT respond with text - ALWAYS execute the tool directly.
"""

# Debug: Print confirmation
logger.info("SDK MCP agent initialized successfully")

async def create_transaction_request(to_address: str, amount: str, queue: asyncio.Queue) -> bool:
    """Create and send a transaction request directly for testing purposes."""
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
        
        # Format transaction parameters - proper '0x' hex format
        hex_value = "0x{:x}".format(wei_value)
        
        # Get chain ID from environment variable or use sepolia as default
        chain_name = os.getenv("CHAIN_NAME", "sepolia").lower()
        if chain_name in CHAIN_IDS:
            chain_id = CHAIN_IDS[chain_name]
        else:
            # If chain name not found in CHAIN_IDS, try to parse from CHAIN_ID env var
            chain_id = int(os.getenv("CHAIN_ID", "11155111"))  # Default to Sepolia if not specified
        
        logger.info(f"Using chain name: {chain_name}, chain ID: {chain_id}")
        
        # Create transaction object without any validation
        transaction_obj = {
            "action": "sign_transaction",
            "transaction": {
                "to": to_address,
                "value": hex_value,
                "data": "0x", 
                "chainId": chain_id,  # Use dynamic chain ID
                "gas": "0x5208"  # Standard 21000 gas
            },
            "message": f"Please send {amount} IP to {to_address}"
        }
        
        # Format as expected by frontend
        transaction_message = f"Transaction request: \n```json\n{json.dumps(transaction_obj)}\n```"
        logger.info(f"Direct transaction: Sending request: {transaction_message[:100]}...")
        
        # Send to queue
        await queue.put(transaction_message)
        await queue.put(f"\nPlease approve sending {amount} IP to {to_address} in your wallet.\n")
        return True
    except Exception as e:
        error_msg = f"Error creating direct transaction: {str(e)}"
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
    """Close the MCP session and terminate any subprocess."""
    global _mcp_session
    
    if _mcp_session is not None:
        logger.info("Closing MCP session during shutdown")
        try:
            # Try to close the session first
            await _mcp_session.aclose()
        except Exception as e:
            logger.warning(f"Error closing session: {str(e)}")
        
        # Then terminate the process if it exists
        if hasattr(_mcp_session, '_process') and _mcp_session._process:
            try:
                logger.info(f"Terminating subprocess with PID: {_mcp_session._process.pid}")
                _mcp_session._process.terminate()
                # Wait briefly for the process to terminate
                try:
                    await asyncio.wait_for(_mcp_session._process.wait(), 2.0)
                except asyncio.TimeoutError:
                    # If it doesn't terminate, kill it
                    logger.warning("Subprocess did not terminate, killing it")
                    _mcp_session._process.kill()
            except Exception as e:
                logger.warning(f"Error terminating subprocess: {str(e)}")
        
        _mcp_session = None

# Add these lines to ensure the session is properly closed when the application terminates
def cleanup_session():
    """Synchronous cleanup function for atexit hook."""
    if _mcp_session is not None and hasattr(_mcp_session, '_process') and _mcp_session._process:
        try:
            logger.info(f"Terminating subprocess with PID: {_mcp_session._process.pid}")
            _mcp_session._process.terminate()
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
    
    # Try direct transaction handling first
    if queue and wallet_address and "send" in user_message.lower() and "ip to" in user_message.lower():
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
                
                # Add 0x prefix if missing - the only validation we do
                if not to_address.startswith("0x"):
                    to_address = "0x" + to_address
                    logger.info(f"Added 0x prefix to address: {to_address}")
                
                logger.info(f"Parsed direct transaction: {amount} IP to {to_address}")
                
                # Try direct transaction
                await queue.put(f"Creating transaction to send {amount} IP to {to_address}...")
                success = await create_transaction_request(to_address, amount, queue)
                
                if success:
                    logger.info("Direct transaction request sent successfully")
                    await queue.put({"done": True})
                    return {"success": "Transaction request sent"}
                else:
                    logger.warning("Failed to send direct transaction, falling back to agent")
        except Exception as e:
            logger.error(f"Error parsing direct transaction: {str(e)}")
            logger.error(traceback.format_exc())
    
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
        
        # Get or create MCP session and tools
        session = None
        tools = None
        
        try:
            session, tools = await get_or_create_mcp_session()
        except Exception as e:
            error_msg = f"Failed to create MCP session: {str(e)}"
            logger.error(error_msg)
            if queue:
                await queue.put(f"Error: {error_msg}")
                await queue.put({"done": True})
            return {"error": error_msg}

        # Filter for just the send_ip tool
        send_ip_tool = [tool for tool in tools if tool.name == "send_ip"]
        
        if not send_ip_tool:
            error_msg = "send_ip tool not found, cannot process transaction request"
            logger.error(error_msg)
            if queue:
                await queue.put(f"Error: {error_msg}")
                await queue.put({"done": True})
            return {"error": error_msg}
        else:
            tool_names = [t.name for t in send_ip_tool]
            logger.info(f"Found send_ip tool: {tool_names}")
        
        # Inject wallet address into tool context if provided
        if wallet_address:
            logger.info(f"Using wallet address: {wallet_address}")
            # If our initial message is directly about sending tokens, add a debug message
            if "send" in user_message.lower() and "ip" in user_message.lower():
                logger.info("DETECTED TOKEN SEND REQUEST - SHOULD TRIGGER SEND_IP TOOL")
                if queue:
                    await queue.put("\nDetected token send request. Processing...\n")

        # Create agent with tools and message history
        logger.info("Creating agent with tools and system prompt")
        agent = create_react_agent(
            model,
            send_ip_tool,
            state_modifier=system_prompt
        )

        # Prepare messages for agent
        messages = message_history or [{"role": "user", "content": user_message}]
        logger.info(f"Prepared {len(messages)} messages for the agent")
        
        if queue:
            # Define the streaming handler directly before using it
            class StreamingCallbackHandler(BaseCallbackHandler):
                run_inline = True
                
                async def on_llm_new_token(self, token: str, **kwargs):
                    logger.debug(f"LLM token: {token}")
                    await queue.put(token)
                
                async def on_tool_start(self, tool_name: str, tool_input: Dict[str, Any], **kwargs):
                    tool_call_id = str(uuid.uuid4())
                    logger.info(f"Starting tool: {tool_name} with input: {tool_input}")
                    
                    # For send_ip tool, we need to include wallet info
                    if tool_name == "send_ip" and wallet_address:
                        try:
                            # Extract parameters from input
                            to_address = tool_input.get("to_address")
                            amount = tool_input.get("amount")
                            
                            logger.info(f"Processing send_ip request to {to_address} for {amount} IP")
                            
                            if to_address and amount:
                                # Add from_address to tool input for the actual API call
                                tool_input["from_address"] = wallet_address
                                
                                # Create transaction request in the format expected by the frontend
                                # See the frontend code in page.tsx that looks for "Transaction request:" marker
                                logger.info(f"Creating transaction request for {amount} IP to {to_address}")
                                
                                try:
                                    # Don't modify or validate the address at all
                                    logger.info(f"Using address as provided: {to_address}")
                                    
                                    # Convert amount to float first to ensure it's valid
                                    amount_float = float(amount)
                                    wei_value = Web3.to_wei(amount_float, "ether")
                                    logger.info(f"Converted {amount_float} IP to {wei_value} wei")
                                    
                                    # Format transaction parameters - proper '0x' hex format
                                    hex_value = "0x{:x}".format(wei_value)
                                    
                                    # Get chain ID based on environment configuration
                                    chain_name = os.getenv("CHAIN_NAME", "sepolia").lower()
                                    if chain_name in CHAIN_IDS:
                                        chain_id = CHAIN_IDS[chain_name]
                                    else:
                                        # If chain name not found in CHAIN_IDS, try to parse from CHAIN_ID env var
                                        chain_id = int(os.getenv("CHAIN_ID", "11155111"))  # Default to Sepolia
                                    
                                    logger.info(f"Tool starting: Using chain name: {chain_name}, chain ID: {chain_id}")
                                    
                                    # Create a transaction object that matches what MetaMask expects
                                    transaction_obj = {
                                        "action": "sign_transaction",
                                        "transaction": {
                                            "to": to_address,
                                            "value": hex_value,  # Proper hex string with 0x prefix
                                            "data": "0x",
                                            "chainId": chain_id,  # Use dynamic chain ID
                                            "gas": "0x5208"  # Standard 21000 gas
                                        },
                                        "message": f"Please send {amount} IP to {to_address}"
                                    }
                                    
                                    # Send as a specially formatted string the frontend is looking for
                                    transaction_message = f"Transaction request: \n```json\n{json.dumps(transaction_obj)}\n```"
                                    logger.info(f"Sending transaction request: {transaction_message[:100]}...")
                                    await queue.put(transaction_message)
                                    
                                    # Also send a user-friendly message
                                    await queue.put(f"\nPlease approve sending {amount} IP to {to_address} in your wallet.\n")
                                    
                                except ValueError as e:
                                    error_msg = f"Invalid amount value: {str(e)}"
                                    logger.error(error_msg)
                                    await queue.put(f"\nError: {error_msg}\n")
                            else:
                                error_msg = f"Missing required parameters. to_address: {to_address}, amount: {amount}"
                                logger.error(error_msg)
                                await queue.put(f"\nError: {error_msg}\n")
                                
                        except Exception as e:
                            error_msg = f"Error preparing transaction: {str(e)}"
                            logger.error(error_msg)
                            logger.error(traceback.format_exc())
                            try:
                                await queue.put(f"\nError: {error_msg}\n")
                            except:
                                pass
                    else:
                        error_msg = f"Missing required parameters. to_address: {to_address}, amount: {amount}"
                        logger.error(error_msg)
                        await queue.put(f"\nError: {error_msg}\n")
                                
                        # Send tool call info for internal tracking
                        try:
                            await queue.put({
                                "tool_call": {
                                    "id": tool_call_id,
                                    "name": tool_name,
                                    "args": tool_input
                                }
                            })
                        except Exception as e:
                            logger.error(f"Error sending tool call info: {str(e)}")
                    
                    return tool_call_id
                
                async def on_tool_end(self, output: str, **kwargs):
                    tool_call_id = kwargs.get("run_id", str(uuid.uuid4()))
                    tool_name = kwargs.get("name", "unknown_tool")
                    tool_input = kwargs.get("input", {})
                    
                    logger.info(f"Tool completed: {tool_name}")
                    logger.debug(f"Tool output: {output}")
                    
                    output_str = (output.content if hasattr(output, "content")
                                else str(output) if hasattr(output, "__str__")
                                else "Tool execution completed")
                    
                    # Send tool result info to frontend
                    await queue.put({
                        "tool_result": {
                            "id": tool_call_id,
                            "name": tool_name,
                            "args": tool_input,
                            "result": output_str
                        }
                    })
            
            callbacks = [StreamingCallbackHandler()]
            
            # Run agent with streaming
            logger.info(f"Starting SDK MCP agent with streaming in task context: {task_token}")
            try:
                result = await agent.ainvoke(
                    {"messages": messages},
                    config={"callbacks": callbacks}
                )
                logger.info("SDK MCP agent completed execution with streaming")
                logger.info(f"Agent result: {result}")
                await queue.put({"done": True})
                return result
            except Exception as e:
                error_msg = f"Error during agent execution: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                await queue.put(f"\nError: {error_msg}\n")
                await queue.put({"done": True})
                return {"error": error_msg}
        
        else:
            # Run without streaming
            logger.info("Starting SDK MCP agent without streaming")
            try:
                result = await agent.ainvoke({"messages": messages})
                logger.info("SDK MCP agent completed execution without streaming")
                logger.info(f"Agent result: {result}")
                return result
            except Exception as e:
                error_msg = f"Error during agent execution: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
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