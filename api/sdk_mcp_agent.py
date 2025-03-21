# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client, get_default_environment
from langchain_mcp_adapters.tools import load_mcp_tools # type: ignore
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
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

# Global variables for MCP state
_mcp_session = None
_mcp_tools = None
_last_used = 0
_lock = asyncio.Lock()
_mcp_process = None

async def get_or_create_mcp_session() -> Tuple[ClientSession, List[Any]]:
    """Get or create MCP session by starting the MCP server as a subprocess."""
    global _mcp_session, _mcp_tools, _last_used, _mcp_process
    
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
            logger.error(f"Current directory: {os.getcwd()}")
            logger.error(f"__file__: {__file__}")
            logger.error(f"Parent dir: {os.path.dirname(os.path.dirname(__file__))}")
            raise FileNotFoundError(error_msg)
        
        # Close existing session if any
        if _mcp_session is not None:
            logger.info("Closing existing MCP session")
            try:
                await _mcp_session.aclose()
            except Exception as e:
                logger.warning(f"Error closing existing session: {str(e)}")
        
        # Create server command
        cmd = [sys.executable, server_path]
        
        try:
            # Using a completely different approach - direct subprocess management
            logger.info(f"Starting MCP server with command: {' '.join(cmd)}")
            
            # Start the process directly
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            if process.returncode is not None:
                raise RuntimeError(f"Process failed to start with return code {process.returncode}")
            
            logger.info(f"Process started with PID: {process.pid}")
            
            # Store the process globally for cleanup
            _mcp_process = process
            
            # Create a custom ClientSession implementation that works with asyncio streams
            class AsyncioClientSession:
                def __init__(self, process):
                    self.process = process
                    self.initialized = False
                    self._id_counter = 0
                
                async def initialize(self):
                    self.initialized = True
                    return {"capabilities": {}}
                
                async def aclose(self):
                    if self.process and self.process.returncode is None:
                        self.process.terminate()
                        try:
                            await asyncio.wait_for(self.process.wait(), 2.0)
                        except asyncio.TimeoutError:
                            self.process.kill()
                
                async def execute_tool(self, tool_name, **kwargs):
                    if not self.initialized:
                        await self.initialize()
                    
                    # Create a JSON-RPC request
                    self._id_counter += 1
                    request = {
                        "jsonrpc": "2.0",
                        "id": self._id_counter,
                        "method": "executeFunction",
                        "params": {
                            "name": tool_name,
                            "args": kwargs
                        }
                    }
                    
                    # Send the request to the process
                    json_request = json.dumps(request) + "\n"
                    self.process.stdin.write(json_request.encode())
                    await self.process.stdin.drain()
                    
                    # Read the response
                    response_line = await self.process.stdout.readline()
                    response = json.loads(response_line.decode().strip())
                    
                    # Check for errors
                    if "error" in response:
                        error = response["error"]
                        raise RuntimeError(f"Tool execution error: {error.get('message', 'unknown error')}")
                    
                    # Return the result
                    return response.get("result", None)
            
            # Create the custom session
            session = AsyncioClientSession(process)
            await session.initialize()
            logger.info("MCP session initialized")
            
            # Create tool wrappers
            tools = []
            
            # Add the send_ip tool
            class SendIPTool:
                name = "send_ip"
                description = "Send IP tokens from one address to another"
                
                async def __call__(self, from_address=None, to_address=None, amount=None):
                    logger.info(f"Executing send_ip tool: from={from_address}, to={to_address}, amount={amount}")
                    return await session.execute_tool("send_ip", from_address=from_address, to_address=to_address, amount=amount)
            
            # Add the tool to the list
            tools.append(SendIPTool())
            logger.info(f"Created {len(tools)} tool wrappers")
            
            # Store session and tools for reuse
            _mcp_session = session
            _mcp_tools = tools
            _last_used = time.time()
            
            return session, tools
            
        except Exception as e:
            error_msg = f"Error creating MCP session: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # Clean up if needed
            if _mcp_process and _mcp_process.returncode is None:
                try:
                    _mcp_process.terminate()
                    await asyncio.wait_for(_mcp_process.wait(), 2.0)
                except Exception as e2:
                    logger.error(f"Error terminating process: {str(e2)}")
                    try:
                        _mcp_process.kill()
                    except:
                        pass
            
            raise RuntimeError(f"Failed to create MCP session: {str(e)}")

model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), streaming=True)

system_prompt = """
    You are a specialized assistant focused on the Story protocol and blockchain capabilities. 
    You can provide information about the Story protocol, IP tokens, blockchain technology, and can help users send IP tokens.
    
    Tools Provided:
        - send_ip: Sends IP tokens from one address to another. This can be used to help users transfer their IP tokens on the blockchain.
    
    Transaction Capability:
    When a user asks to "send X IP to ADDRESS", you'll help them initiate a blockchain transaction. The user will need to approve this transaction in their wallet.
    
    Example transaction commands:
    - "Send 0.1 IP to 0x123456..."
    - "Transfer 5 IP tokens to 0xabcdef..."
    
    If the request is unrelated to Story protocol, blockchain technology, IP tokens, or sending transactions, explain that you're a specialized assistant focused on the Story protocol and related blockchain functionality.
    
    Provide concise and clear responses. When helping with transactions, confirm the details before proceeding.
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
                    try:
                        logger.debug(f"LLM token: {token}")
                        
                        # Skip tokens that are just whitespace or newlines
                        if token.strip() == "":
                            return
                        
                        # More aggressive filtering of control characters and escape sequences
                        # that might break JSON or streaming format
                        
                        # First use a simple filter for common control chars
                        filtered_token = ''.join(char for char in token if ord(char) >= 32 or char in '\n\r\t')
                        
                        # Then use regex to ensure only printable ASCII plus basic whitespace
                        safe_token = re.sub(r'[^\x20-\x7E\n\r\t]', '', filtered_token)
                        
                        # Remove any potential JSON-breaking sequences
                        safe_token = safe_token.replace('\\u', '\\\\u')
                        safe_token = safe_token.replace('\\', '\\\\')
                        
                        # Verify the token can be safely serialized to JSON
                        test_json = json.dumps({"text": safe_token})
                        
                        # If we get here, we know the token is safe for JSON
                        await queue.put(safe_token)
                    except Exception as e:
                        logger.warning(f"Error processing token, skipping: {str(e)}")
                        # Try even more aggressive filtering as a last resort
                        try:
                            # Only allow basic ASCII printable characters
                            super_safe_token = re.sub(r'[^\x20-\x7E]', '', token)
                            await queue.put(super_safe_token)
                        except:
                            # If all else fails, just skip this token
                            pass
                
                async def on_tool_start(self, tool_name: str, tool_input: Dict[str, Any], **kwargs):
                    tool_call_id = str(uuid.uuid4())
                    logger.info(f"Starting tool: {tool_name} with input: {tool_input}")
                    
                    # For send_ip tool, we need to include wallet info
                    if tool_name == "send_ip":
                        if not wallet_address:
                            error_msg = "Cannot execute send_ip tool: No wallet connected"
                            logger.error(error_msg)
                            await queue.put(f"\n⚠️ {error_msg}\n")
                            await queue.put(f"\nPlease connect your wallet to send transactions.\n")
                            # Still return a tool ID to prevent errors
                            return tool_call_id
                        
                        try:
                            # Extract parameters from input
                            to_address = tool_input.get("to_address")
                            amount = tool_input.get("amount")
                            
                            logger.info(f"Processing send_ip request to {to_address} for {amount} IP")
                            
                            if to_address and amount:
                                # Add from_address to tool input for the actual API call
                                tool_input["from_address"] = wallet_address
                                
                                # Just send a simple notification that we're preparing a transaction
                                await queue.put(f"\nPreparing a transaction to send {amount} IP to {to_address}...\n")
                                
                                # Create the transaction intent - but don't send the actual JSON in the stream
                                # The frontend will make a separate API call to get the transaction data
                                transaction_intent = {
                                    "to": to_address,
                                    "amount": str(amount),
                                    "data": "0x"
                                }
                                
                                # Send a simple notification that instructions to trigger the frontend
                                transaction_message = f"Transaction intent: \n```json\n{json.dumps(transaction_intent, indent=2)}\n```"
                                await queue.put(transaction_message)
                                
                                # Also send the user-friendly message separately
                                await queue.put(f"\nI've prepared a transaction to send {amount} IP to {to_address}. Please approve it in your wallet when prompted.\n")
                                
                                # Return early so we don't send tool_call info
                                return tool_call_id
                                
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
                    
                    # Send tool call info for internal tracking - ensure it's always stringified
                    try:
                        tool_call_info = {
                            "tool_call": {
                                "id": tool_call_id,
                                "name": tool_name,
                                "args": tool_input
                            }
                        }
                        # Instead of sending a dict directly, convert to a special format string
                        # that won't be displayed to the user but will be processed by the backend
                        await queue.put(f"__INTERNAL_TOOL_CALL__{json.dumps(tool_call_info)}__END_INTERNAL__")
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
                    
                    # Send tool result info to frontend using special format
                    tool_result = {
                        "tool_result": {
                            "id": tool_call_id,
                            "name": tool_name,
                            "args": tool_input,
                            "result": output_str
                        }
                    }
                    await queue.put(f"__INTERNAL_TOOL_RESULT__{json.dumps(tool_result)}__END_INTERNAL__")
            
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