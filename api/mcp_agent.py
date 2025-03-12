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
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from langchain_core.callbacks.base import BaseCallbackHandler
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager, create_manage_memory_tool, create_search_memory_tool
import openai

load_dotenv()

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), streaming=True)

system_prompt = """
    You are a specialized assistant chat bot focused on the Story protocol and blockchain analytics. 
    Only handle actions directly related to the Story protocol, blockchain, and the tools provided.
    Tools Provided:

        - check_balance: Checks the balance of a given address on the blockchain.
        - get_transactions: Retrieves recent transactions for a specified address, with an optional limit on the number of transactions.
        - get_stats: Fetches current blockchain statistics, including total blocks, average block time, total transactions, and more.
        - get_address_overview: Provides a comprehensive overview of an address, including balance and contract status.
        - get_token_holdings: Lists all ERC-20 token holdings for a specified address, including detailed token information.
        - get_nft_holdings: Retrieves all NFT holdings for a given address, including collection information and metadata.
        - interpret_transaction: Provides a human-readable interpretation of a blockchain transaction based on its hash.

    If it is unrelated to these topics, return: "I'm sorry, I can only help with the Story protocol and blockchain analytics."
    
    Provide concise and clear analyses of findings using the available tools.
    Remember the coin is $IP and the data is coming from blockscout API.
"""

# Initialize memory store
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
)

# Create memory manager with proper namespace template
memory_manager = create_memory_store_manager(
    "openai:gpt-4o-mini",
    namespace=("memories", "{conversation_id}"),  # Use templated namespace
    query_limit=5,  # Limit to 5 threads per conversation
    enable_inserts=True,
    enable_deletes=True
)

# Debug: Print confirmation
logger.info("Memory store and manager initialized successfully")

# Add a function to clean up old memories
async def cleanup_old_memories(conversation_id: Optional[str] = None, max_memories: int = 20):
    """
    Clean up old memories to maintain a maximum number per conversation.
    
    Args:
        conversation_id: Optional ID to track the conversation
        max_memories: Maximum number of memories to keep per conversation
    """
    try:
        # Fix namespace search syntax
        namespace = ("memories", conversation_id) if conversation_id else ("memories",)
        
        # Get all memories for this namespace
        all_memories = store.search(
            namespace,  # Remove namespace= keyword
            query="",  # Add empty query to search all
            limit=1000
        )
        
        # If we have more than max_memories, delete the oldest ones
        if len(all_memories) > max_memories:
            # Sort by creation time (oldest first)
            sorted_memories = sorted(all_memories, key=lambda x: x.created_at)
            
            # Calculate how many to delete
            to_delete = len(sorted_memories) - max_memories
            
            logger.info(f"Cleaning up {to_delete} old memories for conversation {conversation_id}")
            
            # Delete the oldest memories
            for i in range(to_delete):
                store.delete(sorted_memories[i].namespace, sorted_memories[i].key)
                
            logger.info(f"Cleanup complete, kept {max_memories} most recent memories")
    except Exception as e:
        logger.error(f"Error cleaning up old memories: {str(e)}")

# Add a new method to process memories directly
async def process_memory(messages, conversation_id=None):
    """
    Process and store memories from messages.
    
    Args:
        messages: List of message objects with role and content
        conversation_id: Optional conversation ID
    """
    try:
        conv_id = conversation_id or "default"
        
        # Format messages correctly for memory_manager
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                # If already a dict with role/content
                formatted_messages.append(msg)
            else:
                # If a tuple or other format, convert to dict
                formatted_messages.append({
                    "role": msg[0] if isinstance(msg, tuple) else "user",
                    "content": msg[1] if isinstance(msg, tuple) else str(msg)
                })
        
        # Use memory_manager to process
        await memory_manager.ainvoke({
            "messages": formatted_messages,
            "conversation_id": conv_id
        })
        
        return True
    except Exception as e:
        logger.error(f"Error processing memory: {str(e)}")
        logger.error(traceback.format_exc())
        return None

async def run_agent(
    user_message: str, 
    queue: Optional[asyncio.Queue] = None, 
    conversation_id: Optional[str] = None,
    message_history: Optional[List[Dict[str, str]]] = None
):
    """Run the agent with the given user message."""
    logger.info(f"Starting run_agent with message: {user_message[:50]}...")

    # Update the server path to handle both local development and Docker environments
    server_path = os.environ.get("MCP_SERVER_PATH")
    if not server_path:
        # Local development path (relative to parent directory)
        server_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                              "story-mcp-hub/storyscan-mcp/server.py")
    else:
        # Docker environment - use absolute path from environment variable
        server_path = "/app/story-mcp-hub/storyscan-mcp/server.py"

    logger.info(f"Server path: {server_path}")
    
    if not os.path.exists(server_path):
        error_msg = f"Server file not found at {server_path}"
        logger.error(error_msg)
        if queue:
            await queue.put({"error": error_msg})
        raise FileNotFoundError(error_msg)
    
    server_params = StdioServerParameters(
        command="python",
        args=[server_path],
    )

    try:
        logger.info("Initializing stdio client")
        async with stdio_client(server_params) as (read, write):
            logger.info("Stdio client initialized, creating session")
            async with ClientSession(read, write) as session:
                # Initialize the connection
                logger.info("Initializing MCP session")
                await session.initialize()
                logger.info("MCP session initialized successfully")

                # Get tools
                logger.info("Loading MCP tools")
                tools = await load_mcp_tools(session)
                logger.info(f"Loaded {len(tools)} MCP tools")
                
                # Log tool names for debugging
                tool_names = [tool.name for tool in tools]
                logger.info(f"Available tools: {', '.join(tool_names)}")

                # Create memory tools with proper namespace
                memory_tools = [
                    create_manage_memory_tool(
                        namespace=("memories", conversation_id or "default"),
                        store=store
                    ),
                    create_search_memory_tool(
                        namespace=("memories", conversation_id or "default"),
                        store=store
                    )
                ]

                all_tools = tools + memory_tools

                # Create agent with tools and message history
                agent = create_react_agent(
                    model,
                    all_tools,
                    state_modifier=system_prompt
                )

                # Prepare messages for agent
                messages = message_history or [{"role": "user", "content": user_message}]
                
                if queue:
                    # Streaming callback handler definition remains the same
                    class StreamingCallbackHandler(BaseCallbackHandler):
                        run_inline = True
                        
                        async def on_llm_new_token(self, token: str, **kwargs):
                            logger.debug(f"LLM token: {token}")
                            await queue.put(token)
                        
                        async def on_tool_start(self, tool_name: str, tool_input: Dict[str, Any], **kwargs):
                            tool_call_id = str(uuid.uuid4())
                            logger.info(f"Starting tool: {tool_name} with input: {tool_input}")
                            await queue.put({
                                "tool_call": {
                                    "id": tool_call_id,
                                    "name": tool_name,
                                    "args": tool_input
                                }
                            })
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
                    logger.info("Starting agent with streaming")
                    try:
                        result = await agent.ainvoke(
                            {"messages": messages},
                            config={"callbacks": callbacks}
                        )
                        logger.info("Agent completed execution with streaming")
                        await queue.put({"done": True})
                        return result
                    except openai.RateLimitError as e:
                        # Format OpenAI rate limit as an agent message
                        error_message = "I'm sorry, but I've reached my API rate limit. Please try again in a few moments. If this persists, you may need to check your OpenAI account quota."
                        logger.error(f"OpenAI Rate Limit Error: {str(e)}")
                        
                        if queue:
                            # Send the error as a regular message token by token
                            for token in error_message.split():
                                await queue.put(token + " ")
                            await queue.put({"done": True})
                        
                        # Return formatted as an agent response
                        return {
                            "messages": messages + [{"role": "assistant", "content": error_message}]
                        }
                    except Exception as e:
                        # Format other exceptions as agent messages
                        error_message = f"I encountered an issue while processing your request: {str(e)}"
                        logger.error(f"Error during agent execution: {str(e)}")
                        logger.error(traceback.format_exc())
                        
                        if queue:
                            # Send the error as a regular message token by token
                            for token in error_message.split():
                                await queue.put(token + " ")
                            await queue.put({"done": True})
                        
                        # Return formatted as an agent response
                        return {
                            "messages": messages + [{"role": "assistant", "content": error_message}]
                        }
                
                else:
                    # Run without streaming
                    logger.info("Starting agent without streaming")
                    result = await agent.ainvoke({"messages": messages})
                    logger.info("Agent completed execution without streaming")
                    return result

    except Exception as e:
        # Handle exceptions from the stdio client setup
        error_message = f"I'm having trouble connecting to my tools. Please try again later. Technical details: {str(e)}"
        logger.error(f"Error in agent execution: {str(e)}")
        logger.error(traceback.format_exc())
        
        if queue:
            # Send the error as a regular message token by token
            for token in error_message.split():
                await queue.put(token + " ")
            await queue.put({"done": True})
        
        # Return formatted as an agent response
        return {
            "messages": messages + [{"role": "assistant", "content": error_message}]
        }

# Run the async function
if __name__ == "__main__":
    asyncio.run(run_agent("give me some blockchain stats"))