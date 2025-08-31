# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools # type: ignore
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import os
import asyncio
import traceback
import sys  # noqa: F401 (may be used in future for sys.path or exits)
from typing import Dict, Optional, List
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langmem import create_memory_store_manager, create_manage_memory_tool, create_search_memory_tool
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global cache for checkpointers to persist memory across requests
CHECKPOINTER_CACHE: Dict[str, InMemorySaver] = {}

def get_checkpointer(conversation_id: str) -> InMemorySaver:
    """Get or create a checkpointer for a given conversation ID."""
    if conversation_id not in CHECKPOINTER_CACHE:
        logger.info(f"Creating new checkpointer for conversation_id: {conversation_id}")
        CHECKPOINTER_CACHE[conversation_id] = InMemorySaver()
    else:
        logger.info(f"Reusing existing checkpointer for conversation_id: {conversation_id}")
    return CHECKPOINTER_CACHE[conversation_id]

model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"), streaming=True)

system_prompt = """
    You are a specialized assistant focused on the Story protocol and blockchain analytics. 
    Only handle actions directly related to the Story protocol, blockchain, and the tools provided.
    Tools Provided:

        - get_transactions: Retrieves recent transactions for a specified address, with an optional limit on the number of transactions.
        - get_stats: Fetches current blockchain statistics, including total blocks, average block time, total transactions, and more.
        - get_address_overview: Provides a comprehensive overview of an address, including balance and contract status.
        - get_token_holdings: Lists all ERC-20 token holdings for a specified address, including detailed token information.
        - get_nft_holdings: Retrieves all NFT holdings for a given address, including collection information and metadata.
        - interpret_transaction: Provides a human-readable interpretation of a blockchain transaction based on its hash.

    IMPORTANT: If a tool fails with an error, acknowledge the error but try to provide helpful information anyway. Don't just say you can't help - explain what you tried to do and suggest alternatives.
    
    If the request is completely unrelated to Story protocol, blockchain, or IP topics, then return "I'm sorry, I can only help with the Story protocol and blockchain analytics using the tools I have access to. Check the information button for a list of available tools."
    
    Provide concise and clear analyses of findings using the available tools.
    Remember the coin is $IP (IP) and the data is coming from StoryScan/Blockscout API. 

    [IMPORTANT] Format blockchain statistics like this:
    
    **Blockchain Statistics:**
    
    - Total Blocks: [number]
    - Average Block Time: [number] seconds
    - Total Transactions: [number]
    - Total Addresses: [number]
    - IP Price: $[number]
    - Market Cap: $[number]
    - Network Utilization: [number]%
    
    **Gas Prices:**
    
    - Slow: [number] gwei
    - Average: [number] gwei
    - Fast: [number] gwei
    
    **Gas Used:**
    
    - Today: [number] gas
    - Total Gas Used: [number] gas
    [/IMPORTANT]
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

async def _run_with_storyscan_session(
    server_params: StdioServerParameters,
    user_message: str,
    queue: Optional[asyncio.Queue],
    conversation_id: Optional[str],
    message_history: Optional[List[Dict[str, str]]]
):
    """Open Storyscan MCP stdio session, load tools, and run the agent within that session."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            logger.info("Loading Storyscan MCP tools via stdio session")
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

            # Create checkpointer for conversation persistence
            checkpointer = get_checkpointer(conversation_id or "default")
            
            # Create agent with tools, memory, and checkpointer
            agent = create_react_agent(
                model,
                all_tools,
                prompt=system_prompt,
                checkpointer=checkpointer
            )

            # Prepare messages for agent - use single HumanMessage, let checkpointer handle history
            from langchain_core.messages import HumanMessage
            
            # Create single message for current user input - checkpointer handles conversation history
            human_message = HumanMessage(content=user_message)

            if queue:
                # Run agent with proper thread configuration for memory persistence
                logger.info("Starting agent with queue")
                thread_config = {
                    "configurable": {
                        "thread_id": conversation_id or "default"
                    }
                }
                result = await agent.ainvoke({"messages": [human_message]}, config=thread_config)
                logger.info("Agent completed execution")
                
                # Extract and send the AI response
                ai_response = None
                if result and "messages" in result:
                    # Find the last AI message in the result
                    for msg in reversed(result["messages"]):
                        if hasattr(msg, '__class__') and 'AI' in msg.__class__.__name__:
                            if hasattr(msg, 'content') and msg.content:
                                ai_response = msg.content
                                logger.info(f"Sending AI response: {ai_response[:100]}...")
                                # Send the complete response
                                await queue.put(ai_response)
                                break
                
                # Conversation is automatically stored by the checkpointer
                if conversation_id and ai_response:
                    logger.info(f"Conversation automatically persisted for thread_id: {conversation_id}")
                
                await queue.put({"done": True})
                return result

            else:
                # Run without streaming
                logger.info("Starting agent without streaming")
                thread_config = {
                    "configurable": {
                        "thread_id": conversation_id or "default"
                    }
                }
                result = await agent.ainvoke({"messages": [human_message]}, config=thread_config)
                logger.info("Agent completed execution without streaming")
                return result

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
    # If server_path exists from environment, use it as-is (no need to override)

    logger.info(f"Server path: {server_path}")
    
    if not os.path.exists(server_path):
        error_msg = f"Server file not found at {server_path}"
        logger.error(error_msg)
        if queue:
            await queue.put({"error": error_msg})
        raise FileNotFoundError(error_msg)
    
    server_params = StdioServerParameters(
        command="python3",
        args=[server_path],
    )

    try:
        # Run within a live Storyscan MCP stdio session (loads tools from storyscan-mcp)
        return await _run_with_storyscan_session(
            server_params=server_params,
            user_message=user_message,
            queue=queue,
            conversation_id=conversation_id,
            message_history=message_history
        )

    except Exception as e:
        logger.error(f"Error in agent execution: {str(e)}")
        logger.error(traceback.format_exc())
        if queue:
            await queue.put({"error": str(e)})
        raise

# Run the async function
if __name__ == "__main__":
    asyncio.run(run_agent("give me some blockchain stats"))