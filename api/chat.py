import os
from typing import List, AsyncGenerator, Optional
from pydantic import BaseModel
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
import asyncio
import uuid
import logging

from .mcp_agent import run_agent as run_blockscout_agent
from .sdk_mcp_agent import run_agent as run_sdk_agent

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class Request(BaseModel):
    messages: List[Message]
    conversation_id: Optional[str] = None
    mcp_type: Optional[str] = "storyscan"  # Default to storyscan instead of blockscout
    wallet_address: Optional[str] = None    # Wallet address for SDK operations

async def stream_agent_response(
    messages: List[Message], 
    conversation_id: Optional[str] = None,
    mcp_type: str = "storyscan",  # Change default here too
    wallet_address: Optional[str] = None
) -> AsyncGenerator[str, None]:
    """Stream the agent's response"""
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
            
            # Choose the appropriate agent based on mcp_type
            if mcp_type == "sdk":
                logger.info(f"Using SDK MCP agent with wallet: {wallet_address}")
                await run_sdk_agent(
                    last_message, 
                    wallet_address=wallet_address,
                    queue=queue, 
                    conversation_id=conversation_id, 
                    message_history=formatted_messages
                )
            else:
                # Default to blockscout/storyscan agent
                logger.info("Using Storyscan MCP agent")
                await run_blockscout_agent(
                    last_message, 
                    queue=queue, 
                    conversation_id=conversation_id, 
                    message_history=formatted_messages
                )
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
                break
                
            if isinstance(item, dict):
                if "error" in item:
                    yield f"Error: {item['error']}\n"
                    break
                elif "done" in item:
                    # Handle the done signal from the agent
                    break
                elif "tool_call" in item or "tool_result" in item:
                    # Skip tool call/result objects - they are internal
                    continue
                # Forward any other dictionary items (like transaction requests) directly to the front-end
                else:
                    # This could be a transaction request or other structured data that should go to the front-end
                    yield item
            elif isinstance(item, str):
                # If it's a regular string, just send it
                yield item
    finally:
        if not agent_task.done():
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass

@app.post("/chat")
async def handle_chat(request: Request, protocol: str = Query('data')):
    logger.info(f"Received chat request with {len(request.messages)} messages")
    conversation_id = request.conversation_id or str(uuid.uuid4())
    mcp_type = request.mcp_type or "storyscan"
    wallet_address = request.wallet_address
    
    logger.info(f"Using MCP type: {mcp_type}, Wallet: {wallet_address or 'None'}")
    
    return StreamingResponse(
        stream_agent_response(
            request.messages, 
            conversation_id,
            mcp_type,
            wallet_address
        ),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Type': 'text/event-stream',
            'Transfer-Encoding': 'chunked',
            'x-vercel-ai-data-stream': 'v1'
        }
    )

