import os
from typing import List, AsyncGenerator, Optional
from pydantic import BaseModel
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
import asyncio
import uuid
import logging

from .mcp_agent import run_agent

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

async def stream_agent_response(messages: List[Message], conversation_id: Optional[str] = None) -> AsyncGenerator[str, None]:
    """Stream the agent's response"""
    # Get the last user message
    last_message = messages[-1].content if messages and messages[-1].role == "user" else ""
    
    logger.info(f"Processing user message: {last_message[:50]}...")
    
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
            await run_agent(last_message, queue, conversation_id, formatted_messages)
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
                
            if isinstance(item, dict) and "error" in item:
                yield f"Error: {item['error']}\n"
                break
            elif isinstance(item, dict):
                continue
            elif isinstance(item, str):
                if not (item.startswith('{') or item.startswith('e:{')):
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
    
    return StreamingResponse(
        stream_agent_response(request.messages, conversation_id),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Content-Type': 'text/event-stream',
            'Transfer-Encoding': 'chunked',
            'x-vercel-ai-data-stream': 'v1'
        }
    )

