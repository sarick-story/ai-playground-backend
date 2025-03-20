import os
from typing import List, AsyncGenerator, Optional, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import asyncio
import uuid
import logging
import json
import re

from .mcp_agent import run_agent as run_blockscout_agent
from .sdk_mcp_agent import run_agent as run_sdk_agent, create_transaction_request

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

class TransactionRequest(BaseModel):
    to_address: str
    amount: str
    wallet_address: str
    private_key: Optional[str] = None

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
                    # Instead of forwarding the dictionary directly, convert it to a string
                    # This ensures consistent data format for the frontend parser
                    yield json.dumps(item)
            elif isinstance(item, str):
                # Check if this is a specially formatted tool call/result message
                if item.startswith("__INTERNAL_TOOL_CALL__") and item.endswith("__END_INTERNAL__"):
                    # Extract and process the internal tool call, but don't forward to the user
                    continue
                elif item.startswith("__INTERNAL_TOOL_RESULT__") and item.endswith("__END_INTERNAL__"):
                    # Extract and process the internal tool result, but don't forward to the user
                    continue
                # For transaction intents, we'll just pass the regular formatted text
                # and handle the transaction in a separate API endpoint
                else:
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
    
    # Get the last user message
    last_message = request.messages[-1].content if request.messages and request.messages[-1].role == "user" else ""
    
    # Check if this is a transaction request for the SDK MCP
    if mcp_type == "sdk" and wallet_address and ("send" in last_message.lower() and "ip to" in last_message.lower()):
        logger.info("Detected transaction request, redirecting to transaction endpoint")
        
        # Parse the transaction details
        pattern = r"send\s+([0-9.]+)\s+ip\s+to\s+(0x[a-fA-F0-9]+)"
        match = re.search(pattern, last_message, re.IGNORECASE)
        
        # Try alternative pattern if first one doesn't match
        if not match:
            logger.info("First pattern didn't match, trying alternative pattern")
            # More lenient pattern that doesn't require 0x prefix
            pattern = r"send\s+([0-9.]+)\s+ip\s+to\s+([a-fA-F0-9]+)"
            match = re.search(pattern, last_message, re.IGNORECASE)
            
            # Even more lenient pattern as last resort
            if not match:
                pattern = r"send\s+([0-9.]+)\s+ip\s+to\s+(.*?)($|\s|\.)"
                match = re.search(pattern, last_message, re.IGNORECASE)
        
        if match:
            amount = match.group(1)
            to_address = match.group(2).strip()
            
            # Ensure address has 0x prefix
            if not to_address.startswith('0x'):
                to_address = f"0x{to_address}"
            
            logger.info(f"Parsed transaction: {amount} IP to {to_address}")
            
            # Return a direct JSON response with transaction details
            return JSONResponse(content={
                "is_transaction": True,
                "transaction_details": {
                    "to_address": to_address,
                    "amount": amount,
                    "wallet_address": wallet_address
                },
                "message": f"I'll send {amount} IP to {to_address}. Please approve the transaction in your wallet."
            })
        else:
            logger.warning("Transaction intent detected but couldn't parse the details")
    
    # For non-transaction requests, continue with streaming as before
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

@app.post("/transaction")
async def handle_transaction(request: TransactionRequest):
    """Dedicated endpoint for transaction handling"""
    logger.info(f"Received transaction request to: {request.to_address}, amount: {request.amount}")
    
    if not request.wallet_address:
        raise HTTPException(status_code=400, detail="Wallet address is required")
    
    try:
        # Create a queue for response
        queue = asyncio.Queue()
        
        # Create the transaction directly
        success = await create_transaction_request(
            to_address=request.to_address,
            amount=request.amount,
            queue=queue,
            private_key=request.private_key
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create transaction")
        
        # Format the transaction for wagmi
        # For Ethereum, the standard gas limit for simple transfers is 21000
        gas_limit = 21000
        
        # Return a proper JSON response for the transaction
        return JSONResponse(content={
            "transaction": {
                "to": request.to_address,
                "value": request.amount,
                "data": "0x",
                "gas": hex(gas_limit)  # Convert to hex string for wagmi
            },
            "message": f"Transaction to send {request.amount} IP to {request.to_address}"
        })
        
    except Exception as e:
        logger.error(f"Error handling transaction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

