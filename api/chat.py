import os
from typing import List, AsyncGenerator, Optional
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
    mcp_type: Optional[str] = "storyscan"  # Default to storyscan
    wallet_address: Optional[str] = None    # Wallet address for SDK operations

class TransactionRequest(BaseModel):
    to_address: str
    amount: str
    wallet_address: str
    private_key: Optional[str] = None

class InterruptConfirmationRequest(BaseModel):
    interrupt_id: str
    conversation_id: str
    confirmed: bool
    wallet_address: Optional[str] = None

def is_transaction_intent(message: str) -> bool:
    """Detect if a message contains a transaction intent"""
    # Simple pattern matching for 'send X IP to ADDRESS'
    patterns = [
        r"send\s+([0-9.]+)\s+ip\s+to\s+(0x[a-fA-F0-9]+)",
        r"send\s+([0-9.]+)\s+ip\s+to\s+([a-fA-F0-9]+)",
        r"send\s+([0-9.]+)\s+ip\s+to\s+(.*?)($|\s|\.)"
    ]
    
    for pattern in patterns:
        if re.search(pattern, message, re.IGNORECASE):
            return True
    return False

def parse_transaction_details(message: str) -> dict:
    """Extract transaction details from a message"""
    # Try each pattern in order of specificity
    patterns = [
        r"send\s+([0-9.]+)\s+ip\s+to\s+(0x[a-fA-F0-9]+)",
        r"send\s+([0-9.]+)\s+ip\s+to\s+([a-fA-F0-9]+)",
        r"send\s+([0-9.]+)\s+ip\s+to\s+(.*?)($|\s|\.)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            amount = match.group(1)
            to_address = match.group(2).strip()
            
            # Ensure address has 0x prefix
            if not to_address.startswith('0x'):
                to_address = f"0x{to_address}"
                
            return {
                "amount": amount,
                "to_address": to_address
            }
    
    # Default fallback - shouldn't reach here if is_transaction_intent is used first
    return {"amount": "0", "to_address": "0x0"}

async def stream_agent_response(
    messages: List[Message], 
    conversation_id: Optional[str] = None,
    mcp_type: str = "storyscan",
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
                logger.info("End of stream")
                break
                
            if isinstance(item, dict) and "error" in item:
                yield f"Error: {item['error']}"
                break
            
            if isinstance(item, str):
                # Skip any JSON-like structures, but allow normal text through
                if not (item.startswith('{') or item.startswith('e:{')):
                    yield item
    
    except Exception as e:
        logger.error(f"Error in stream_agent_response: {str(e)}")
        yield f"Error: {str(e)}"
            
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
    
    # For SDK MCP: Quick check if this is a transaction request before running the full agent
    if mcp_type == "sdk" and wallet_address and is_transaction_intent(last_message):
        logger.info("Detected transaction intent in SDK MCP request")
        tx_details = parse_transaction_details(last_message)
        
        # Return a JSON response with transaction details
        return JSONResponse(content={
            "is_transaction": True,
            "transaction_details": {
                "to_address": tx_details["to_address"],
                "amount": tx_details["amount"],
                "wallet_address": wallet_address
            },
            "message": f"I'll send {tx_details['amount']} IP to {tx_details['to_address']}. Please approve the transaction in your wallet."
        })
    
    # For regular messages, use streaming
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
            'x-acme-stream-format': 'vercel-ai',
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

@app.post("/interrupt/confirm")
async def handle_interrupt_confirmation(request: InterruptConfirmationRequest):
    """Handle interrupt confirmation from frontend and resume execution."""
    logger.info(f"Received interrupt confirmation: {request.interrupt_id}, confirmed: {request.confirmed}")
    
    try:
        # Import supervisor system to handle resume
        from .supervisor_agent_system import resume_interrupted_conversation
        
        # Resume the conversation with the user's decision
        result = await resume_interrupted_conversation(
            conversation_id=request.conversation_id,
            interrupt_id=request.interrupt_id,
            confirmed=request.confirmed,
            wallet_address=request.wallet_address
        )
        
        return JSONResponse(content={
            "status": "resumed" if request.confirmed else "cancelled",
            "conversation_id": request.conversation_id,
            "result": result
        })
        
    except Exception as e:
        logger.error(f"Error handling interrupt confirmation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

