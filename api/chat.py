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
import traceback

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
        # Use plain text for error message
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
        # Use simple text streaming - no JSON wrapping, no SSE format
        logger.info(f"Starting raw text streaming for {mcp_type} MCP")
        
        while True:
            item = await queue.get()
            
            if item is None:
                logger.info("Received None item, ending stream")
                break
                
            if isinstance(item, dict) and "error" in item:
                error_message = f"Error: {item['error']}"
                logger.info(f"Received error: {error_message}")
                yield error_message
                break
            elif isinstance(item, dict) and "tool_result" in item:
                # Extract and stream formatted tool results
                tool_result = item["tool_result"]["result"]
                tool_name = item["tool_result"]["name"]
                logger.info(f"Processing tool result from {tool_name}")
                
                # Special handling for get_stats tool results
                if tool_name == "get_stats" and isinstance(tool_result, dict) and "content" in tool_result:
                    # This handles the specific format of blockchain statistics
                    logger.info("Processing get_stats tool result with content structure")
                    content_items = tool_result["content"]
                    for content_item in content_items:
                        if isinstance(content_item, dict) and "text" in content_item:
                            formatted_text = content_item["text"]
                            logger.info(f"Extracted stats text: {formatted_text[:50]}...")
                            yield formatted_text
                else:
                    # Handle general tool results
                    if not isinstance(tool_result, str):
                        # Check if this is a structured result with raw data
                        if isinstance(tool_result, dict):
                            # Try to extract only the meaningful text content
                            if "raw_data" in tool_result and "content" in tool_result:
                                logger.info("Found raw_data structure, extracting content")
                                content = tool_result.get("content", [])
                                extracted = False
                                if isinstance(content, list):
                                    for item in content:
                                        if isinstance(item, dict) and "text" in item:
                                            # Use just the text content, not the raw data
                                            tool_result = item["text"]
                                            extracted = True
                                            break
                                if not extracted:
                                    # Convert to string but remove the raw_data section
                                    tool_result_dict = dict(tool_result)
                                    if "raw_data" in tool_result_dict:
                                        del tool_result_dict["raw_data"]
                                    tool_result = str(tool_result_dict)
                            else:
                                # Default conversion
                                tool_result = str(tool_result)
                        else:
                            tool_result = str(tool_result)
                    
                    # Filter the result as we do with other strings
                    filtered_result = ''.join(ch for ch in tool_result if ord(ch) >= 32 or ch in '\n\r\t')
                    logger.info(f"Streaming tool result: {filtered_result[:50]}...")
                    
                    # Final check - don't send JSON objects directly to the client
                    if filtered_result.startswith('{') and filtered_result.endswith('}'):
                        try:
                            data = json.loads(filtered_result)
                            if "content" in data and isinstance(data["content"], list):
                                for item in data["content"]:
                                    if isinstance(item, dict) and "text" in item:
                                        filtered_result = item["text"]
                                        break
                        except:
                            pass
                    
                    yield filtered_result
            elif isinstance(item, dict):
                # For direct dictionary responses without going through tool_result
                logger.info(f"Processing direct dictionary response: {str(item)[:50]}...")
                
                # If this is a direct API response like get_stats, extract the text
                if "content" in item:
                    logger.info("Direct response contains content field")
                    content = item["content"]
                    if isinstance(content, list):
                        for content_item in content:
                            if isinstance(content_item, dict) and "text" in content_item:
                                text = content_item["text"]
                                filtered_text = ''.join(ch for ch in text if ord(ch) >= 32 or ch in '\n\r\t')
                                logger.info(f"Yielding text from content item: {filtered_text[:50]}...")
                                yield filtered_text
                                # Skip other processing
                                continue
                    elif isinstance(content, str):
                        filtered_content = ''.join(ch for ch in content if ord(ch) >= 32 or ch in '\n\r\t')
                        logger.info(f"Yielding content string: {filtered_content[:50]}...")
                        yield filtered_content
                        # Skip other processing
                        continue
                
                # Skip other dictionary items that don't have text content we can extract
                logger.info("Skipping dictionary item without extractable text content")
                continue
            elif isinstance(item, str):
                # Check if this is a string containing direct JSON
                if item.startswith('{') and ('"content":' in item or '"raw_data":' in item):
                    logger.info("String appears to contain structured JSON data, attempting to extract text content")
                    try:
                        data = json.loads(item)
                        if "content" in data and isinstance(data["content"], list):
                            for content_item in data["content"]:
                                if isinstance(content_item, dict) and "text" in content_item:
                                    filtered_item = content_item["text"]
                                    yield filtered_item
                                    continue
                    except:
                        logger.warning("Failed to parse JSON from string", exc_info=True)
                
                # Proceed with normal string processing if not handled above
                # Skip internal/control messages
                if not (item.startswith('{') or item.startswith('e:{') or item.startswith('__INTERNAL')):
                    # Simply yield the raw text without any JSON formatting or SSE data: prefix
                    filtered_item = ''.join(ch for ch in item if ord(ch) >= 32 or ch in '\n\r\t')
                    yield filtered_item
    
    except Exception as e:
        logger.error(f"Error in stream_agent_response: {str(e)}")
        logger.error(traceback.format_exc())
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
    
    # Check if this is a transaction request for the SDK MCP
    # ONLY for SDK MCP, not Storyscan
    if mcp_type == "sdk" and wallet_address and ("send" in last_message.lower() and "ip to" in last_message.lower()):
        logger.info("Detected SDK MCP transaction request, processing as JSON response")
        
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

