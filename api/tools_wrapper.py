import os
from typing import List, Callable, Tuple, Any, Union, Dict, Optional
import inspect
import uuid
import json
from typing_extensions import Annotated

# Load SDK MCP tools the same way as in sdk_mcp_agent.py (stdio session + load_mcp_tools)
from mcp import ClientSession, StdioServerParameters  # type: ignore
from mcp.client.stdio import stdio_client  # type: ignore
from langchain_mcp_adapters.tools import load_mcp_tools  # type: ignore

from langgraph.prebuilt import interrupt, InjectedStore, InjectedState  # Import interrupt, InjectedStore and InjectedState
from langgraph.store.base import BaseStore
from langgraph.types import Command
from .interrupt_handler import create_simple_confirmation_interrupt, create_transaction_interrupt, FeeInformation, send_standard_interrupt





async def load_sdk_mcp_tools() -> List:  # returns a list of tool callables compatible with LangChain/LangGraph
    """Load SDK MCP tools via stdio MCP session (same pattern as sdk_mcp_agent.py).

    Resolution order for server path:
    - SDK_MCP_SERVER_PATH env var, if set
    - Default relative path: ../../story-mcp-hub/story-sdk-mcp/server.py
    """
    server_path = os.environ.get("SDK_MCP_SERVER_PATH")
    if not server_path:
        # Resolve relative to this file: <repo>/ai-playground-backend/api/.. /.. /story-mcp-hub/story-sdk-mcp/server.py
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        candidate = os.path.join(repo_root, "story-mcp-hub", "story-sdk-mcp", "server.py")
        if os.path.exists(candidate):
            server_path = candidate
        else:
            raise FileNotFoundError("Could not find story-sdk-mcp server.py. Set SDK_MCP_SERVER_PATH to override.")

    server_params = StdioServerParameters(
        command="python3",
        args=[server_path],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            return tools

# Tools will be loaded and organized by name when needed
_tools_cache = None

async def get_tools_by_name():
    """Load MCP tools and organize them by name for easy access."""
    global _tools_cache
    if _tools_cache is None:
        tools_list = await load_sdk_mcp_tools()
        _tools_cache = {tool.name: tool for tool in tools_list}
    return _tools_cache



def create_simple_confirmation_wrapper(
    original_tool: Callable,
    tool_description: Optional[str] = None
):
    """Create a simple wrapper that interrupts to show parameters before executing the tool.
    
    This is for tools that don't need prechecks - just user confirmation.
    
    Args:
        original_tool: The tool to wrap with confirmation
        tool_description: Optional description of what the tool does
        
    Returns:
        A wrapped tool that interrupts for confirmation
    """
    
    # Determine if original tool is async
    is_async = inspect.iscoroutinefunction(original_tool)
    
    async def async_wrapped(*args, **kwargs):
        """Async wrapper with simple interrupt."""
        
        # Get tool name for both functions and StructuredTool objects
        tool_name = getattr(original_tool, '__name__', getattr(original_tool, 'name', 'tool'))
        
        # Create standardized interrupt message
        parameters_dict = {
            "args": list(args) if args else [],
            "kwargs": dict(kwargs) if kwargs else {}
        }
        
        interrupt_msg = create_simple_confirmation_interrupt(
            tool_name=tool_name,
            parameters=parameters_dict
        )
        
        # Send standardized interrupt
        send_standard_interrupt(interrupt_msg)
        
        # Execute after confirmation
        # For StructuredTool objects, need to handle parameter passing correctly
        if hasattr(original_tool, 'invoke'):
            # StructuredTool - use invoke method with proper parameter handling
            # LangChain may pass args in kwargs, so we need to extract them
            if 'args' in kwargs and isinstance(kwargs['args'], (list, tuple)):
                # If args are passed in kwargs, use them as positional arguments
                actual_args = kwargs['args']
                remaining_kwargs = {k: v for k, v in kwargs.items() if k != 'args'}
                tool_input = {}
                if actual_args:
                    # For single argument tools, pass the first argument directly
                    if len(actual_args) == 1:
                        # Use the original tool's expected parameter name
                        if hasattr(original_tool, 'args_schema') and original_tool.args_schema:
                            # Get the first field name from the schema
                            field_names = list(original_tool.args_schema.model_fields.keys())
                            if field_names:
                                tool_input[field_names[0]] = actual_args[0]
                            else:
                                tool_input = actual_args[0]
                        else:
                            tool_input = actual_args[0]
                    else:
                        tool_input = actual_args
                tool_input.update(remaining_kwargs)
            else:
                # Use provided args and kwargs as-is
                tool_input = {**kwargs}
                for i, arg in enumerate(args):
                    tool_input[f'arg_{i}'] = arg
            
            if is_async:
                return await original_tool.ainvoke(tool_input)
            else:
                return original_tool.invoke(tool_input)
        else:
            # Regular function - call directly
            if is_async:
                return await original_tool(*args, **kwargs)
            else:
                return original_tool(*args, **kwargs)
    
    def sync_wrapped(*args, **kwargs):
        """Sync wrapper with simple interrupt."""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(async_wrapped(*args, **kwargs))
        else:
            # Already in a loop, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, async_wrapped(*args, **kwargs))
                return future.result()
    
    # Choose wrapper based on original tool type
    # For StructuredTool objects, always use async wrapper since they're async by nature
    if hasattr(original_tool, 'invoke'):
        wrapped = async_wrapped
    else:
        wrapped = async_wrapped if is_async else sync_wrapped
    # Handle both function and StructuredTool objects
    wrapped.__name__ = getattr(original_tool, '__name__', getattr(original_tool, 'name', 'wrapped_tool'))
    wrapped.__doc__ = getattr(original_tool, '__doc__', getattr(original_tool, 'description', ''))
    
    return wrapped


def wrap_mint_and_register_ip_with_terms(
    mint_and_register_ip_original: Callable
):
    """Specific wrapper for mint_and_register_ip_with_terms that handles SPG contract fee prechecks.
    
    This wrapper:
    1. Checks if spg_nft_contract is provided (not None)
    2. If None (default SPG contract): sets fee=0, token=WIP (0x1514000000000000000000000000000000000000)
    3. If custom contract: runs get_spg_nft_contract_minting_fee_and_token precheck
    4. Shows fee information to user via interrupt
    5. On confirmation, injects the fee parameters and executes
    
    Args:
        mint_and_register_ip_original: The original mint_and_register_ip_with_terms tool
        
    Returns:
        Wrapped tool with automatic fee detection and confirmation
    """
    
    async def wrapped_mint_and_register_ip(
        commercial_rev_share: int,
        derivatives_allowed: bool,
        registration_metadata: dict,
        commercial_use: bool = True,
        minting_fee: int = 0,
        recipient: Optional[str] = None,
        spg_nft_contract: Optional[str] = None,
        spg_nft_contract_max_minting_fee: Optional[int] = None,
        spg_nft_contract_mint_fee_token: Optional[str] = None,
        store: Annotated[BaseStore, InjectedStore] = None
    ):
        """Wrapped mint_and_register_ip_with_terms with SPG fee detection."""
        
        # Generate cache key for this invocation
        cache_key = f"mint_register_{spg_nft_contract}_{hash((commercial_rev_share, derivatives_allowed))}"
        
        # Check if we have cached precheck results (for re-execution after interrupt)
        cached_fee_info = None
        if store:
            try:
                namespace = ("spg_fee_prechecks",)
                cached = store.get(namespace, cache_key)
                if cached:
                    cached_fee_info = json.loads(cached.value)
            except Exception:
                pass
        
        # Determine fee information based on SPG contract
        fee_info = None
        
        if spg_nft_contract is None:
            # Default SPG contract - no fees needed
            fee_info = {
                "minting_fee": 0,
                "fee_token": "0x1514000000000000000000000000000000000000",  # WIP token address
                "fee_display": "0 wei (default SPG contract - no fee)"
            }
        elif not spg_nft_contract_max_minting_fee:
            # Custom SPG contract - need to get fee information
            if cached_fee_info:
                # Use cached info on re-execution
                fee_info = cached_fee_info
            else:
                # Load the fee checking tool and run the precheck
                try:
                    # Load MCP tools to get the fee checking tool
                    mcp_tools = await load_sdk_mcp_tools()
                    get_spg_fee_tool = None
                    
                    for tool in mcp_tools:
                        if hasattr(tool, 'name') and tool.name == "get_spg_nft_contract_minting_fee_and_token":
                            get_spg_fee_tool = tool
                            break
                    
                    if get_spg_fee_tool:
                        fee_result = await get_spg_fee_tool.invoke({
                            "spg_nft_contract": spg_nft_contract
                        })
                        
                        # Parse the result to extract fee and token
                        # The actual format will depend on the tool's return value
                        fee_info = {
                            "minting_fee": fee_result.get("fee", 0),
                            "fee_token": fee_result.get("token", "0x1514000000000000000000000000000000000000"),
                            "fee_display": fee_result.get("fee_display", f"{fee_result.get('fee', 0)} wei")
                        }
                        
                        # Cache the fee info
                        if store:
                            try:
                                namespace = ("spg_fee_prechecks",)
                                store.put(namespace, cache_key, json.dumps(fee_info))
                            except Exception:
                                pass
                    else:
                        raise Exception("get_spg_nft_contract_minting_fee_and_token tool not found")
                            
                except Exception as e:
                    # If precheck fails, use defaults
                    print(f"Warning: Failed to get SPG contract fees: {e}")
                    fee_info = {
                        "minting_fee": 0,
                        "fee_token": "0x1514000000000000000000000000000000000000",
                        "fee_display": "0 wei (fallback - fee check failed)"
                    }
        else:
            # Fee information already provided by user
            fee_info = {
                "minting_fee": spg_nft_contract_max_minting_fee,
                "fee_token": spg_nft_contract_mint_fee_token or "0x1514000000000000000000000000000000000000",
                "fee_display": f"{spg_nft_contract_max_minting_fee} wei (user provided)"
            }
        
        # Prepare interrupt message with all information
        interrupt_params = {
            "commercial_rev_share": commercial_rev_share,
            "derivatives_allowed": derivatives_allowed,
            "registration_metadata": registration_metadata,
            "commercial_use": commercial_use,
            "minting_fee": minting_fee,
            "recipient": recipient or "sender address",
            "spg_nft_contract": spg_nft_contract or "default SPG contract"
        }
        
        # Add fee information
        display_info = {}
        if fee_info:
            display_info["spg_contract_fee"] = fee_info["fee_display"]
            display_info["fee_token"] = fee_info["fee_token"]
            
            if fee_info["minting_fee"] == 0:
                display_info["total_cost"] = f"SPG fee: {fee_info['fee_display']} + License minting fee: {minting_fee} wei"
                display_info["contract_type"] = "Default SPG contract" if spg_nft_contract is None else "Custom SPG contract (no fee)"
            else:
                display_info["total_cost"] = f"SPG fee: {fee_info['fee_display']} + License minting fee: {minting_fee} wei"
                display_info["contract_type"] = "Custom SPG contract (with fees)"
        
        # Create standardized interrupt message with fee information
        fee_information = None
        if fee_info and fee_info.get("minting_fee", 0) > 0:
            fee_information = FeeInformation(
                fee_amount=str(fee_info["minting_fee"]),
                fee_token=fee_info["fee_token"],
                fee_display=fee_info["fee_display"],
                total_cost=display_info.get("total_cost")
            )
        
        interrupt_msg = create_transaction_interrupt(
            tool_name="mint_and_register_ip_with_terms",
            operation="Mint NFT, Register IP, and Attach License Terms",
            parameters=interrupt_params,
            fee_info=fee_information,
            gas_estimate="~300,000 gas"
        )
        
        # Send standardized interrupt
        send_standard_interrupt(interrupt_msg)
        
        # After confirmation, prepare final parameters
        final_params = {
            "commercial_rev_share": commercial_rev_share,
            "derivatives_allowed": derivatives_allowed,
            "registration_metadata": registration_metadata,
            "commercial_use": commercial_use,
            "minting_fee": minting_fee,
            "recipient": recipient,
            "spg_nft_contract": spg_nft_contract
        }
        
        # Inject fee parameters based on contract type
        if fee_info:
            if spg_nft_contract is None:
                # Default SPG contract - set the fee parameters to 0 and WIP token
                final_params["spg_nft_contract_max_minting_fee"] = 0
                final_params["spg_nft_contract_mint_fee_token"] = "0x1514000000000000000000000000000000000000"
            else:
                # Custom SPG contract - use the determined fee information
                final_params["spg_nft_contract_max_minting_fee"] = fee_info["minting_fee"]
                final_params["spg_nft_contract_mint_fee_token"] = fee_info["fee_token"]
        
        # Clean up cache after confirmation
        if store and fee_info:
            try:
                namespace = ("spg_fee_prechecks",)
                store.delete(namespace, cache_key)
            except Exception:
                pass
        
        # Execute the original tool with all parameters
        result = await mint_and_register_ip_original(**final_params)
        
        return result
    
    # Preserve metadata
    wrapped_mint_and_register_ip.__name__ = "mint_and_register_ip_with_terms"
    wrapped_mint_and_register_ip.__doc__ = mint_and_register_ip_original.__doc__
    
    return wrapped_mint_and_register_ip


def wrap_mint_license_tokens(
    mint_license_tokens_original: Callable,
    get_license_minting_fee: Callable,
    get_license_revenue_share: Callable
):
    """Specific wrapper for mint_license_tokens that handles license fee and revenue share prechecks.
    
    This wrapper:
    1. Runs get_license_minting_fee and get_license_revenue_share prechecks
    2. Shows fee and revenue share information to user via interrupt
    3. On confirmation, injects the fee parameters and executes
    
    Args:
        mint_license_tokens_original: The original mint_license_tokens tool
        get_license_minting_fee: Tool to get license minting fee
        get_license_revenue_share: Tool to get revenue share percentage
        
    Returns:
        Wrapped tool with automatic fee detection and confirmation
    """
    
    async def wrapped_mint_license_tokens(
        licensor_ip_id: str,
        license_terms_id: int,
        receiver: Optional[str] = None,
        amount: int = 1,
        max_minting_fee: Optional[int] = None,
        max_revenue_share: Optional[int] = None,
        license_template: Optional[str] = None,
        store: Annotated[BaseStore, InjectedStore] = None
    ):
        """Wrapped mint_license_tokens with fee and revenue share prechecks."""
        
        # Generate cache key for this invocation
        cache_key = f"mint_license_{license_terms_id}_{hash((licensor_ip_id, amount))}"
        
        # Check if we have cached precheck results
        cached_precheck_info = None
        if store:
            try:
                namespace = ("license_fee_prechecks",)
                cached = store.get(namespace, cache_key)
                if cached:
                    cached_precheck_info = json.loads(cached.value)
            except Exception:
                pass
        
        # Run prechecks if not provided by user
        precheck_info = None
        if not max_minting_fee or not max_revenue_share:
            if cached_precheck_info:
                # Use cached info on re-execution
                precheck_info = cached_precheck_info
            else:
                # Run both prechecks
                try:
                    # Get minting fee
                    fee_result = await get_license_minting_fee(
                        license_terms_id=license_terms_id
                    )
                    
                    # Get revenue share
                    revenue_result = await get_license_revenue_share(
                        license_terms_id=license_terms_id
                    )
                    
                    # Parse results (format depends on actual tool returns)
                    precheck_info = {
                        "minting_fee": fee_result.get("fee", 0),
                        "revenue_share": revenue_result.get("share", 0),
                        "fee_display": fee_result.get("display", "0 wei"),
                        "share_display": revenue_result.get("display", "0%")
                    }
                    
                    # Cache the precheck info
                    if store:
                        try:
                            namespace = ("license_fee_prechecks",)
                            store.put(namespace, cache_key, json.dumps(precheck_info))
                        except Exception:
                            pass
                            
                except Exception as e:
                    print(f"Warning: Failed to run prechecks: {e}")
                    # Use provided values or defaults
                    precheck_info = {
                        "minting_fee": max_minting_fee or 0,
                        "revenue_share": max_revenue_share or 0,
                        "fee_display": f"{max_minting_fee or 0} wei",
                        "share_display": f"{max_revenue_share or 0}%"
                    }
        
        # Prepare interrupt message
        interrupt_params = {
            "licensor_ip_id": licensor_ip_id,
            "license_terms_id": license_terms_id,
            "receiver": receiver or "caller address",
            "amount": amount,
            "license_template": license_template or "default template"
        }
        
        # Add precheck information
        fee_display = {}
        if precheck_info:
            fee_display["minting_fee"] = precheck_info["fee_display"]
            fee_display["revenue_share"] = precheck_info["share_display"]
            fee_display["total_cost"] = f"{precheck_info['minting_fee'] * amount} wei for {amount} tokens"
        elif max_minting_fee and max_revenue_share:
            fee_display["minting_fee"] = f"{max_minting_fee} wei"
            fee_display["revenue_share"] = f"{max_revenue_share}%"
            fee_display["total_cost"] = f"{max_minting_fee * amount} wei for {amount} tokens"
        
        # Create standardized interrupt message with fee information
        fee_information = None
        if precheck_info:
            fee_information = FeeInformation(
                fee_amount=str(precheck_info["minting_fee"]),
                fee_token="WIP",  # License tokens are typically paid in WIP
                fee_display=precheck_info["fee_display"],
                total_cost=f"{precheck_info['minting_fee'] * amount} wei for {amount} tokens"
            )
        elif max_minting_fee:
            fee_information = FeeInformation(
                fee_amount=str(max_minting_fee),
                fee_token="WIP",
                fee_display=f"{max_minting_fee} wei",
                total_cost=f"{max_minting_fee * amount} wei for {amount} tokens"
            )
        
        interrupt_msg = create_transaction_interrupt(
            tool_name="mint_license_tokens",
            operation="Mint License Tokens",
            parameters=interrupt_params,
            fee_info=fee_information,
            gas_estimate="~200,000 gas"
        )
        
        # Send standardized interrupt
        send_standard_interrupt(interrupt_msg)
        
        # Prepare final parameters
        final_params = {
            "licensor_ip_id": licensor_ip_id,
            "license_terms_id": license_terms_id,
            "receiver": receiver,
            "amount": amount,
            "license_template": license_template
        }
        
        # Inject precheck parameters if we have them
        if precheck_info:
            final_params["max_minting_fee"] = precheck_info["minting_fee"]
            final_params["max_revenue_share"] = precheck_info["revenue_share"]
        else:
            final_params["max_minting_fee"] = max_minting_fee
            final_params["max_revenue_share"] = max_revenue_share
        
        # Clean up cache
        if store and precheck_info:
            try:
                namespace = ("license_fee_prechecks",)
                store.delete(namespace, cache_key)
            except Exception:
                pass
        
        # Execute the original tool
        result = await mint_license_tokens_original(**final_params)
        
        return result
    
    # Preserve metadata
    wrapped_mint_license_tokens.__name__ = "mint_license_tokens"
    wrapped_mint_license_tokens.__doc__ = mint_license_tokens_original.__doc__
    
    return wrapped_mint_license_tokens


# Example usage functions
async def example_usage():
    """Example of how to use the specific wrappers."""
    
    # Load original tools from MCP
    tools = await load_sdk_mcp_tools()
    
    # Find the specific tools we need
    mint_and_register_ip_original = None
    mint_license_tokens_original = None
    get_license_fee_tool = None
    get_license_share_tool = None
    
    for tool in tools:
        if tool.name == "mint_and_register_ip_with_terms":
            mint_and_register_ip_original = tool
        elif tool.name == "mint_license_tokens":
            mint_license_tokens_original = tool
        elif tool.name == "get_license_minting_fee":
            get_license_fee_tool = tool
        elif tool.name == "get_license_revenue_share":
            get_license_share_tool = tool
    
    # Create wrapped versions
    if mint_and_register_ip_original:
        wrapped_mint_and_register = wrap_mint_and_register_ip_with_terms(
            mint_and_register_ip_original
        )
    
    if mint_license_tokens_original and get_license_fee_tool and get_license_share_tool:
        wrapped_mint_license = wrap_mint_license_tokens(
            mint_license_tokens_original,
            get_license_fee_tool,
            get_license_share_tool
        )
    
    return {
        "mint_and_register_wrapped": wrapped_mint_and_register,
        "mint_license_wrapped": wrapped_mint_license
    }

async def create_wrapped_tool_collections_from_tools(mcp_tools):
    """Create tool collections with confirmation wrappers from pre-loaded MCP tools."""
    # Organize tools by name
    tools = {tool.name: tool for tool in mcp_tools}
    return _create_tool_collections(tools)

async def create_wrapped_tool_collections():
    """Create tool collections with confirmation wrappers."""
    tools = await get_tools_by_name()
    return _create_tool_collections(tools)

def _create_tool_collections(tools):
    """Internal function to create tool collections from tools dictionary."""
    # Helper function to safely wrap tools
    def safe_wrap(tool_name, description=None):
        tool = tools.get(tool_name)
        if tool:
            return create_simple_confirmation_wrapper(tool, description)
        else:
            print(f"Warning: Tool '{tool_name}' not found in MCP tools")
            return None
    
    # Create tool collections
    dispute_tools = []
    if tools.get("raise_dispute"):
        dispute_tools.append(safe_wrap("raise_dispute", "Raise a dispute against an IP asset"))
    
    group_tools = []  # No tools in this category yet
    
    ip_account_tools = []
    for tool_name in ["get_erc20_token_balance", "mint_test_erc20_tokens"]:
        wrapped = safe_wrap(tool_name)
        if wrapped:
            ip_account_tools.append(wrapped)
    
    ip_asset_tools = []
    ip_asset_tool_names = [
        "mint_and_register_ip_with_terms",
        "register", 
        "upload_image_to_ipfs",
        "create_ip_metadata"
    ]
    for tool_name in ip_asset_tool_names:
        wrapped = safe_wrap(tool_name)
        if wrapped:
            ip_asset_tools.append(wrapped)
    
    license_tools = []
    license_tool_names = [
        "get_license_terms",
        "mint_license_tokens", 
        "attach_license_terms"
    ]
    for tool_name in license_tool_names:
        wrapped = safe_wrap(tool_name)
        if wrapped:
            license_tools.append(wrapped)
    
    nft_client_tools = []
    nft_tool_names = [
        "create_spg_nft_collection",
        "get_spg_nft_contract_minting_fee_and_token"
    ]
    for tool_name in nft_tool_names:
        wrapped = safe_wrap(tool_name)
        if wrapped:
            nft_client_tools.append(wrapped)
    
    permission_tools = []  # No tools in this category yet
    
    royalty_tools = []
    royalty_tool_names = ["pay_royalty_on_behalf", "claim_all_revenue"]
    for tool_name in royalty_tool_names:
        wrapped = safe_wrap(tool_name)
        if wrapped:
            royalty_tools.append(wrapped)
    
    wip_tools = []
    wip_tool_names = ["deposit_wip", "transfer_wip"]
    for tool_name in wip_tool_names:
        wrapped = safe_wrap(tool_name)
        if wrapped:
            wip_tools.append(wrapped)
    
    return {
        "dispute_tool": dispute_tools,
        "group_tool": group_tools,
        "ip_account_tool": ip_account_tools,
        "ip_asset_tool": ip_asset_tools,
        "license_tool": license_tools,
        "nft_client_tool": nft_client_tools,
        "permission_tool": permission_tools,
        "royalty_tool": royalty_tools,
        "wip_tool": wip_tools
    }

# Cache for tool collections
_tool_collections_cache = None

async def get_tool_collections():
    """Get cached tool collections or create them if not cached."""
    global _tool_collections_cache
    if _tool_collections_cache is None:
        _tool_collections_cache = await create_wrapped_tool_collections()
    return _tool_collections_cache

# For backwards compatibility, provide individual access functions
async def get_dispute_tools():
    collections = await get_tool_collections()
    return collections["dispute_tool"]

async def get_group_tools():
    collections = await get_tool_collections()
    return collections["group_tool"]

async def get_ip_account_tools():
    collections = await get_tool_collections()
    return collections["ip_account_tool"]

async def get_ip_asset_tools():
    collections = await get_tool_collections()
    return collections["ip_asset_tool"]

async def get_license_tools():
    collections = await get_tool_collections()
    return collections["license_tool"]

async def get_nft_client_tools():
    collections = await get_tool_collections()
    return collections["nft_client_tool"]

async def get_permission_tools():
    collections = await get_tool_collections()
    return collections["permission_tool"]

async def get_royalty_tools():
    collections = await get_tool_collections()
    return collections["royalty_tool"]

async def get_wip_tools():
    collections = await get_tool_collections()
    return collections["wip_tool"]