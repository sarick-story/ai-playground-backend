# Tool sets for each agent
IP_ASSET_TOOLS = set([
    "mint_and_register_ip_with_terms",
    "register", 
    "upload_image_to_ipfs",
    "create_ip_metadata",
    "get_spg_nft_minting_token"
])

IP_ACCOUNT_TOOLS = set([
    "get_erc20_balance",
    "mint_test_erc20_tokens"
])

LICENSE_TOOLS = set([
    "get_license_terms",
    "mint_license_tokens",
    "attach_license_terms",
    "predict_minting_license_fee"
])

NFT_CLIENT_TOOLS = set([
    "create_spg_nft_collection",
    "mint_and_register_ip_with_terms",
    "get_spg_nft_minting_token"
])

DISPUTE_TOOLS = set([
    "raise_dispute"
])

GROUP_TOOLS = set([])  # currently no tools available

PERMISSION_TOOLS = set([])  # currently no tools available

ROYALTY_TOOLS = set([
    "pay_royalty_on_behalf",
    "claim_all_revenue"
])

WIP_TOOLS = set([
    "deposit_wip",
    "transfer_wip"
])

# Cache for categorized tools
_categorized_tools_cache = None

def categorize_tools(mcp_tools):
    """
    Categorize tools using the defined sets and cache the result.
    
    Args:
        mcp_tools: Dictionary of tool_name -> tool_object
        
    Returns:
        Dictionary of category_name -> list of tool_objects
    """
    global _categorized_tools_cache
    
    # Return cached result if available
    if _categorized_tools_cache is not None:
        return _categorized_tools_cache
    
    # Initialize empty tool lists
    ip_asset_tool = []
    ip_account_tool = []
    license_tool = []
    nft_client_tool = []
    dispute_tool = []
    group_tool = []
    permission_tool = []
    royalty_tool = []
    wip_tool = []
    
    # For loop to categorize tools
    for tool in mcp_tools:
        if tool.name in IP_ASSET_TOOLS:
            ip_asset_tool.append(tool)
        if tool.name in IP_ACCOUNT_TOOLS:
            ip_account_tool.append(tool)
        if tool.name in LICENSE_TOOLS:
            license_tool.append(tool)
        if tool.name in NFT_CLIENT_TOOLS:
            nft_client_tool.append(tool)
        if tool.name in DISPUTE_TOOLS:
            dispute_tool.append(tool)
        if tool.name in GROUP_TOOLS:
            group_tool.append(tool)
        if tool.name in PERMISSION_TOOLS:
            permission_tool.append(tool)
        if tool.name in ROYALTY_TOOLS:
            royalty_tool.append(tool)
        if tool.name in WIP_TOOLS:
            wip_tool.append(tool)
    
    # Create and cache the result
    _categorized_tools_cache = {
        "ip_asset_tool": ip_asset_tool,
        "ip_account_tool": ip_account_tool,
        "license_tool": license_tool,
        "nft_client_tool": nft_client_tool,
        "dispute_tool": dispute_tool,
        "group_tool": group_tool,
        "permission_tool": permission_tool,
        "royalty_tool": royalty_tool,
        "wip_tool": wip_tool,
    }
    
    return _categorized_tools_cache
