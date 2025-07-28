"""
Middleware for enforcing confirmation requirements on MCP tools.
This provides a scalable way to ensure user confirmation for sensitive operations.
"""

import re
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ToolWorkflowConfig:
    """Configuration for a tool's workflow requirements"""
    requires_confirmation: bool = False
    prerequisite_tools: List[str] = None
    confirmation_data_extractors: Dict[str, str] = None  # Maps data names to tool names
    confirmation_message_template: str = None

# Define workflow configurations for tools that need them
TOOL_WORKFLOWS: Dict[str, ToolWorkflowConfig] = {
    "mint_license_tokens": ToolWorkflowConfig(
        requires_confirmation=True,
        prerequisite_tools=["get_license_minting_fee", "get_license_revenue_share"],
        confirmation_data_extractors={
            "minting_fee": "get_license_minting_fee",
            "revenue_share": "get_license_revenue_share"
        },
        confirmation_message_template=(
            "This license requires a minting fee of {minting_fee} wei and has "
            "{revenue_share}% revenue share. Do you want to proceed?"
        )
    ),
    "mint_and_register_ip_with_terms": ToolWorkflowConfig(
        requires_confirmation=True,
        prerequisite_tools=["get_spg_nft_contract_minting_fee_and_token"],
        confirmation_data_extractors={
            "spg_fee": "get_spg_nft_contract_minting_fee_and_token"
        },
        confirmation_message_template=(
            "This SPG contract requires a minting fee of {spg_fee}. "
            "Do you want to proceed?"
        )
    ),
    "raise_dispute": ToolWorkflowConfig(
        requires_confirmation=True,
        prerequisite_tools=[],
        confirmation_data_extractors={},
        confirmation_message_template=(
            "You are about to raise a dispute with a bond of {bond_amount} wei. "
            "This action cannot be undone. Do you want to proceed?"
        )
    )
}

class ConfirmationMiddleware:
    """Middleware that intercepts tool calls and enforces confirmation requirements"""
    
    def __init__(self, confirmation_callback: Callable[[str], bool]):
        """
        Args:
            confirmation_callback: Function that presents message to user and returns True/False
        """
        self.confirmation_callback = confirmation_callback
        self.workflow_state: Dict[str, Any] = {}
        
    def extract_workflow_from_docstring(self, tool_name: str, docstring: str) -> Optional[ToolWorkflowConfig]:
        """
        Dynamically extract workflow requirements from tool docstring.
        This makes the system even more scalable - no need to hardcode workflows.
        """
        if not docstring:
            return None
            
        # Check if this tool has a workflow requirement
        if "🤖 AGENT WORKFLOW" not in docstring and "MANDATORY WORKFLOW" not in docstring:
            return None
            
        config = ToolWorkflowConfig()
        
        # Extract prerequisite tools
        prereq_pattern = r"(?:FIRST|call|Call)\s*[:\s]*(\w+)\("
        prerequisites = re.findall(prereq_pattern, docstring)
        if prerequisites:
            config.prerequisite_tools = prerequisites
            config.requires_confirmation = True
            
        # Check if confirmation is mentioned
        if any(phrase in docstring for phrase in [
            "confirmation", "confirm", "Do you want to proceed"
        ]):
            config.requires_confirmation = True
            
        return config if config.requires_confirmation else None
    
    async def intercept_tool_call(self, tool_name: str, tool_args: Dict[str, Any], 
                                  tool_callable: Callable, tool_docstring: str = None) -> Any:
        """
        Intercept a tool call and enforce confirmation if required.
        
        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments being passed to the tool
            tool_callable: The actual tool function
            tool_docstring: Tool's docstring (for dynamic workflow extraction)
            
        Returns:
            Tool result or raises exception if confirmation denied
        """
        # Try to get workflow config (hardcoded or dynamic)
        workflow_config = TOOL_WORKFLOWS.get(tool_name)
        
        # If no hardcoded config, try to extract from docstring
        if not workflow_config and tool_docstring:
            workflow_config = self.extract_workflow_from_docstring(tool_name, tool_docstring)
            
        # If no workflow requirements, just execute
        if not workflow_config or not workflow_config.requires_confirmation:
            return await tool_callable(**tool_args)
            
        # Check if prerequisites were completed
        if workflow_config.prerequisite_tools:
            missing_prereqs = [
                tool for tool in workflow_config.prerequisite_tools
                if tool not in self.workflow_state
            ]
            if missing_prereqs:
                raise ValueError(
                    f"Cannot call {tool_name} without first calling: {', '.join(missing_prereqs)}. "
                    "Please follow the required workflow."
                )
        
        # Build confirmation message
        if workflow_config.confirmation_message_template:
            # Fill in template with actual values
            template_data = {}
            
            # Get data from prerequisite tool results
            for data_name, source_tool in (workflow_config.confirmation_data_extractors or {}).items():
                if source_tool in self.workflow_state:
                    template_data[data_name] = self.workflow_state[source_tool]
                    
            # Add current tool arguments
            template_data.update(tool_args)
            
            confirmation_message = workflow_config.confirmation_message_template.format(**template_data)
        else:
            # Generic confirmation message
            confirmation_message = (
                f"You are about to execute {tool_name} with the following parameters:\n"
                f"{tool_args}\n\n"
                "This action will make blockchain transactions. Do you want to proceed?"
            )
            
        # Request confirmation
        logger.info(f"Requesting confirmation for {tool_name}")
        confirmed = await self.confirmation_callback(confirmation_message)
        
        if not confirmed:
            raise ValueError(f"User cancelled {tool_name} operation.")
            
        # Execute the tool
        result = await tool_callable(**tool_args)
        
        # Store result for potential use by dependent tools
        self.workflow_state[tool_name] = result
        
        return result
        
    def reset_workflow_state(self):
        """Reset workflow state between independent operations"""
        self.workflow_state = {}


# Example usage in the agent:
"""
# In sdk_mcp_agent.py, you could wrap tool calls:

async def get_user_confirmation(message: str) -> bool:
    # Send to queue for user confirmation
    await queue.put({
        "type": "confirmation_request",
        "message": message
    })
    
    # Wait for user response
    # (Implementation depends on your message handling)
    response = await wait_for_user_response()
    return response.lower() in ["yes", "y", "proceed", "confirm"]

# Create middleware instance
confirmation_middleware = ConfirmationMiddleware(get_user_confirmation)

# Wrap tool execution
async def execute_tool_with_confirmation(tool, **kwargs):
    return await confirmation_middleware.intercept_tool_call(
        tool_name=tool.name,
        tool_args=kwargs,
        tool_callable=tool.func,
        tool_docstring=tool.description
    )
""" 