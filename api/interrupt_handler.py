"""
Standardized interrupt handling for LangGraph dynamic interrupts.
"""

import json
import uuid
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, asdict
from langgraph.prebuilt import interrupt


class InterruptType(str, Enum):
    """Types of interrupts that can occur."""
    TOOL_CONFIRMATION = "tool_confirmation"
    TRANSACTION_APPROVAL = "transaction_approval"
    PARAMETER_CONFIRMATION = "parameter_confirmation"
    ERROR_RECOVERY = "error_recovery"


@dataclass
class FeeInformation:
    """Standardized fee information structure."""
    fee_amount: str
    fee_token: str
    fee_display: str
    total_cost: Optional[str] = None
    additional_fees: Optional[Dict[str, str]] = None


@dataclass
class BlockchainImpact:
    """Information about blockchain transaction impact."""
    action: str
    network: str
    estimated_gas: Optional[str] = None
    transaction_type: Optional[str] = None
    affects_balance: bool = True


@dataclass
class StandardInterruptMessage:
    """Standardized interrupt message format for frontend consumption."""
    
    # Core identification
    interrupt_id: str
    interrupt_type: InterruptType
    timestamp: str
    
    # Tool/operation information
    tool_name: str
    operation: str
    description: str
    
    # Parameters and context
    parameters: Dict[str, Any]
    message: str
    
    # Optional detailed information
    fee_information: Optional[FeeInformation] = None
    blockchain_impact: Optional[BlockchainImpact] = None
    
    # UI hints
    confirmation_required: bool = True
    severity: str = "normal"  # "low", "normal", "high", "critical"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Convert enum to string
        result["interrupt_type"] = result["interrupt_type"].value
        return result
    
    def to_frontend_format(self) -> str:
        """Convert to special frontend-parseable format in stream."""
        data = self.to_dict()
        return f"__INTERRUPT_START__{json.dumps(data)}__INTERRUPT_END__"


def create_standard_interrupt(
    tool_name: str,
    operation: str,
    parameters: Dict[str, Any],
    description: Optional[str] = None,
    fee_info: Optional[FeeInformation] = None,
    blockchain_impact: Optional[BlockchainImpact] = None,
    severity: str = "normal",
    custom_message: Optional[str] = None
) -> StandardInterruptMessage:
    """Create a standardized interrupt message."""
    
    from datetime import datetime
    
    interrupt_msg = StandardInterruptMessage(
        interrupt_id=str(uuid.uuid4()),
        interrupt_type=InterruptType.TOOL_CONFIRMATION,
        timestamp=datetime.now().isoformat(),
        tool_name=tool_name,
        operation=operation,
        description=description or f"Execute {tool_name}",
        parameters=parameters,
        message=custom_message or f"Please confirm execution of {operation}",
        fee_information=fee_info,
        blockchain_impact=blockchain_impact,
        severity=severity
    )
    
    return interrupt_msg


def send_standard_interrupt(interrupt_msg: StandardInterruptMessage) -> None:
    """Send standardized interrupt to LangGraph."""
    
    # Send the structured interrupt message
    interrupt(interrupt_msg.to_dict())


def create_transaction_interrupt(
    tool_name: str,
    operation: str,
    parameters: Dict[str, Any],
    fee_info: Optional[FeeInformation] = None,
    gas_estimate: Optional[str] = None,
    network: str = "Story Protocol"
) -> StandardInterruptMessage:
    """Create a transaction-specific interrupt."""
    
    blockchain_impact = BlockchainImpact(
        action=operation,
        network=network,
        estimated_gas=gas_estimate,
        transaction_type="blockchain_transaction",
        affects_balance=True
    )
    
    return create_standard_interrupt(
        tool_name=tool_name,
        operation=operation,
        parameters=parameters,
        description=f"Blockchain transaction: {operation}",
        fee_info=fee_info,
        blockchain_impact=blockchain_impact,
        severity="high",
        custom_message=f"Please approve this blockchain transaction: {operation}"
    )


# Convenience functions for common interrupt patterns
def create_simple_confirmation_interrupt(tool_name: str, parameters: Dict[str, Any]) -> StandardInterruptMessage:
    """Create a simple tool confirmation interrupt."""
    return create_standard_interrupt(
        tool_name=tool_name,
        operation=f"Execute {tool_name}",
        parameters=parameters,
        severity="normal"
    )


def create_fee_confirmation_interrupt(
    tool_name: str,
    operation: str,
    parameters: Dict[str, Any],
    fee_amount: str,
    fee_token: str,
    fee_display: str
) -> StandardInterruptMessage:
    """Create an interrupt with fee information."""
    
    fee_info = FeeInformation(
        fee_amount=fee_amount,
        fee_token=fee_token,
        fee_display=fee_display
    )
    
    return create_transaction_interrupt(
        tool_name=tool_name,
        operation=operation,
        parameters=parameters,
        fee_info=fee_info
    )