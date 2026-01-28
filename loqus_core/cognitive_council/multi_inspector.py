#!/usr/bin/env python3
"""
Multi-Inspector for the Cognitive Council

This module implements a multi-inspector system that runs multiple inspectors
in parallel and uses voting to determine the final result.
"""

import asyncio
import logging
from typing import List, Any
from enum import Enum
from dataclasses import dataclass
import time

# Configure logging
logger = logging.getLogger(__name__)


class ConsensusResult(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_HUMAN = "requires_human"


class ActionPlan:
    """Data class for ActionPlan - represents an action to be inspected"""
    def __init__(self, content: str = ""):
        self.content = content


class Inspector:
    """Base inspector class"""
    def __init__(self, name: str):
        self.name = name

    async def inspect(self, action_plan: ActionPlan) -> str:
        # This is a placeholder - in real implementation this would be
        # replaced with actual inspection logic
        await asyncio.sleep(0.01)  # Simulate async work
        
        # Mock logic based on content
        if "unsafe" in action_plan.content.lower() or "dangerous" in action_plan.content.lower() or "forbidden" in action_plan.content.lower():
            return "rejected"
        elif "safe" in action_plan.content.lower() or "permitted" in action_plan.content.lower() or "harmless" in action_plan.content.lower():
            return "approved"
        else:
            return "requires_human"


async def multi_inspector(plan: ActionPlan, timeout: float = 5.0) -> ConsensusResult:
    """
    Run multiple inspectors in parallel and determine consensus result.
    
    Args:
        plan: The action plan to inspect
        timeout: Maximum time to wait for inspectors (default 5 seconds)
        
    Returns:
        ConsensusResult: The final consensus result
    """
    # Create inspectors
    inspectors = [
        Inspector("Inspector_1"),
        Inspector("Inspector_2"),
        Inspector("Inspector_3")
    ]

    try:
        # Run inspectors in parallel with timeout
        tasks = [inspector.inspect(plan) for inspector in inspectors]
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
        
        # Log all results for analysis
        logger.info(f"Inspector results: {results}")
        
        # Voting logic
        if all(result == "approved" for result in results):
            return ConsensusResult.APPROVED
        elif any(result == "rejected" for result in results):
            return ConsensusResult.REJECTED
        else:
            return ConsensusResult.REQUIRES_HUMAN
            
    except asyncio.TimeoutError:
        logger.warning("Timeout occurred while waiting for inspectors")
        # If timeout occurs, we default to requiring human review
        return ConsensusResult.REQUIRES_HUMAN
    except Exception as e:
        logger.error(f"Error during inspection: {e}")
        # If there's an error, we default to requiring human review
        return ConsensusResult.REQUIRES_HUMAN


# For backward compatibility with existing code
# These are the original mock functions from the old implementation

def mock_inspector_1(action_plan: ActionPlan) -> str:
    """Mock inspector 1 - simulates an inspector that evaluates action plans"""
    # Simulate some processing time
    time.sleep(0.01)
    
    # Return a mock result based on content
    if "unsafe" in action_plan.content.lower():
        return "rejected"
    elif "safe" in action_plan.content.lower() or "permitted" in action_plan.content.lower():
        return "approved"
    else:
        return "requires_human"


def mock_inspector_2(action_plan: ActionPlan) -> str:
    """Mock inspector 2 - simulates another inspector"""
    # Simulate some processing time
    time.sleep(0.01)
    
    # Return a mock result based on content
    if "dangerous" in action_plan.content.lower():
        return "rejected"
    elif "harmless" in action_plan.content.lower() or "permitted" in action_plan.content.lower():
        return "approved"
    else:
        return "requires_human"


def mock_inspector_3(action_plan: ActionPlan) -> str:
    """Mock inspector 3 - simulates a third inspector"""
    # Simulate some processing time
    time.sleep(0.01)
    
    # Return a mock result based on content
    if "forbidden" in action_plan.content.lower():
        return "rejected"
    elif "permitted" in action_plan.content.lower() or "safe" in action_plan.content.lower():
        return "approved"
    else:
        return "requires_human"
