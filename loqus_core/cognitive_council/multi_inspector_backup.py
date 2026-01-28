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

# Configure logging
logger = logging.getLogger(__name__)

class ConsensusResult(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    REQUIRES_HUMAN = "requires_human"

class ActionPlan:
    """Placeholder for ActionPlan class - this would be defined elsewhere in the system"""
    def __init__(self, content: str = ""):
        self.content = content


def mock_inspector_1(action_plan: ActionPlan) -> str:
    """Mock inspector 1 - simulates an inspector that evaluates action plans"""
    # Simulate some processing time
    import time
    time.sleep(0.01)
    
    # Return a mock result based on content
    if "unsafe" in action_plan.content.lower():
        return "rejected"
    elif "safe" in action_plan.content.lower():
        return "approved"
    else:
        return "requires_human"


def mock_inspector_2(action_plan: ActionPlan) -> str:
    """Mock inspector 2 - simulates another inspector"""
    # Simulate some processing time
    import time
    time.sleep(0.01)
    
    # Return a mock result based on content
    if "dangerous" in action_plan.content.lower():
        return "rejected"
    elif "harmless" in action_plan.content.lower():
        return "approved"
    else:
        return "requires_human"


def mock_inspector_3(action_plan: ActionPlan) -> str:
    """Mock inspector 3 - simulates a third inspector"""
    # Simulate some processing time
    import time
    time.sleep(0.01)
    
    # Return a mock result based on content
    if "forbidden" in action_plan.content.lower():
        return "rejected"
    elif "permitted" in action_plan.content.lower():
        return "approved"
    else:
        return "requires_human"


async def run_inspector(inspector_func, action_plan: ActionPlan) -> str:
    """Run a single inspector asynchronously"""
    try:
        result = inspector_func(action_plan)
        return result
    except Exception as e:
        logger.error(f"Inspector failed with error: {e}")
        return "requires_human"

async def multi_inspector(action_plan: ActionPlan) -> ConsensusResult:
    """Run multiple inspectors in parallel and determine consensus result"""
    # Run all inspectors in parallel
    inspectors = [mock_inspector_1, mock_inspector_2, mock_inspector_3]
    
    # Execute all inspectors concurrently
    tasks = [run_inspector(inspector, action_plan) for inspector in inspectors]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Count results
    approved_count = 0
    rejected_count = 0
    requires_human_count = 0
    
    for result in results:
        if isinstance(result, Exception):
            # If an exception occurred, treat it as requiring human review
            requires_human_count += 1
        else:
            if result == "approved":
                approved_count += 1
            elif result == "rejected":
                rejected_count += 1
            elif result == "requires_human":
                requires_human_count += 1

    # Determine consensus
    if rejected_count > 0:
        # If any inspector rejects, reject the action
        logger.info(f"Action rejected by inspectors. Results: approved={approved_count}, rejected={rejected_count}, requires_human={requires_human_count}")
        return ConsensusResult.REJECTED
    elif approved_count >= 2:
        # If 2 or more inspectors approve, approve the action
        logger.info(f"Action approved by inspectors. Results: approved={approved_count}, rejected={rejected_count}, requires_human={requires_human_count}")
        return ConsensusResult.APPROVED
    else:
        # If we don't have enough approvals and no rejections, require human review
        logger.info(f"Action requires human review. Results: approved={approved_count}, rejected={rejected_count}, requires_human={requires_human_count}")
        return ConsensusResult.REQUIRES_HUMAN


# Example usage
if __name__ == "__main__":
    # Test with different action plans
    test_plan = ActionPlan("This is a safe action")
    
    # Run the multi-inspector
    result = asyncio.run(multi_inspector(test_plan))
    print(f"Consensus result: {result.value}")
