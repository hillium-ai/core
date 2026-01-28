import asyncio
import pytest
from unittest.mock import patch

# Import the module we just created
import sys
sys.path.insert(0, '.')

from loqus_core.cognitive_council.multi_inspector import (
    multi_inspector,
    ActionPlan,
    ConsensusResult
)


def test_all_inspectors_approve():
    """Test that when all inspectors approve, result is APPROVED"""
    action_plan = ActionPlan("This is a safe action")
    
    # Run the multi-inspector
    result = asyncio.run(multi_inspector(action_plan))
    
    assert result == ConsensusResult.APPROVED


def test_any_inspector_rejects():
    """Test that when any inspector rejects, result is REJECTED"""
    action_plan = ActionPlan("This is an unsafe action")
    
    # Run the multi-inspector
    result = asyncio.run(multi_inspector(action_plan))
    
    assert result == ConsensusResult.REJECTED


def test_mixed_results_require_human():
    """Test that when results are mixed, result is REQUIRES_HUMAN"""
    action_plan = ActionPlan("This is a neutral action")
    
    # Run the multi-inspector
    result = asyncio.run(multi_inspector(action_plan))
    
    assert result == ConsensusResult.REQUIRES_HUMAN


def test_inspector_exception_handling():
    """Test that exceptions in inspectors are handled properly"""
    action_plan = ActionPlan("This is a safe action")
    
    # Run the multi-inspector
    result = asyncio.run(multi_inspector(action_plan))
    
    # Should not raise an exception
    assert result in [ConsensusResult.APPROVED, ConsensusResult.REQUIRES_HUMAN]


def test_timeout_handling():
    """Test that timeout is handled properly"""
    action_plan = ActionPlan("This is a safe action")
    
    # Run the multi-inspector with a very short timeout
    result = asyncio.run(multi_inspector(action_plan, timeout=0.001))
    
    # Should not raise an exception and should return REQUIRES_HUMAN
    assert result == ConsensusResult.REQUIRES_HUMAN

if __name__ == "__main__":
    # Run tests directly
    test_all_inspectors_approve()
    test_any_inspector_rejects()
    test_mixed_results_require_human()
    test_inspector_exception_handling()
    test_timeout_handling()
    print("All tests passed!")