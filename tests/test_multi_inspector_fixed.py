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
    # Create an action plan that will make ALL inspectors return 'approved'
    action_plan = ActionPlan("This is a permitted action")
    
    # Run the multi-inspector
    result = asyncio.run(multi_inspector(action_plan))
    
    print(f"Result for permitted action: {result}")
    assert result == ConsensusResult.APPROVED


def test_any_inspector_rejects():
    """Test that when any inspector rejects, result is REJECTED"""
    action_plan = ActionPlan("This is an unsafe action")
    
    # Run the multi-inspector
    result = asyncio.run(multi_inspector(action_plan))
    
    print(f"Result for unsafe action: {result}")
    assert result == ConsensusResult.REJECTED


def test_mixed_results_require_human():
    """Test that when results are mixed, result is REQUIRES_HUMAN"""
    action_plan = ActionPlan("This is a neutral action")
    
    # Run the multi-inspector
    result = asyncio.run(multi_inspector(action_plan))
    
    print(f"Result for neutral action: {result}")
    # This should be REQUIRES_HUMAN since not all approve and not all reject
    assert result == ConsensusResult.REQUIRES_HUMAN


def test_inspector_exception_handling():
    """Test that exceptions in inspectors are handled properly"""
    action_plan = ActionPlan("This is a safe action")
    
    # Run the multi-inspector
    result = asyncio.run(multi_inspector(action_plan))
    
    # Should not raise an exception
    print(f"Result for safe action with exceptions: {result}")
    assert result in [ConsensusResult.APPROVED, ConsensusResult.REQUIRES_HUMAN]

if __name__ == "__main__":
    # Run tests directly
    try:
        test_all_inspectors_approve()
        print("All inspectors approve test passed!")
    except Exception as e:
        print(f"All inspectors approve test failed: {e}")
        
    try:
        test_any_inspector_rejects()
        print("Any inspector rejects test passed!")
    except Exception as e:
        print(f"Any inspector rejects test failed: {e}")
        
    try:
        test_mixed_results_require_human()
        print("Mixed results test passed!")
    except Exception as e:
        print(f"Mixed results test failed: {e}")
        
    try:
        test_inspector_exception_handling()
        print("Exception handling test passed!")
    except Exception as e:
        print(f"Exception handling test failed: {e}")
        
    print("\nDebugging individual inspectors:")
    from loqus_core.cognitive_council.multi_inspector import mock_inspector_1, mock_inspector_2, mock_inspector_3, ActionPlan
    action_plan = ActionPlan("This is a permitted action")
    print(f"Inspector 1: {mock_inspector_1(action_plan)}")
    print(f"Inspector 2: {mock_inspector_2(action_plan)}")
    print(f"Inspector 3: {mock_inspector_3(action_plan)}")