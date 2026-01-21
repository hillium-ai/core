import pytest
import tempfile
import os
from loqus_core.memory.beliefs import BeliefRecord, RecordType, TimeReference, BeliefStore, SessionSummary


def test_belief_creation():
    belief = BeliefRecord(
        belief_type="observation",
        content="User prefers morning activities",
        confidence=0.8
    )
    assert belief.id is not None
    assert belief.confidence == 0.8


def test_belief_validation_confidence():
    with pytest.raises(ValueError):
        BeliefRecord(
            belief_type="test",
            content="test",
            confidence=1.5  # Invalid
        )


def test_belief_serialization():
    belief = BeliefRecord(
        belief_type="observation",
        content="Test content",
        record_type=RecordType.Observation
    )
    data = belief.to_dict()
    restored = BeliefRecord.from_dict(data)
    assert restored.content == belief.content


def test_session_summary_creation():
    summary = SessionSummary(summary="Line 1\nLine 2\nLine 3")
    assert summary.session_id is not None
    assert summary.summary == "Line 1\nLine 2\nLine 3"


def test_session_summary_validation():
    with pytest.raises(ValueError):
        SessionSummary(summary=123)  # Invalid - not a string


def test_belief_store_crud_operations():
    # Use a temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        # Initialize store
        store = BeliefStore(db_path=temp_path)
        
        # Create a belief
        belief = BeliefRecord(
            belief_type="observation",
            content="Test observation",
            confidence=0.9
        )
        
        # Test create
        belief_id = store.create_belief(belief)
        assert belief_id == belief.id
        
        # Test read
        retrieved = store.get_belief(belief_id)
        assert retrieved is not None
        assert retrieved.content == "Test observation"
        
        # Test list
        all_beliefs = store.list_beliefs()
        assert len(all_beliefs) == 1
        assert all_beliefs[0].content == "Test observation"
        
        # Test delete
        deleted = store.delete_belief(belief_id)
        assert deleted is True
        
        # Verify deletion
        retrieved_after_delete = store.get_belief(belief_id)
        assert retrieved_after_delete is None
        
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_belief_store_file_locking():
    # Use a temporary file for testing
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name
    
    try:
        store = BeliefStore(db_path=temp_path)
        
        # Create a belief
        belief = BeliefRecord(
            belief_type="observation",
            content="Test observation for locking",
        )
        
        # Test that we can create and read without issues
        belief_id = store.create_belief(belief)
        retrieved = store.get_belief(belief_id)
        assert retrieved is not None
        
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)