#!/usr/bin/env python3
"""Integration test for hillium_backend PyO3 bindings.

NOTE: This test requires the hillium_backend library to be built first.
Run: maturin develop --release
Or:  cargo build --release -p hillium_backend

This test is SKIPPED if hillium_backend is not available.
"""

import sys
import os
import pytest

# Add the debug build to the path
debug_path = os.path.join('target', 'debug')
if os.path.exists(debug_path):
    sys.path.insert(0, debug_path)


def _check_hillium_backend():
    """Check if hillium_backend is available and has HippoLink."""
    try:
        import hillium_backend
        if not hasattr(hillium_backend, 'HippoLink'):
            return False, "hillium_backend has no HippoLink attribute"
        return True, hillium_backend
    except ImportError as e:
        return False, str(e)


# Module-level check (safe - doesn't crash)
_available, _backend_or_error = _check_hillium_backend()

# Skip all tests if backend not available
pytestmark = pytest.mark.skipif(
    not _available,
    reason=f"hillium_backend not available: {_backend_or_error if not _available else 'N/A'}"
)


@pytest.fixture
def hillium_backend():
    """Provide the hillium_backend module."""
    if not _available:
        pytest.skip(f"hillium_backend not available: {_backend_or_error}")
    return _backend_or_error


@pytest.fixture
def hippo_link(hillium_backend):
    """Create a HippoLink instance for testing."""
    return hillium_backend.HippoLink('test_memory')


class TestHippoLinkLevel1:
    """Test Level 1 (Sensory) methods."""
    
    def test_conversation_read_write(self, hippo_link):
        """Test read/write conversation."""
        hippo_link.write_conversation('conv1', 'Hello World')
        content = hippo_link.read_conversation('conv1')
        assert content == 'Hello World'
    
    def test_intent_get_set(self, hippo_link):
        """Test get/set intent."""
        hippo_link.set_intent('conv1', 'greeting')
        intent = hippo_link.get_intent('conv1')
        assert intent == 'greeting'


class TestHippoLinkLevel2:
    """Test Level 2 (Working Memory) methods."""
    
    def test_note_store_get(self, hippo_link):
        """Test store/get note."""
        hippo_link.store_note('note1', 'Test note content')
        note = hippo_link.get_note('note1')
        assert note == 'Test note content'


class TestHippoLinkLevel25:
    """Test Level 2.5 (Associative) methods."""
    
    def test_associative_updates(self, hippo_link):
        """Test get_associative_updates returns a list."""
        updates = hippo_link.get_associative_updates()
        assert isinstance(updates, list)
    
    def test_needs_consolidation(self, hippo_link):
        """Test needs_consolidation returns a bool."""
        needs = hippo_link.needs_consolidation()
        assert isinstance(needs, bool)


class TestHippoLinkUtility:
    """Test utility methods."""
    
    def test_telemetry(self, hippo_link):
        """Test get_telemetry returns a string."""
        telemetry = hippo_link.get_telemetry()
        assert isinstance(telemetry, str)
    
    def test_heartbeat(self, hippo_link):
        """Test heartbeat executes without error."""
        hippo_link.heartbeat()  # Should not raise
    
    def test_is_estopped(self, hippo_link):
        """Test is_estopped returns a bool."""
        estopped = hippo_link.is_estopped()
        assert isinstance(estopped, bool)


if __name__ == '__main__':
    # For manual execution outside pytest
    if not _available:
        print(f"❌ hillium_backend not available: {_backend_or_error}")
        sys.exit(1)
    
    print("✓ Successfully imported hillium_backend")
    
    try:
        link = _backend_or_error.HippoLink('test_memory')
        print("✓ Successfully created HippoLink instance")
        
        # Quick smoke test
        link.write_conversation('conv1', 'Hello World')
        content = link.read_conversation('conv1')
        assert content == 'Hello World'
        print("✓ Level 1: read_conversation/write_conversation working")
        
        link.heartbeat()
        print("✓ Utility: heartbeat executed successfully")
        
        print("\n✅ All smoke tests passed!")
        print("Run 'pytest test_hillium_backend.py -v' for full test suite")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
