#!/usr/bin/env python3
"""Integration test for hillium_backend PyO3 bindings."""

import sys
import os

# Add the debug build to the path
debug_path = os.path.join('target', 'debug')
if os.path.exists(debug_path):
    sys.path.insert(0, debug_path)

try:
    import hillium_backend
    print("✓ Successfully imported hillium_backend")
    
    # Test HippoLink creation
    link = hillium_backend.HippoLink('test_memory')
    print("✓ Successfully created HippoLink instance")
    
    # Test Level 1 (Sensory) methods
    link.write_conversation('conv1', 'Hello World')
    content = link.read_conversation('conv1')
    assert content == 'Hello World', f"Expected 'Hello World', got '{content}'"
    print("✓ Level 1: read_conversation/write_conversation working")
    
    link.set_intent('conv1', 'greeting')
    intent = link.get_intent('conv1')
    assert intent == 'greeting', f"Expected 'greeting', got '{intent}'"
    print("✓ Level 1: get_intent/set_intent working")
    
    # Test Level 2 (Working Memory) methods
    link.store_note('note1', 'Test note content')
    note = link.get_note('note1')
    assert note == 'Test note content', f"Expected 'Test note content', got '{note}'"
    print("✓ Level 2: store_note/get_note working")
    
    # Test Level 2.5 (Associative) methods
    updates = link.get_associative_updates()
    assert isinstance(updates, list), f"Expected list, got {type(updates)}"
    print(f"✓ Level 2.5: get_associative_updates returned {len(updates)} updates")
    
    needs_consolidation = link.needs_consolidation()
    assert isinstance(needs_consolidation, bool), f"Expected bool, got {type(needs_consolidation)}"
    print(f"✓ Level 2.5: needs_consolidation returned {needs_consolidation}")
    
    # Test utility methods
    telemetry = link.get_telemetry()
    assert isinstance(telemetry, str), f"Expected str, got {type(telemetry)}"
    print(f"✓ Utility: get_telemetry returned: {telemetry}")
    
    link.heartbeat()
    print("✓ Utility: heartbeat executed successfully")
    
    estopped = link.is_estopped()
    assert isinstance(estopped, bool), f"Expected bool, got {type(estopped)}"
    print(f"✓ Utility: is_estopped returned {estopped}")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
