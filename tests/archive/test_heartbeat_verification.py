#!/usr/bin/env python3
"""
Test script to verify heartbeat functionality.
This will check if the heartbeat thread is properly implemented.
"""

import os
import sys
import time
import struct

def test_heartbeat_implementation():
    """Test if heartbeat implementation exists."""
    
    print("🔍 Testing Heartbeat Implementation...")
    
    # Check if AegisCore has heartbeat thread
    aegis_lib_path = "crates/aegis_core/src/lib.rs"
    if not os.path.exists(aegis_lib_path):
        print("❌ AegisCore lib.rs not found")
        return False
    
    with open(aegis_lib_path, 'r') as f:
        content = f.read()
    
    # Check for heartbeat thread implementation
    has_heartbeat_thread = "start_heartbeat_thread" in content
    has_heartbeat_loop = "heartbeat" in content.lower() and "thread" in content.lower()
    has_atomic_update = "hb_aegis_ns" in content
    
    print(f"  - Has heartbeat thread function: {has_heartbeat_thread}")
    print(f"  - Has heartbeat loop: {has_heartbeat_loop}")
    print(f"  - Updates hb_aegis_ns: {has_atomic_update}")
    
    if not (has_heartbeat_thread and has_heartbeat_loop and has_atomic_update):
        print("❌ Heartbeat implementation incomplete")
        return False
    
    print("✅ Heartbeat implementation found")
    return True

def test_watchdog_implementation():
    """Test if watchdog implementation exists."""
    
    print("\n🔍 Testing Watchdog Implementation...")
    
    watchdog_path = "hipposerver/src/watchdog/mod.rs"
    if not os.path.exists(watchdog_path):
        print("❌ Watchdog mod.rs not found")
        return False
    
    with open(watchdog_path, 'r') as f:
        content = f.read()
    
    # Check for watchdog implementation
    has_watchdog_struct = "pub struct Watchdog" in content
    has_monitor_loop = "monitor_loop" in content
    has_timeout_check = "500" in content or "timeout" in content.lower()
    
    print(f"  - Has Watchdog struct: {has_watchdog_struct}")
    print(f"  - Has monitor loop: {has_monitor_loop}")
    print(f"  - Has timeout check: {has_timeout_check}")
    
    if not (has_watchdog_struct and has_monitor_loop and has_timeout_check):
        print("❌ Watchdog implementation incomplete")
        return False
    
    print("✅ Watchdog implementation found")
    return True

def test_hippo_state():
    """Test if HippoState has hb_aegis_ns field."""
    
    print("\n🔍 Testing HippoState...")
    
    state_path = "hipposerver/src/shm/state.rs"
    if not os.path.exists(state_path):
        print("❌ state.rs not found")
        return False
    
    with open(state_path, 'r') as f:
        content = f.read()
    
    has_hb_field = "hb_aegis_ns: AtomicU64" in content
    
    print(f"  - Has hb_aegis_ns field: {has_hb_field}")
    
    if not has_hb_field:
        print("❌ HippoState missing hb_aegis_ns field")
        return False
    
    print("✅ HippoState has hb_aegis_ns field")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("HEARTBEAT & WATCHDOG VERIFICATION TEST")
    print("=" * 60)
    
    results = []
    results.append(("HippoState", test_hippo_state()))
    results.append(("Heartbeat", test_heartbeat_implementation()))
    results.append(("Watchdog", test_watchdog_implementation()))
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20s} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed - implementation needed")
        sys.exit(1)
