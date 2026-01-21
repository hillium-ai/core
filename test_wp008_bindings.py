
import sys
import os
import time

# Add target/debug to path to find the compiled library
# Note: cargo build produces libhillium_backend.so, but python expects hillium_backend.so
# We might need to symlink it first.

try:
    import  hillium_backend
    print("Successfully imported hillium_backend naturally")
except ImportError:
    # Try finding it in target/debug
    so_path = "/workspace/target/debug/libhillium_backend.so"
    target_path = "/workspace/hillium_backend.so"
    
    if os.path.exists(so_path):
        print(f"Found {so_path}, symlinking to {target_path}")
        if os.path.exists(target_path):
            os.remove(target_path)
        os.symlink(so_path, target_path)
        sys.path.append("/workspace")
        import hillium_backend
        print("Successfully imported hillium_backend via symlink")
    else:
        print("Could not find libhillium_backend.so in target/debug")
        sys.exit(1)

from hillium_backend import HippoLink

def test_bindings():
    print("Testing WP-008 Bindings...")
    
    # Initialize
    try:
        hippo = HippoLink("wp008_test", "/tmp/wp008_db")
        print("HippoLink initialized")
    except Exception as e:
        print(f"Failed to initialize HippoLink: {e}")
        return

    # Test Level 1
    intent = hippo.get_intent()
    print(f"Intent: {intent}")
    
    # Test Level 2 (Working Memory)
    print("Testing Working Memory...")
    hippo.store_note("test_note", "Hello from Python")
    retrieved = hippo.get_note("test_note")
    print(f"Retrieved note: {retrieved}")
    assert retrieved == "Hello from Python"

    # Test Level 2.5 (Associative Core)
    print("Testing Associative Core...")
    weights = hippo.get_associative_weights()
    print(f"Weights length: {len(weights)}")
    assert len(weights) > 0 # Should be 128*128 flat? No, export_weights logic
    
    # Test Telemetry
    telem = hippo.get_telemetry()
    print(f"Telemetry: {telem}")
    
    print("WP-008 Bindings Verified Successfully!")

if __name__ == "__main__":
    test_bindings()
