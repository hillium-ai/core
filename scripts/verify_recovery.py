import sys
import os
import time

# Add current dir to path to find the linked .so
sys.path.append(os.path.join(os.getcwd(), "loqus_core"))

try:
    import hillium_backend
    from hillium_backend import HippoLink
    print("✅ Python successfully imported hillium_backend")
except ImportError as e:
    print(f"❌ Python failed to import hillium_backend: {e}")
    # List files in loqus_core for debug
    print(f"Contents of loqus_core: {os.listdir('loqus_core')}")
    sys.exit(1)

# Try to connect to HippoServer
try:
    hippo = HippoLink()
    print("✅ HippoLink initialized and connected to SHM")
    
    # Write something
    hippo.write_conversation("Test message from Python recovery script")
    print("✅ Wrote to conversation buffer")
    
    # Read back
    text = hippo.read_conversation()
    print(f"✅ Read back: {text}")
    assert text == "Test message from Python recovery script"
    
    # Check intent
    intent = hippo.get_intent()
    print(f"✅ Initial intent: {intent}")
    
    hippo.set_intent("Listening")
    intent = hippo.get_intent()
    print(f"✅ Updated intent: {intent}")
    assert "Listening" in intent
    
    print("\n🎉 RECOVERY VERIFIED: Python bridge to HippoServer is working.")
except Exception as e:
    print(f"❌ Failed to communicate with HippoServer: {e}")
    sys.exit(1)
