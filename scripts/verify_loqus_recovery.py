import json
import os
import sys
import logging

# Ensure loqus_core is in the path
sys.path.insert(0, os.getcwd())

from loqus_core.router import CascadeRouter
from loqus_core.reasoning.depth_calculator import determine_cognitive_depth
from loqus_core.whoiam.engine import WhoIAmEngine
from loqus_core.synapps.runtime import SynAppRuntime

def test_router():
    print("\n--- Testing CascadeRouter ---")
    router = CascadeRouter()
    
    # Test case 1: Low complexity
    model = router.route(0.2)
    print(f"Low complexity (0.2) -> {model}")
    assert model == "tiny-llama"
    
    # Test case 2: High complexity, good battery
    model = router.route(0.9, battery_level=1.0)
    print(f"High complexity (0.9), Battery 100% -> {model}")
    assert model == "mistral-large"
    
    # Test case 3: High complexity, low battery
    model = router.route(0.9, battery_level=0.1)
    print(f"High complexity (0.9), Battery 10% -> {model}")
    assert model == "qwen2-7b" or model == "tiny-llama"
    print("Router Test: PASS")

def test_depth_calculator():
    print("\n--- Testing DepthCalculator ---")
    # Simple query
    depth1 = determine_cognitive_depth("What time is it?")
    print(f"Simple query depth: {depth1}")
    assert depth1 in [1, 2, 3, 5]
    
    # Complex query
    depth2 = determine_cognitive_depth("Analyze the architectural trade-offs between Zero-Copy IPC and JSON-over-SHM for high-frequency trading systems.")
    print(f"Complex query depth: {depth2}")
    assert depth2 > depth1
    print("Depth Calculator Test: PASS")

def test_whoiam():
    print("\n--- Testing WhoIAmEngine ---")
    engine = WhoIAmEngine()
    prompt = engine.get_system_prompt("Solve this math problem.")
    print(f"Generated prompt (start): {prompt[:100]}...")
    assert "Jarvis" in prompt
    assert "battery" in prompt
    print("WhoIAm Test: PASS")

def test_synapps():
    print("\n--- Testing SynAppRuntime ---")
    runtime = SynAppRuntime(timeout=5)
    input_data = {
        "command": "sys_info",
        "args": {}
    }
    
    # Use path to sys_info.py
    script_path = "loqus_core/synapps/sys_info.py"
    result = runtime.execute_synapp(script_path, input_data)
    
    print(f"SynApp Result Success: {result.get('success')}")
    assert result.get('success') is True
    assert 'platform' in result.get('result', {})
    print("SynApps Test: PASS")

if __name__ == "__main__":
    try:
        test_router()
        test_depth_calculator()
        test_whoiam()
        test_synapps()
        print("\nALL RECOVERY TESTS PASSED! 🚀")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
