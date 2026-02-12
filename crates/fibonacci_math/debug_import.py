# Debug script to understand import issues

import sys
import os

print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Add the current directory to Python path
sys.path.insert(0, '/workspace/crates/fibonacci_math')
sys.path.insert(0, '/workspace/crates/fibonacci_math/python')

print(f"Updated Python path: {sys.path}")

# Try to list what's in the fibonacci_math directory
try:
    print("\nContents of /workspace/crates/fibonacci_math:")
    for item in os.listdir('/workspace/crates/fibonacci_math'):
        print(f"  {item}")
        
    print("\nContents of /workspace/crates/fibonacci_math/python:")
    for item in os.listdir('/workspace/crates/fibonacci_math/python'):
        print(f"  {item}")
        
except Exception as e:
    print(f"Error listing directories: {e}")

# Try to import
try:
    import fibonacci_math
    print("\n✅ Successfully imported fibonacci_math")
    print(f"Module file: {fibonacci_math.__file__}")
    print(f"Module attributes: {dir(fibonacci_math)}")
except ImportError as e:
    print(f"\n❌ Failed to import fibonacci_math: {e}")
    
    # Try to import the internal module
    try:
        from ._fibonacci_math import *
        print("✅ Successfully imported internal module")
    except Exception as e2:
        print(f"❌ Failed to import internal module: {e2}")
