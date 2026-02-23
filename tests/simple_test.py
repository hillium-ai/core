import sys
import os

# Add the current directory to Python path
sys.path.insert(0, '.')

try:
    # Try to import the module
    import fibonacci_math
    print('SUCCESS: fibonacci_math imported')
    
    # Check what's available
    print('Available attributes:', dir(fibonacci_math))
    
except Exception as e:
    print(f'Error importing fibonacci_math: {e}')
    import traceback
    traceback.print_exc()
