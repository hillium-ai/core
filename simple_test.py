import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

# Try to import and test
try:
    from loqus_core.cognitive_council.the_solver import TheSolver
    print('Import successful')
    ts = TheSolver()
    print('Instance created successfully')
    result = ts.solve('solve 2 + 2')
    print('Test result:', result)
except Exception as e:
    print('Error:', str(e))
    import traceback
    traceback.print_exc()