import sys
sys.path.insert(0, '.');
from loqus_core.cognitive_council.the_solver import TheSolver

def test_solver():
    solver = TheSolver()
    
    # Test basic math expression
    result = solver.solve('solve 2 + 2')
    print(f'Result for "solve 2 + 2": {result}')
    
    # Test more complex expression
    result = solver.solve('calculate 10 * 5')
    print(f'Result for "calculate 10 * 5": {result}')
    
    print('All tests passed!')

test_solver()