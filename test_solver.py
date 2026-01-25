import sys
sys.path.insert(0, '/Users/jsaldana/GitLocalRepo/hillium-core')

from loqus_core.cognitive_council.the_solver import TheSolver

ts = TheSolver()
print('Test 1 - Simple math:', ts.solve('solve 2 + 2'))
print('Test 2 - Complex math:', ts.solve('solve 3 * (4 + 5)'))
print('Test 3 - Division:', ts.solve('solve 10 / 2'))