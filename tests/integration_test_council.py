import unittest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

from loqus_core.cognitive_council.the_solver import TheSolver
from loqus_core.council.core import CognitiveCouncil


class TestCouncilIntegration(unittest.TestCase):
    def test_solver_in_council_roles(self):
        """Test that TheSolver is properly integrated into the council"""
        # Import the council structure
        from loqus_core.cognitive_council import COUNCIL
        
        # Check that TheSolver is in the council
        self.assertIn('TheSolver', COUNCIL['roles'])
        
        # Check that we can access the solver
        self.assertIsInstance(COUNCIL['solver'], TheSolver)

    def test_solver_can_solve_math(self):
        """Test that TheSolver can solve math problems"""
        from loqus_core.cognitive_council import COUNCIL
        
        solver = COUNCIL['solver']
        
        # Test basic math
        result = solver.solve("Calculate 2 + 2")
        self.assertTrue(result['success'])
        self.assertEqual(result['result'], 4)
        
        # Test more complex math
        result = solver.solve("Solve 10 * 5")
        self.assertTrue(result['success'])
        self.assertEqual(result['result'], 50)

    def test_solver_security(self):
        """Test that TheSolver security measures work"""
        from loqus_core.cognitive_council import COUNCIL
        
        solver = COUNCIL['solver']
        
        # Test that dangerous code is blocked
        result = solver.solve("import os")
        self.assertFalse(result['success'])
        
        # Test that dangerous eval is blocked
        result = solver.solve("eval('1+1')")
        self.assertFalse(result['success'])


if __name__ == '__main__':
    unittest.main()
