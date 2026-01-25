import unittest
from unittest.mock import patch, MagicMock
from loqus_core.cognitive_council.the_solver import TheSolver


class TestTheSolver(unittest.TestCase):
    def setUp(self):
        self.solver = TheSolver()

    def test_dsl_generation_basic_math(self):
        # Test basic arithmetic DSL generation
        problem = "What is 2 + 2?"
        dsl = self.solver.generate_dsl(problem)
        self.assertIn("2 + 2", dsl)
        # The DSL should contain result assignment
        self.assertIn("result =", dsl)

    def test_dsl_generation_quadratic(self):
        # Test quadratic equation DSL generation
        problem = "Solve x^2 + 5x + 6 = 0"
        dsl = self.solver.generate_dsl(problem)
        self.assertIn("x**2", dsl)
        self.assertIn("5*x", dsl)
        self.assertIn("6", dsl)

    @patch('restrictedpython.compile_restricted')
    def test_sandboxed_execution_security(self, mock_compile):
        # Test that sandbox prevents file I/O
        mock_compile.return_value = "exec_result"
        
        # This should not allow file operations
        with self.assertRaises(Exception) as context:
            self.solver.execute_in_sandbox("open('test.txt', 'r')")
        
        # Verify that the sandbox prevents dangerous operations
        self.assertIn("RestrictedPython", str(context.exception))

    def test_retry_mechanism_syntax_error(self):
        # Test that retry works for syntax errors
        with patch.object(self.solver, 'execute_in_sandbox') as mock_execute:
            # First attempt fails with syntax error
            mock_execute.side_effect = [SyntaxError("invalid syntax"), "result"]
            
            # Should retry once and succeed
            result = self.solver.solve("test problem")
            self.assertEqual(result, "result")

    def test_retry_limit_exceeded(self):
        # Test that retry limit is enforced
        with patch.object(self.solver, 'execute_in_sandbox') as mock_execute:
            mock_execute.side_effect = SyntaxError("invalid syntax")
            
            # Should raise exception after 3 retries
            with self.assertRaises(SyntaxError):
                self.solver.solve("test problem")

    def test_logic_precision(self):
        # Test that calculations are precise
        problem = "What is 10 * 10?"
        dsl = self.solver.generate_dsl(problem)
        result = self.solver.execute_in_sandbox(dsl)
        self.assertEqual(result, 100)


if __name__ == '__main__':
    unittest.main()