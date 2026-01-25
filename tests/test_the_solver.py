import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

from loqus_core.cognitive_council.the_solver import TheSolver


class TestTheSolver(unittest.TestCase):
    def setUp(self):
        self.solver = TheSolver(max_retries=3)

    def test_solver_initialization(self):
        """Test that TheSolver initializes correctly"""
        self.assertIsInstance(self.solver, TheSolver)
        self.assertEqual(self.solver.max_retries, 3)

    def test_query_validation_valid(self):
        """Test that valid queries pass validation"""
        valid_query = "Calculate 2 + 2"
        try:
            result = self.solver._validate_query(valid_query)
            self.assertEqual(result, "Calculate 2 + 2")
        except Exception as e:
            self.fail(f"_validate_query raised exception: {e}")

    def test_query_validation_empty(self):
        """Test that empty queries raise ValueError"""
        with self.assertRaises(ValueError):
            self.solver._validate_query("")

    def test_query_validation_dangerous_patterns(self):
        """Test that dangerous patterns are caught"""
        dangerous_queries = [
            "import os",
            "exec('print(1)')",
            "eval('1+1')",
            "__import__('os')",
            "open('/etc/passwd')",
            "sys.exit()",
        ]
        
        for query in dangerous_queries:
            with self.subTest(query=query):
                with self.assertRaises(ValueError):
                    self.solver._validate_query(query)

    def test_dsl_generation_math_expression(self):
        """Test DSL generation for math expressions"""
        query = "Solve 2 + 2"
        code = self.solver.generate_dsl(query)
        self.assertIn("result =", code)
        self.assertIn("2 + 2", code)

    def test_dsl_generation_non_math_expression(self):
        """Test DSL generation for non-math expressions"""
        query = "What is your name?"
        code = self.solver.generate_dsl(query)
        self.assertIn("Error: Not a mathematical query", code)

    def test_execute_in_sandbox_valid(self):
        """Test sandbox execution with valid code"""
        code = "result = 2 + 2"
        result = self.solver.execute_in_sandbox(code)
        self.assertTrue(result['success'])
        self.assertEqual(result['result'], 4)

    def test_execute_in_sandbox_invalid(self):
        """Test sandbox execution with invalid code"""
        code = "result = 2 / 0"
        result = self.solver.execute_in_sandbox(code)
        self.assertTrue(result['success'])
        self.assertIn("Error: Division by zero", result['result'])

    def test_solve_math_expression(self):
        """Test solving a math expression"""
        query = "Calculate 5 * 6"
        result = self.solver.solve(query)
        self.assertTrue(result['success'])
        self.assertEqual(result['result'], 30)

    def test_solve_invalid_expression(self):
        """Test solving an invalid expression"""
        query = "Calculate 1 / 0"
        result = self.solver.solve(query)
        self.assertTrue(result['success'])
        self.assertIn("Error: Division by zero", result['result'])

    def test_solve_non_math_expression(self):
        """Test solving a non-math expression"""
        query = "What is the weather like?"
        result = self.solver.solve(query)
        self.assertTrue(result['success'])
        self.assertIn("Error: Not a mathematical query", result['result'])

    def test_solve_with_dangerous_input(self):
        """Test solving with dangerous input"""
        query = "import os; os.system('ls')"
        result = self.solver.solve(query)
        self.assertFalse(result['success'])
        self.assertIn("ValueError", result['error'])

    def test_get_capabilities(self):
        """Test that capabilities are returned correctly"""
        capabilities = self.solver.get_capabilities()
        self.assertIn('name', capabilities)
        self.assertEqual(capabilities['name'], 'TheSolver')
        self.assertIn('description', capabilities)
        self.assertIn('max_retries', capabilities)
        self.assertIn('supported_operations', capabilities)
        self.assertIn('security_level', capabilities)

    @patch('loqus_core.cognitive_council.the_solver.logger')
    def test_logging_on_error(self, mock_logger):
        """Test that errors are logged properly"""
        # This test ensures that logging is called when errors occur
        query = "import os"
        result = self.solver.solve(query)
        # The logger should have been called at least once
        self.assertTrue(mock_logger.warning.called or mock_logger.error.called)


if __name__ == '__main__':
    unittest.main()
