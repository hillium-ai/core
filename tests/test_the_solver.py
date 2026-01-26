import pytest
from loqus_core.cognitive_council.the_solver import TheSolver


def test_solver_exists():
    """Test that TheSolver class can be imported and instantiated"""
    solver = TheSolver()
    assert solver is not None
    assert hasattr(solver, 'solve')
    assert hasattr(solver, 'generate_dsl')
    assert hasattr(solver, 'execute_in_sandbox')


def test_solver_solve_math_problem():
    """Test that TheSolver can solve basic math problems"""
    solver = TheSolver()
    result = solver.solve("2 + 2")
    assert result == 4


def test_solver_solve_complex_problem():
    """Test that TheSolver can solve more complex math problems"""
    solver = TheSolver()
    result = solver.solve("10 * 5 + 3")
    assert result == 53


def test_solver_syntax_error_handling():
    """Test that TheSolver handles syntax errors correctly with retries"""
    solver = TheSolver()
    # This should not raise an exception but return an error message
    result = solver.solve("2 + + 2")  # Invalid syntax
    assert "Error" in str(result) or result == 0


def test_solver_sandbox_security():
    """Test that TheSolver sandbox prevents dangerous operations"""
    solver = TheSolver()
    # Try to execute dangerous code
    dangerous_code = "result = __import__('os')"
    result = solver.execute_in_sandbox(dangerous_code)
    # Should not be able to import os
    assert "Error" in str(result) or result == 0


def test_solver_retry_mechanism():
    """Test that TheSolver retry mechanism works"""
    solver = TheSolver(max_retries=3)
    # This should not raise an exception but return an error message
    result = solver.solve("2 + + 2")  # Invalid syntax
    assert "Error" in str(result) or result == 0


def test_solver_capabilities():
    """Test that TheSolver returns correct capabilities"""
    solver = TheSolver()
    capabilities = solver.get_capabilities()
    assert "name" in capabilities
    assert "description" in capabilities
    assert "max_retries" in capabilities
    assert capabilities["name"] == "TheSolver"


def test_solver_dsl_generation():
    """Test that TheSolver can generate DSL correctly"""
    solver = TheSolver()
    code = solver.generate_dsl("2 + 2")
    assert "result =" in code
    assert "2 + 2" in code


def test_solver_validation():
    """Test that TheSolver validates input correctly"""
    solver = TheSolver()
    # Valid input
    try:
        solver._validate_query("2 + 2")
        assert True
    except Exception:
        assert False, "Valid query should not raise exception"
    
    # Invalid input
    try:
        solver._validate_query("")
        assert False, "Empty query should raise exception"
    except ValueError:
        assert True
    
    # Dangerous input
    try:
        solver._validate_query("import os")
        assert False, "Dangerous query should raise exception"
    except ValueError:
        assert True
