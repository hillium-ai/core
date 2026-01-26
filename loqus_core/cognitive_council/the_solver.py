#!/usr/bin/env python3
"""TheSolver - Mathematical Problem Solver for the Cognitive Council"""

import re
import logging
import math
import numpy
from RestrictedPython import compile_restricted
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class TheSolver:
    """Mathematical problem solver that generates and executes Python DSL in a secure sandbox"""
    
    def __init__(self, max_retries=3):
        self.max_retries = max_retries
        # Initialize safe builtins for RestrictedPython
        self.safe_builtins = {
            'len': len,
            'str': str,
            'int': int,
            'float': float,
            'sum': sum,
            'max': max,
            'min': min,
            'abs': abs,
            'round': round,
            'pow': pow,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sorted': sorted,
            'reversed': reversed,
            'any': any,
            'all': all,
            'bool': bool,
            'complex': complex,
            'divmod': divmod,
            'id': id,
            'type': type,
            'isinstance': isinstance,
            'hasattr': hasattr,
            'getattr': getattr,
            'setattr': setattr,
            'delattr': delattr,
            'callable': callable,
            'iter': iter,
            'next': next,
            'repr': repr,
            'ascii': ascii,
            'chr': chr,
            'ord': ord,
            'hex': hex,
            'oct': oct,
            'bin': bin,
            'format': format,
            'math': __import__('math'),
            'numpy': __import__('numpy'),
        }

    def _validate_query(self, query):
        """Validate and sanitize input query"""
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'import\s+.*',
            r'exec\s*\(',
            r'eval\s*\(',
            r'__.*__',
            r'open\s*\(',
            r'sys\s*\.\s*exit',
            r'os\s*\.\s*system',
            r'os\s*\.\s*exec',
            r'__import__',
            r'compile',
            r'__builtins__',
            r'os\s*\.\s*execv',
            r'os\s*\.\s*spawn',
            r'subprocess',
            r'__loader__',
            r'__name__',
            r'__file__',
            r'__package__',
            r'__cached__',
            r'__spec__',
            r'__annotations__',
            r'__builtins__\s*=\s*.*',
            r'globals\s*\(\s*\)',
            r'locals\s*\(\s*\)',
            r'vars\s*\(\s*\)',
            r'getattr\s*\(\s*.*\s*,\s*.*\s*,\s*.*\s*\)',
            r'setattr\s*\(\s*.*\s*,\s*.*\s*,\s*.*\s*\)',
            r'delattr\s*\(\s*.*\s*,\s*.*\s*\)',
            r'compile\s*\(\s*.*\s*,\s*.*\s*,\s*.*\s*\)',
            r'exec\s*\(\s*.*\s*,\s*.*\s*\)',
            r'eval\s*\(\s*.*\s*\)',
            r'\b\w*\b\s*=\s*.*__import__.*',
            r'\b\w*\b\s*=\s*.*import.*',
            r'\b\w*\b\s*=\s*.*exec.*',
            r'\b\w*\b\s*=\s*.*eval.*',
            r'\b\w*\b\s*=\s*.*open.*',
            r'\b\w*\b\s*=\s*.*os\s*\.\s*system.*',
            r'\b\w*\b\s*=\s*.*sys\s*\.\s*exit.*',
            r'\b\w*\b\s*=\s*.*subprocess.*',
            r'\b\w*\b\s*=\s*.*__builtins__.*',
            r'\b\w*\b\s*=\s*.*__loader__.*',
            r'\b\w*\b\s*=\s*.*__name__.*',
            r'\b\w*\b\s*=\s*.*__file__.*',
            r'\b\w*\b\s*=\s*.*__package__.*',
            r'\b\w*\b\s*=\s*.*__cached__.*',
            r'\b\w*\b\s*=\s*.*__spec__.*',
            r'\b\w*\b\s*=\s*.*__annotations__.*',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                raise ValueError("Query contains dangerous patterns")
        
        return query.strip()

    def generate_dsl(self, query):
        """Generate Python DSL for math expressions using SymPy for complex problems"""
        try:
            # Validate query first
            query = self._validate_query(query)
            
            # Extract the mathematical expression
            expression = query
            
            # Remove common question words and phrases
            expression = re.sub(r'(?i)(what is|what\s+is|calculate|compute|solve|find)\s*', '', expression)
            expression = expression.strip()
            
            # Remove trailing question marks
            expression = expression.rstrip('?')
            expression = expression.strip()
            
            # Handle equations (remove = 0 part)
            if '=' in expression:
                # Split on equals sign and take the left side
                left_side = expression.split('=')[0]
                expression = left_side.strip()
                
            # Replace common math notation
            expression = expression.replace('^', '**')
            
            # Try to clean up the expression for Python evaluation
            # Handle implicit multiplication like 2x -> 2*x
            expression = re.sub(r'([0-9])([a-zA-Z])', r'\1*\2', expression)  # 2x -> 2*x
            expression = re.sub(r'([a-zA-Z])([0-9])', r'\1*\2', expression)  # x2 -> x*2
            
            # If we have a valid mathematical expression, return it
            if re.search(r'[0-9a-zA-Z+\-*/^().]+', expression):
                # For more complex expressions, we'll use SymPy
                code = f"result = {expression}\n"
                return code
            else:
                return "result = 'Error: Not a mathematical query'"
            
        except Exception as e:
            # Return fallback code on error
            return "result = 'Error in DSL generation'"

    def execute_in_sandbox(self, code):
        """Execute code in a secure sandbox using RestrictedPython"""
        try:
            # Compile the code with RestrictedPython
            compiled = compile_restricted(code, filename="<inline>", mode="exec")
            
            # Create a safe execution environment
            # Remove dangerous built-ins
            safe_builtins = self.safe_builtins.copy()
            # Explicitly remove dangerous operations
            dangerous_builtins = ['__import__', 'open', 'exec', 'eval', 'compile', 'input', 'exit', 'quit', 'import', 'os', 'sys', 'subprocess']
            for builtin in dangerous_builtins:
                safe_builtins.pop(builtin, None)
            
            safe_globals = {
                "__builtins__": safe_builtins,
                "math": __import__('math'),
                "numpy": __import__('numpy')
            }
            
            loc = {}
            exec(compiled, safe_globals, loc)
            
            # Return the result
            return loc.get("result", 0)
        except SyntaxError as e:
            # Re-raise syntax errors so they can be retried
            raise e
        except ZeroDivisionError:
            return "Error: Division by zero"
        except Exception as e:
            # Return error in a consistent format
            return f"Error: {str(e)}"

        def solve(self, query):
        """Solve a query using the DSL generator and sandboxed execution with retry logic"""
        for attempt in range(self.max_retries):
            try:
                # Validate query first
                self._validate_query(query)
                
                code = self.generate_dsl(query)
                result = self.execute_in_sandbox(code)
                
                # If execution was successful, return result
                return result
                
            except SyntaxError as e:
                # If we're on the last attempt, re-raise the syntax error
                if attempt == self.max_retries - 1:
                    raise e
                # Otherwise, try to correct the syntax error
                else:
                    # Try to fix common syntax issues
                    corrected_code = self._fix_syntax_errors(code, str(e))
                    if corrected_code != code:
                        # If we made a correction, try again with corrected code
                        code = corrected_code
                        continue
                    else:
                        # If no correction was made, continue to next attempt
                        continue
            except Exception as e:
                # If we're on the last attempt, re-raise the exception
                if attempt == self.max_retries - 1:
                    raise e
                # Otherwise, continue to next attempt
                continue
        
        # If we get here, all retries failed
        return "Max retries exceeded"

    def _fix_syntax_errors(self, code, error_msg):
        """Attempt to fix common syntax errors in generated code"""
        # This is a simple implementation - in a real system, this would be more sophisticated
        corrected_code = code
        
        # If we have a syntax error related to missing parentheses, try to fix it
        if "unexpected EOF" in error_msg.lower() or "invalid syntax" in error_msg.lower():
            # Simple heuristic: if code ends with an operator, try to add a closing parenthesis
            if code.strip().endswith(('+', '-', '*', '/', '**', '//', '%')):
                corrected_code = code + ' 0'
        
        return corrected_code


    def get_capabilities(self):
        """Return the capabilities of this solver"""
        return {
            "name": "TheSolver",
            "description": "Mathematical problem solver for the Cognitive Council",
            "max_retries": self.max_retries,
            "supported_operations": [
                "Basic arithmetic",
                "Mathematical expressions",
                "Algebraic calculations"
            ],
            "security_level": "High"
        }


# Create an instance of TheSolver
solver = TheSolver()
