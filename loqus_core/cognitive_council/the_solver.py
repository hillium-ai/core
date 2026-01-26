#!/usr/bin/env python3
"""TheSolver - Mathematical Problem Solver for the Cognitive Council"""

import re
import logging
import time
from RestrictedPython import compile_restricted

logger = logging.getLogger(__name__)

class TheSolver:
    """Mathematical problem solver that generates and executes Python DSL in a secure sandbox"""
    
    def __init__(self, max_retries=3):
        self.max_retries = max_retries

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
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                raise ValueError("Query contains dangerous patterns")
        
        return query.strip()

    def generate_dsl(self, query):
        """Generate Python DSL for math expressions"""
        try:
            # Validate query first
            query = self._validate_query(query)
            
            # Simple expression extraction
            expression = query
            
            # Remove common question words and phrases
            expression = re.sub(r'(?i)(what is|what\s+is|calculate|compute|solve|find)\s*', '', expression)
            expression = expression.strip()
            
            # Remove trailing question marks
            expression = expression.rstrip('?')
            expression = expression.strip()
            
            # Replace common math notation
            expression = expression.replace('^', '**')
            
            # Simple code generation
            code = f"result = {expression}\n"
            return code
            
        except Exception as e:
            return "result = 'Error in DSL generation'"

    def execute_in_sandbox(self, code):
        """Execute code in a secure sandbox using RestrictedPython"""
        try:
            # Compile the code with RestrictedPython
            compiled = compile_restricted(code, filename="<inline>", mode="exec")
            
            # Create a safe execution environment
            safe_globals = {
                "__builtins__": {
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
                }
            }
            
            loc = {}
            exec(compiled, safe_globals, loc)
            
            # Return the result
            return loc.get("result", 0)
        except SyntaxError as e:
            # Return syntax error as string instead of raising it
            return f"Error: Syntax error - {str(e)}"
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
            except ValueError:
                return "Error: Invalid query"
            
            # Generate DSL
            code = self.generate_dsl(query)
            
            # Execute in sandbox
            result = self.execute_in_sandbox(code)
            
            # If we get a syntax error, we should retry
            if isinstance(result, str) and result.startswith("Error: Syntax error"):
                if attempt < self.max_retries - 1:  # Don't wait on the last attempt
                    time.sleep(0.1)  # Brief delay before retry
                    continue  # Retry
                else:
                    return result  # Return the error after max retries
            
            # If we get a valid result or non-syntax error, return it
            return result
        
        # This should not be reached, but just in case
        return "Error: Max retries exceeded"

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
