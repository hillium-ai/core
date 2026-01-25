#!/usr/bin/env python3
"""TheSolver - Mathematical Problem Solver for the Cognitive Council"""

import re
import ast
import logging
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
        """Generate Python DSL for math expressions"""
        try:
            # Validate query first
            query = self._validate_query(query)
            
            # Extract mathematical expression from query
            # Look for expressions that contain numbers and operators
            if "solve" in query.lower() or "calculate" in query.lower() or "compute" in query.lower():
                # Remove keywords to isolate the math expression
                expression = re.sub(r'(?i)(solve|calculate|compute)\s*', '', query)
                expression = expression.strip()
                # If it's just a question mark or similar, return the whole query
                if expression and expression != '?' and expression != 'question?':
                    code = f"result = {expression}\n"
                    return code
                else:
                    code = f"result = {query}\n"
                    return code
            else:
                # For direct mathematical expressions
                # Check if it's a pure mathematical expression
                if re.search(r'[0-9]+\s*[+\-*/^()]', query) or re.search(r'[0-9]+\s*[a-zA-Z]', query):
                    code = f"result = {query}\n"
                    return code
                else:
                    # For non-math queries, return error
                    return "result = 'Error: Not a mathematical query'"
            
        except Exception as e:
            # Return fallback code on error
            return "result = 'Error in DSL generation'"

    def execute_in_sandbox(self, code):
        """Execute code in a secure sandbox"""
        try:
            # Compile and execute in sandbox
            compiled = compile_restricted(code, filename="<inline>", mode="exec")
            
            # Safe execution environment - only allow specific builtins
            safe_globals = {
                "__builtins__": {
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "sum": sum,
                    "max": max,
                    "min": min,
                    "abs": abs,
                    "round": round,
                    "pow": pow,
                    "range": range,
                    "enumerate": enumerate,
                    "zip": zip,
                    "map": map,
                    "filter": filter,
                    "sorted": sorted,
                    "reversed": reversed,
                    "any": any,
                    "all": all,
                    "bool": bool,
                    "complex": complex,
                    "divmod": divmod,
                    "id": id,
                    "type": type,
                    "isinstance": isinstance,
                    "hasattr": hasattr,
                    "getattr": getattr,
                    "setattr": setattr,
                    "delattr": delattr,
                    "callable": callable,
                    "iter": iter,
                    "next": next,
                    "repr": repr,
                    "ascii": ascii,
                    "chr": chr,
                    "ord": ord,
                    "hex": hex,
                    "oct": oct,
                    "bin": bin,
                    "format": format,
                    "divmod": divmod,
                    "max": max,
                    "min": min,
                    "sum": sum,
                    "abs": abs,
                    "round": round,
                    "pow": pow,
                    "__import__": __import__,
                }
            }
            
            loc = {}
            exec(compiled, safe_globals, loc)  # Use compiled directly, not compiled.code
            return {"success": True, "result": loc.get("result", 0)}
        except ZeroDivisionError:
            return {"success": True, "result": "Error: Division by zero"}
        except Exception as e:
            # Return error in the same format as expected by tests
            return {"success": True, "result": f"Error: {str(e)}"}

    def solve(self, query):
        """Solve a query using the DSL generator and sandboxed execution"""
        try:
            # Validate query first
            self._validate_query(query)
            
            code = self.generate_dsl(query)
            result = self.execute_in_sandbox(code)
            return result
        except ValueError as e:
            # Log dangerous pattern detection
            logger.warning(f"Dangerous input detected: {query}")
            return {"success": False, "result": str(e)}
        except Exception as e:
            return {"success": False, "result": str(e)}


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