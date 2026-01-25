import ast
import sys
import re
from typing import Any, Dict, Optional

# Mock Cognitive Council integration
COUNCIL_ROLES = ['Reasoner', 'Analyst', 'Evaluator', 'Synthesizer']


class TheSolver:
    """
    TheSolver - Cognitive Council role that generates Python DSL code
    for deterministic logic execution.
    """
    
    def __init__(self):
        self.max_retries = 3
        self.role_name = "TheSolver"
        
        # Register as 5th role in council
        if len(COUNCIL_ROLES) < 5:
            COUNCIL_ROLES.append(self.role_name)
        
    def _validate_query(self, query: str) -> str:
        """
        Validate and sanitize input query to prevent injection attacks.
        """
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
    
    def generate_dsl(self, query: str) -> str:
        """
        Generate Python DSL code from natural language query.
        """
        try:
            # Validate query first
            query = self._validate_query(query)
            # Simple DSL generator for math problems
            if "calculate" in query.lower() or "solve" in query.lower() or "+" in query or "*" in query or "/" in query:
                # Extract numbers and operations
                try:
                    # Simple pattern matching for basic math
                    if "plus" in query.lower() or "+" in query:
                        parts = query.lower().replace("plus", "+").split("+")
                        numbers = [float(x.strip()) for x in parts if x.strip().replace(".", "", 1).isdigit()]
                        return f"result = {sum(numbers)}"
                    elif "times" in query.lower() or "*" in query:
                        parts = query.lower().replace("times", "*").split("*")
                        numbers = [float(x.strip()) for x in parts if x.strip().replace(".", "", 1).isdigit()]
                        if len(numbers) >= 2:
                            result = numbers[0] * numbers[1]
                            return f"result = {result}"
                        else:
                            return "result = 0"
                    elif "divided by" in query.lower() or "/" in query:
                        parts = query.lower().replace("divided by", "/").split("/")
                        numbers = [float(x.strip()) for x in parts if x.strip().replace(".", "", 1).isdigit()]
                        if len(numbers) >= 2:
                            return f"result = {numbers[0] / numbers[1]}"
                        else:
                            return "result = 0"
                    else:
                        # Default to eval for simple expressions
                        # This is a simplified approach - in production, use proper parsing
                        return f"result = {eval(query)}"
                except Exception:
                    # Fallback to basic expression
                    return "result = 0"
            else:
                # For non-math queries, return a basic structure
                return "result = 'No computation required'"

    def execute_in_sandbox(self, code: str) -> Dict[str, Any]:
        """
        Execute code in a sandboxed environment.
        """
        try:
            # Create a safe execution environment
            safe_globals = {
                "__builtins__": {
                    "abs": abs,
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "sum": sum,
                    "max": max,
                    "min": min,
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
                }
            }
            
            # Execute
            loc = {}
            exec(code, safe_globals, loc)
            
            # Return result
            return {
                "success": True,
                "result": loc.get("result", None)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def solve(self, query: str) -> Dict[str, Any]:
        """
        Main method to solve a query using DSL generation and sandboxed execution.
        """
        # Input validation
        if not isinstance(query, str) or not query.strip():
            return {
                "success": False,
                "error": "Invalid query"
            }
        
        # Generate DSL
        dsl_code = self.generate_dsl(query)
        
        # Execute with retry logic
        for attempt in range(self.max_retries):
            try:
                result = self.execute_in_sandbox(dsl_code)
                
                if result["success"]:
                    return result
                else:
                    # Retry with improved code generation
                    dsl_code = self.generate_dsl(query + " (retry)")
                    continue
            except Exception as e:
                # Handle runtime errors
                if attempt < self.max_retries - 1:
                    # Retry with modified code
                    dsl_code = self.generate_dsl(query + " (retry)")
                    continue
                else:
                    return {
                        "success": False,
                        "error": f"Failed after {self.max_retries} attempts: {str(e)}"
                    }
        
        return result

# Create instance
solver = TheSolver()

# Export for integration
__all__ = ["solver", "TheSolver"]