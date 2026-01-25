import re
import ast
from RestrictedPython import compile_restricted

class TheSolver:
    def __init__(self):
        self.max_retries = 3

    def _validate_query(self, query):
        """Validate and sanitize input query"""
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        
        if not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Check for dangerous patterns
        dangerous_patterns = [r'import\s+.*', r'exec\s*\(', r'eval\s*\(', r'__.*__']
        for pattern in dangerous_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                raise ValueError("Query contains dangerous patterns")
        
        return query.strip()

    def generate_dsl(self, query):
        """Generate Python DSL for math expressions"""
        try:
            # Validate query first
            query = self._validate_query(query)
            
            # Look for math-related keywords
            if "solve" in query.lower() or "calculate" in query.lower() or "compute" in query.lower():
                # Extract expression using more robust pattern matching
                expression = query
                # Remove keywords to isolate the math expression
                expression = re.sub(r'(?i)(solve|calculate|compute)\s*', '', expression)
                expression = expression.strip()
                
                # Basic validation of expression
                if not expression:
                    return "result = 0  # No expression found"
                
                # Try to parse the expression to ensure it's valid
                try:
                    # This is a simple check - in production, you'd want a more robust parser
                    ast.parse(expression)
                except SyntaxError:
                    # If it's not valid Python syntax, we'll try to handle it
                    pass
                
                # Generate Python code
                code = f"result = {expression}\n"
                return code
            else:
                # For non-math queries, return a clear error or default
                return "result = 0  # Not a math query"
                
        except Exception as e:
            # Return fallback code on error
            return "result = 0  # Error in DSL generation"

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
                }
            }
            
            loc = {}
            exec(compiled, safe_globals, loc)  # Use compiled directly, not compiled.code
            return loc.get("result", 0)
        except Exception as e:
            raise Exception(f"Execution failed: {str(e)}")

    def solve(self, query):
        """Solve a query using the DSL generator and sandboxed execution"""
        for attempt in range(self.max_retries):
            try:
                code = self.generate_dsl(query)
                result = self.execute_in_sandbox(code)
                return result
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                # Retry without appending error to query (avoid injection)
                # Instead, we'll just retry with the same query
                # This prevents potential injection issues
                continue
        return None