import ast
code = "2 + + 2"
try:
    ast.parse(code)
    print("No syntax error")
except SyntaxError as e:
    print("Syntax error:", e)
