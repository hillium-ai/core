import ast

code = 'result = 2 + + 2'
print('Testing syntax error detection:')
try:
    ast.parse(code)
    print('No syntax error detected in AST parsing')
except SyntaxError as e:
    print('Syntax error detected:', e)

# Test what actually happens with eval
try:
    result = eval(code)
    print('Eval result:', result)
except SyntaxError as e:
    print('Eval syntax error:', e)