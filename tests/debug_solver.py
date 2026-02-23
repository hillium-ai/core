import ast

code = 'result = 2 + + 2'
print('Testing AST parsing:')
try:
    ast.parse(code)
    print('No syntax error in code')
except SyntaxError as e:
    print('Syntax error:', e)

# Now test what happens when we actually execute this
try:
    result = eval(code)
    print('Eval result:', result)
except SyntaxError as e:
    print('Eval syntax error:', e)