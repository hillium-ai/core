from loqus_core.cognitive_council.the_solver import TheSolver; solver = TheSolver(); print("Security test:"); result = solver.solve("import os; os.system(\"echo hacked\")"); print("Result:", result)
