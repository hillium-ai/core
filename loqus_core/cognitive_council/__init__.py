# Cognitive Council - Main Entry Point

from .the_solver import TheSolver

# Define the roles in the Cognitive Council
ROLES = [
    "Reasoner",      # 1st role
    "Analyst",       # 2nd role
    "Evaluator",     # 3rd role
    "Synthesizer",   # 4th role
    "TheSolver",     # 5th role - New addition
]

# Initialize the council with all roles
COUNCIL = {
    "roles": ROLES,
    "solver": TheSolver(),
}

# Export the main components
__all__ = ["TheSolver", "ROLES", "COUNCIL"]