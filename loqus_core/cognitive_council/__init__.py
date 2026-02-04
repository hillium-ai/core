# Cognitive Council - Main Entry Point

from .the_solver import TheSolver
from .alignment_monitor import CollectiveAlignmentMonitor
from .alignment_monitor import CollectiveAlignmentMonitor

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

class CognitiveCouncil:
    """
    Cognitive Council - A collection of specialized agents that work together
    to solve complex problems through reasoning, analysis, evaluation, and synthesis.
    """
    
    def __init__(self):
        self.roles = ROLES
        self.solver = TheSolver()
        self.alignment_monitor = CollectiveAlignmentMonitor()
        
    def get_role(self, role_name):
        """
        Get a specific role from the council
        """
        if role_name in self.roles:
            return role_name
        return None
        
    def get_solver(self):
        """
        Get the solver component of the council
        """
        return self.solver
        
    def get_all_roles(self):
        """
        Get all roles in the council
        """
        return self.roles

# Export the main components
__all__ = ["TheSolver", "ROLES", "COUNCIL", "CognitiveCouncil"]