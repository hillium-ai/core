import json
import time
import os
import uuid
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SyntheticMemory:
    """
    Data class for synthetic memory entries generated during simulation.
    """
    id: str
    content: str
    metadata: Dict[str, Any]
    timestamp: str


class GymSimulator:
    """
    Simulator for executing training scenarios and generating synthetic experience data.
    """
    def __init__(self):
        self.active_scenarios = []
        logger.info("Gym Simulator initialized.")
    
    def run_scenario(self, scenario_description: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Simulate a training scenario and generate experiences.
        
        Args:
            scenario_description: Description of the scenario to simulate.
            metadata: Optional metadata for the simulation.
            
        Returns:
            bool: True if simulation was successful.
        """
        logger.info(f"Starting simulation for scenario: {scenario_description}")
        
        # Simulate some processing time
        time.sleep(0.1)
        
        # In a real implementation, this would interact with a mock environment
        # and generate actual SyntheticMemory objects.
        logger.info("Simulation completed successfully.")
        return True

    def generate_synthetic_data(self, instruction: str, response: str) -> SyntheticMemory:
        """
        Manually generate a synthetic memory entry.
        """
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        memory = SyntheticMemory(
            id=memory_id,
            content=f"Instruction: {instruction}\nResponse: {response}",
            metadata={"type": "synthetic", "generated_at": timestamp},
            timestamp=timestamp
        )
        
        logger.info(f"Generated synthetic memory: {memory_id}")
        return memory

# Example usage
if __name__ == "__main__":
    simulator = GymSimulator()
    simulator.run_scenario("Obstacle avoidance in low light")
    synthetic_mem = simulator.generate_synthetic_data(
        "Move forward until you see a wall", 
        "Moving forward. Wall detected at 50cm. Stopping."
    )
    print(asdict(synthetic_mem))
