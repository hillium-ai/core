import json
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhoIAmEngine:
    """
    WhoIAm Engine manages robot personality and identity.
    Handles persona loading, system prompt injection, and system state.
    """
    
    def __init__(self, persona_file: str = "personas/jarvis.json"):
        self.persona_file = persona_file
        self.current_persona = None
        self.system_state = {}
        self.load_persona()
        self.load_system_state()
    
    def load_persona(self) -> None:
        """
        Load persona from JSON file.
        """
        try:
            # Try absolute or relative path
            path = self.persona_file
            if not os.path.exists(path):
                 path = os.path.join(os.getcwd(), self.persona_file)
            
            if os.path.exists(path):
                with open(path, 'r') as f:
                    persona_data = json.load(f)
                    self.current_persona = persona_data
                    logger.info(f"Loaded persona from {path}")
            else:
                logger.error(f"Persona file {path} not found")
                self.current_persona = {}
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in persona file {self.persona_file}")
            self.current_persona = {}
        except Exception as e:
            logger.error(f"Error loading persona: {e}")
            self.current_persona = {}
    
    def load_system_state(self) -> None:
        """
        Load system state (battery, CPU, etc.).
        For now, this is a stub implementation.
        """
        # This is a stub - in a real implementation, this would access actual system data
        self.system_state = {
            "battery": 80,
            "cpu": 45,
            "memory": 60
        }
        logger.info("Loaded system state")
    
    def get_persona_prompt(self) -> str:
        """
        Get the persona prompt to prepend to system prompts.
        """
        if not self.current_persona:
            return ""
        
        persona_str = ""
        if "name" in self.current_persona:
            persona_str += f"You are {self.current_persona['name']}. "
        
        if "traits" in self.current_persona:
            traits = ", ".join(self.current_persona["traits"])
            persona_str += f"You are {traits}. "
        
        if "description" in self.current_persona:
            persona_str += f"{self.current_persona['description']} "
        
        return persona_str
    
    def get_system_prompt(self, base_prompt: str) -> str:
        """
        Get system prompt with persona and system state injected.
        """
        persona_prompt = self.get_persona_prompt()
        
        # Inject system state
        system_state_str = ""
        if self.system_state:
            battery = self.system_state.get("battery", 0)
            cpu = self.system_state.get("cpu", 0)
            memory = self.system_state.get("memory", 0)
            system_state_str = f"My battery is at {battery}%. My CPU usage is {cpu}%. My memory usage is {memory}%. "
        
        return f"{persona_prompt}{system_state_str}{base_prompt}"
    
    def switch_persona(self, persona_file: str) -> None:
        """
        Switch to a different persona.
        """
        self.persona_file = persona_file
        self.load_persona()
        logger.info(f"Switched to persona: {persona_file}")
    
    def get_weighted_graph_interface(self):
        """
        Stubbed WeightedGraph interface for future Neo4j integration.
        """
        # This is a stub - will be implemented in future work
        class WeightedGraph:
            def __init__(self):
                pass
            
            def get_node(self, node_id):
                pass
            
            def get_neighbors(self, node_id):
                pass
            
            def add_edge(self, node1, node2, weight):
                pass
        
        return WeightedGraph()

import os
# Global instance
whoiam_engine = WhoIAmEngine()
