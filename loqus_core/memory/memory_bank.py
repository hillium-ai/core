from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid


class MemoryBank:
    """
    MemoryBank for storing and retrieving beliefs, plans, and other cognitive memories.
    
    This is a simplified implementation for the test environment.
    """
    
    def __init__(self):
        self.memory_store: Dict[str, Any] = {}
        self.beliefs: Dict[str, Dict[str, Any]] = {}
        self.plans: Dict[str, Dict[str, Any]] = {}
        self.results: Dict[str, Dict[str, Any]] = {}
        
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state from memory.
        
        Returns:
            Dictionary containing current state information
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "memory_size": len(self.memory_store)
        }
        
    def store_plan(self, plan: Dict[str, Any]) -> str:
        """
        Store a plan in memory.
        
        Args:
            plan: The plan to store
            
        Returns:
            ID of the stored plan
        """
        plan_id = str(uuid.uuid4())
        self.plans[plan_id] = plan
        self.memory_store[plan_id] = plan
        return plan_id
        
    def store_result(self, key: str, result: Dict[str, Any]) -> None:
        """
        Store a result in memory.
        
        Args:
            key: Key to store the result under
            result: The result to store
        """
        self.results[key] = result
        self.memory_store[key] = result
        
    def get_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a plan from memory.
        
        Args:
            plan_id: ID of the plan to retrieve
            
        Returns:
            The plan if found, None otherwise
        """
        return self.plans.get(plan_id)
        
    def get_result(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a result from memory.
        
        Args:
            key: Key of the result to retrieve
            
        Returns:
            The result if found, None otherwise
        """
        return self.results.get(key)
        
    def get_all_plans(self) -> List[Dict[str, Any]]:
        """
        Get all stored plans.
        
        Returns:
            List of all stored plans
        """
        return list(self.plans.values())
        
    def get_all_results(self) -> List[Dict[str, Any]]:
        """
        Get all stored results.
        
        Returns:
            List of all stored results
        """
        return list(self.results.values())