import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from loqus_core.cognitive_council import TheSolver
from loqus_core.memory import MemoryBank
from loqus_core.reasoning import ReasoningEngine
from loqus_core.safety import SafetyMonitor

logger = logging.getLogger(__name__)

@dataclass
class RerouterConfig:
    """Configuration for the RerouterNetwork"""
    max_retries: int = 3
    timeout_seconds: int = 30
    enable_logging: bool = True
    

class RerouterNetwork:
    """Network for rerouting tasks and managing execution flow"""
    
    def __init__(self, config: Optional[RerouterConfig] = None):
        self.config = config or RerouterConfig()
        self.cognitive_council = TheSolver()
        self.memory_bank = MemoryBank()
        self.reasoning_engine = ReasoningEngine()
        self.safety_monitor = SafetyMonitor()
        self.active_rerouters = {}
        
    async def route_task(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Route a task through the network"""
        try:
            # Log the routing attempt
            if self.config.enable_logging:
                logger.info(f"Routing task {task_id} with data: {task_data}")
            
            # Check safety first
            safety_check = await self.safety_monitor.check_safety(task_data)
            if not safety_check.is_safe:
                return {
                    "status": "rejected",
                    "reason": safety_check.reason,
                    "task_id": task_id
                }
            
            # Determine optimal routing path
            routing_path = await self._determine_routing_path(task_data)
            
            # Execute the task through the determined path
            result = await self._execute_task_with_path(task_id, task_data, routing_path)
            
            # Store result in memory
            await self.memory_bank.store_result(task_id, result)
            
            return {
                "status": "completed",
                "result": result,
                "task_id": task_id,
                "routing_path": routing_path
            }
            
        except Exception as e:
            logger.error(f"Error routing task {task_id}: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "task_id": task_id
            }
    
    async def _determine_routing_path(self, task_data: Dict[str, Any]) -> list:
        """Determine the optimal routing path for a task"""
        # This is a simplified implementation
        # In a real system, this would use reasoning and memory to determine the best path
        
        # Simple heuristic: route based on task type
        task_type = task_data.get("type", "default")
        
        if task_type == "complex_reasoning":
            return ["reasoning_engine", "cognitive_council", "memory_bank"]
        elif task_type == "data_processing":
            return ["memory_bank", "safety_monitor"]
        else:
            return ["cognitive_council", "safety_monitor"]
    
    async def _execute_task_with_path(self, task_id: str, task_data: Dict[str, Any], routing_path: list) -> Dict[str, Any]:
        """Execute a task through the specified routing path"""
        result = task_data.copy()
        
        for component in routing_path:
            if component == "cognitive_council":
                result = await self.cognitive_council.process(result)
            elif component == "memory_bank":
                result = await self.memory_bank.process(result)
            elif component == "reasoning_engine":
                result = await self.reasoning_engine.process(result)
            elif component == "safety_monitor":
                safety_check = await self.safety_monitor.check_safety(result)
                if not safety_check.is_safe:
                    raise Exception(f"Safety check failed: {safety_check.reason}")
                
        return result
    
    async def retry_route_task(self, task_id: str, task_data: Dict[str, Any], max_retries: Optional[int] = None) -> Dict[str, Any]:
        """Retry routing a task with exponential backoff"""
        retries = max_retries or self.config.max_retries
        
        for attempt in range(retries):
            try:
                result = await self.route_task(task_id, task_data)
                if result["status"] == "completed":
                    return result
                
                # Wait before retrying (exponential backoff)
                await asyncio.sleep(2 ** attempt)
                
            except Exception as e:
                logger.warning(f"Retry attempt {attempt + 1} failed for task {task_id}: {str(e)}")
                
        return {
            "status": "failed",
            "error": f"Max retries ({retries}) exceeded",
            "task_id": task_id
        }
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get current status of the rerouter network"""
        return {
            "active_rerouters": len(self.active_rerouters),
            "config": self.config.__dict__,
            "timestamp": datetime.now().isoformat()
        }