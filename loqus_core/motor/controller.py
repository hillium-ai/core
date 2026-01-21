import json
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock HAL interface (to be replaced with actual hillium_hal)
class MockHAL:
    def __init__(self):
        self.driver_state = "idle"
        
    def move_joints(self, joints: list, speed: float = 1.0):
        logger.info(f"Moving joints: {joints} with speed {speed}")
        self.driver_state = "moving"
        time.sleep(0.1)  # Simulate movement
        self.driver_state = "idle"
        
    def gripper_open(self):
        logger.info("Opening gripper")
        self.driver_state = "gripper_open"
        time.sleep(0.1)  # Simulate action
        self.driver_state = "idle"
        
    def gripper_close(self):
        logger.info("Closing gripper")
        self.driver_state = "gripper_closed"
        time.sleep(0.1)  # Simulate action
        self.driver_state = "idle"

# Mock E-Stop interface (to be replaced with actual hippo)
class MockEStop:
    def __init__(self):
        self._is_estopped = False
        
    def is_estopped(self) -> bool:
        return self._is_estopped
        
    def set_estop(self, state: bool):
        self._is_estopped = state

# SafetyResult enum
class SafetyResult(Enum):
    SUCCESS = "success"
    REQUIRES_HUMAN = "requires_human"
    ERROR = "error"

# Primitive library mapping high-level actions to HAL commands
primitive_library = {
    "Grab Cup": [
        {"type": "gripper_close"},
        {"type": "move_joints", "joints": [0, 0, 0, 0, 0, 0], "speed": 0.5}
    ],
    "Release Cup": [
        {"type": "gripper_open"},
        {"type": "move_joints", "joints": [0, 0, 0, 0, 0, 0], "speed": 0.5}
    ],
    "Move To Position": [
        {"type": "move_joints", "joints": [0, 0, 0, 0, 0, 0], "speed": 0.5}
    ],
    "Home Position": [
        {"type": "move_joints", "joints": [0, 0, 0, 0, 0, 0], "speed": 0.5}
    ]
}

def execute_plan(
    validation_result: Dict[str, Any],
    hal_interface: Optional[MockHAL] = None,
    estop_interface: Optional[MockEStop] = None,
    rate_limit_seconds: float = 0.01
) -> SafetyResult:
    """
    Execute a validation result plan with E-Stop handling.
    
    Args:
        validation_result: The validation result containing the plan to execute
        hal_interface: HAL interface to use (defaults to MockHAL if not provided)
        estop_interface: E-Stop interface to use (defaults to MockEStop if not provided)
        rate_limit_seconds: Minimum time between commands (rate limiting)
        
    Returns:
        SafetyResult: The result of the execution
    """
    # Initialize interfaces if not provided
    if hal_interface is None:
        hal_interface = MockHAL()
    if estop_interface is None:
        estop_interface = MockEStop()
    
    # Validate input
    try:
        plan = validation_result.get("plan", [])
        if not isinstance(plan, list):
            logger.error("Invalid plan format: plan must be a list")
            return SafetyResult.ERROR
        
        # Check for E-Stop
        if estop_interface.is_estopped():
            logger.warning("E-Stop triggered, halting execution")
            return SafetyResult.ERROR
        
        # Process each step in the plan
        last_command_time = 0
        for step in plan:
            # Check for E-Stop during execution
            if estop_interface.is_estopped():
                logger.warning("E-Stop triggered during execution, halting")
                return SafetyResult.ERROR
            
            # Rate limiting
            current_time = time.time()
            time_since_last = current_time - last_command_time
            if time_since_last < rate_limit_seconds:
                sleep_time = rate_limit_seconds - (current_time - last_command_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            # Handle different step types
            if step.get("type") == "RequiresHuman":
                logger.info("Human intervention required, skipping step")
                continue
            
            # Execute primitive
            action = step.get("action")
            if action in primitive_library:
                primitives = primitive_library[action]
                for primitive in primitives:
                    # Check E-Stop before each primitive
                    if estop_interface.is_estopped():
                        return SafetyResult.ERROR
                        
                    primitive_type = primitive["type"]
                    if primitive_type == "move_joints":
                        joints = primitive.get("joints", [])
                        speed = primitive.get("speed", 1.0)
                        hal_interface.move_joints(joints, speed)
                    elif primitive_type == "gripper_open":
                        hal_interface.gripper_open()
                    elif primitive_type == "gripper_close":
                        hal_interface.gripper_close()
                    else:
                        logger.warning(f"Unknown primitive type: {primitive_type}")
            else:
                logger.warning(f"Unknown action: {action}")
            
            last_command_time = time.time()
            
        logger.info("Plan executed successfully")
        return SafetyResult.SUCCESS
        
    except Exception as e:
        logger.error(f"Error executing plan: {str(e)}")
        return SafetyResult.ERROR
