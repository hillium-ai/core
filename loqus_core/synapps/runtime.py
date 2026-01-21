import subprocess
import json
import sys
import os
import resource
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, ValidationError
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define JSON schemas for input/output

class SynAppInput(BaseModel):
    command: str
    args: Optional[Dict[str, Any]] = None


class SynAppOutput(BaseModel):
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration: Optional[float] = None


class SynAppRuntime:
    def __init__(self, timeout: int = 5, sandbox_dir: str = "./sandbox"):
        self.timeout = timeout
        self.sandbox_dir = sandbox_dir
        
        # Create sandbox directory if it doesn't exist
        os.makedirs(sandbox_dir, exist_ok=True)
        
        # Set resource limits
        try:
            # Limit to 100MB virtual memory
            resource.setrlimit(resource.RLIMIT_AS, (100 * 1024 * 1024, 100 * 1024 * 1024))
            # Limit to 10 seconds CPU time
            resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        except ValueError as e:
            logger.warning(f"Could not set resource limits: {e}")

    def execute_synapp(self, script_path: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a SynApp script with input data
        """
        start_time = time.time()
        
        try:
            # Validate input data
            try:
                input_model = SynAppInput(**input_data)
            except ValidationError as e:
                logger.error(f"Invalid input data: {e}")
                return SynAppOutput(
                    success=False,
                    error=f"Invalid input data: {str(e)}"
                ).dict()
            
            # Check payload size (max 1MB)
            payload_size = len(json.dumps(input_data))
            if payload_size > 1024 * 1024:  # 1MB limit
                logger.error("Input payload exceeds 1MB limit")
                return SynAppOutput(
                    success=False,
                    error="Input payload exceeds 1MB limit"
                ).dict()
            
            # Prepare execution environment
            env = os.environ.copy()
            env["PYTHONPATH"] = self.sandbox_dir
            
            # Resolve absolute path for script
            abs_script_path = os.path.abspath(script_path)
            cmd = [sys.executable, abs_script_path]
            
            process = subprocess.run(
                cmd,
                input=json.dumps(input_data),
                text=True,
                capture_output=True,
                timeout=self.timeout,
                env=env,
                cwd=self.sandbox_dir
            )
            
            duration = time.time() - start_time
            
            # Parse output
            if process.returncode == 0:
                try:
                    output_data = json.loads(process.stdout)
                    
                    # Validate output
                    try:
                        output_model = SynAppOutput(**output_data)
                        return {**output_model.dict(), "duration": duration}
                    except ValidationError as e:
                        logger.error(f"Invalid output data: {e}")
                        return SynAppOutput(
                            success=False,
                            error=f"Invalid output data: {str(e)}"
                        ).dict()
                        
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON output from SynApp")
                    return SynAppOutput(
                        success=False,
                        error="Failed to parse JSON output from SynApp"
                    ).dict()
            else:
                logger.error(f"SynApp failed with return code {process.returncode}: {process.stderr}")
                return SynAppOutput(
                    success=False,
                    error=f"SynApp failed: {process.stderr}"
                ).dict()
                
        except subprocess.TimeoutExpired:
            logger.error("SynApp execution timed out")
            return SynAppOutput(
                success=False,
                error="SynApp execution timed out"
            ).dict()
        
        except Exception as e:
            logger.error(f"Unexpected error executing SynApp: {e}")
            return SynAppOutput(
                success=False,
                error=f"Unexpected error: {str(e)}"
            ).dict()

# Example usage
if __name__ == "__main__":
    runtime = SynAppRuntime()
    
    # Example input
    input_data = {
        "command": "sys_info",
        "args": {}
    }
    
    # Execute example SynApp
    result = runtime.execute_synapp("./sys_info.py", input_data)
    print(json.dumps(result, indent=2))
