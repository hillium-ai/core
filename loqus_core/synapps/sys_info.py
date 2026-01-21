import json
import sys
import os
from datetime import datetime
import platform

def get_system_info():
    """
    Get basic system information without external dependencies
    """
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "system": platform.system(),
        "timestamp": datetime.now().isoformat()
    }


def main():
    """
    Main function that reads input and outputs system info
    """
    try:
        # Read input from stdin
        input_data = sys.stdin.read().strip()
        
        if not input_data:
            # No input, return basic system info
            info = get_system_info()
        else:
            # Parse input (though we don't use it in this example)
            try:
                input_obj = json.loads(input_data)
                info = get_system_info()
            except json.JSONDecodeError:
                 info = {"error": "Invalid input JSON"}
            
        # Output result as JSON
        result = {
            "success": True,
            "result": info,
            "error": None
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        # Output error
        error_result = {
            "success": False,
            "result": None,
            "error": str(e)
        }
        print(json.dumps(error_result))


if __name__ == "__main__":
    main()
