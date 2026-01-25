
import sys
import os
from pathlib import Path

# Force use of local source
sys.path.insert(0, "/Users/jsaldana/GitLocalRepo/Hillium-Levitate/src")

from levitate.observability.analyst import run_analysis_cli

if __name__ == "__main__":
    print(f"Forcing regeneration for run_1769303529_7e93fc from {os.getcwd()}")
    try:
        run_analysis_cli(run_id="run_1769303529_7e93fc")
        print("Regeneration complete.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
