import time
import subprocess
import os
import uuid
from pathlib import Path

def measure_sync_lag(iterations=10):
    """
    Measures the time it takes for a file written on Host (MacOS)
    to become visible inside the Docker container (Hillium-Forge).
    """
    print(f"🔬 EXPERIMENT: Measuring MacOS -> Docker Sync Lag ({iterations} iterations)")
    print(f"📂 CWD: {os.getcwd()}")
    
    delays = []
    failures = 0
    
    for i in range(iterations):
        # 1. Generate unique filename
        filename = f"sync_test_{uuid.uuid4().hex}.tmp"
        host_path = Path(filename).resolve()
        
        # 2. Prepare Docker command (stat to check existence)
        # We use stat because it's fast and returns 0 only if file exists
        docker_cmd = ["docker", "exec", "hillium-forge", "stat", f"/workspace/{filename}"]
        
        # 3. WRITE on Host
        start_time = time.time()
        with open(host_path, "w") as f:
            f.write("synchronize_me")
        write_done_time = time.time()
        
        # 4. POLL in Docker immediately
        visible_time = None
        attempts = 0
        max_attempts = 100 # 5 seconds max (50ms sleep)
        
        while attempts < max_attempts:
            res = subprocess.run(
                docker_cmd, 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            if res.returncode == 0:
                visible_time = time.time()
                break
            # Busy wait or tiny sleep? simple busy-ish check
            # time.sleep(0.01) # 10ms
            attempts += 1
            
        # 5. Cleanup
        if host_path.exists():
            os.remove(host_path)
            
        # 6. Record Results
        if visible_time:
            lag = (visible_time - write_done_time) * 1000 # ms
            delays.append(lag)
            print(f"   Run {i+1}: Lag = {lag:.2f} ms (Attempts: {attempts})")
        else:
            failures += 1
            print(f"   Run {i+1}: ❌ TIMEOUT (> {max_attempts*10}ms)")

    # 7. Statistics
    if delays:
        avg = sum(delays) / len(delays)
        min_d = min(delays)
        max_d = max(delays)
        print(f"\n📊 RESULTS:")
        print(f"   Average Lag: {avg:.2f} ms")
        print(f"   Min Lag:     {min_d:.2f} ms")
        print(f"   Max Lag:     {max_d:.2f} ms")
        print(f"   Failures:    {failures}")
        
        if max_d > 100:
            print("\n⚠️ CONCLUSION: Significant Sync Lag Detected (>100ms).")
            print("   Titan checks occurring <100ms after write WILL fail.")
        else:
            print("\n✅ CONCLUSION: Sync is fast (<100ms). Lag might not be the main issue.")
    else:
        print("\n❌ CRITICAL: No files synced successfully.")

if __name__ == "__main__":
    measure_sync_lag()
