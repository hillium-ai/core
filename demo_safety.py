
import sys
import asyncio
from pathlib import Path
import subprocess

# Add Hillium-Levitate source to path
sys.path.append("/Users/jsaldana/GitLocalRepo/Hillium-Levitate/src")

from levitate.operations.git_snapshot_manager import ensure_wp_snapshot

async def demo():
    print("🚦 Starting Git Safety Demo on hillium-core...")
    repo_root = Path(".")
    
    # 1. Modify a dummy file to ensure dirty state
    dummy = repo_root / "SAFETY_DEMO.md"
    dummy.write_text("# Safety Demo\nThis file mimics uncommitted work.")
    print("📝 Created dirty file: SAFETY_DEMO.md")
    
    # 2. Run Snapshot
    print("🛡️ Executing ensure_wp_snapshot('WP-SAFETY-DEMO')...")
    try:
        tag = await ensure_wp_snapshot("WP-SAFETY-DEMO", repo_root)
        print(f"✅ Snapshot Success! Tag: {tag}")
    except Exception as e:
        print(f"❌ Snapshot Failed: {e}")
        return

    # 3. Verify
    print("\n🔍 Verification:")
    log = subprocess.run(["git", "log", "-1"], capture_output=True, text=True).stdout
    print(log)
    
    if "[🛡️ Safety]" in log:
        print("🎉 SUCCESS: Auto-Commit detected!")
    else:
        print("⚠️ FAILURE: Auto-Commit not found in HEAD.")

if __name__ == "__main__":
    asyncio.run(demo())
