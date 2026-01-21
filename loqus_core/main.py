#!/usr/bin/env python3
"""Main Heartbeat Loop for LoqusCore - 20Hz processing cycle."""

import asyncio
import signal
import threading
import time
import json
from typing import Optional, Any

# Local imports
try:
    from loqus_core.router import HippoLink, HippoState
except ImportError:
    # Fallback if names changed during restoration
    from loqus_core.hippo import HippoLink, HippoState

from loqus_core.inference.manager import NativeModelManager
from loqus_core.council.core import CognitiveCouncil
from loqus_core.safety.layer7 import CognitiveSafetyValidator


class Loop:
    """Main Heartbeat Loop controller."""
    
    def __init__(self):
        """Initialize all subsystems."""
        self.running = False
        self.heartbeat_thread: Optional[threading.Thread] = None
        
        # Initialize subsystems
        self.hippo = HippoLink()
        self.model_manager = NativeModelManager()
        self.council = CognitiveCouncil()
        self.safety = CognitiveSafetyValidator()
        
        # System status flags
        self.boot_time = time.time()
    
    async def heartbeat_worker(self):
        """Dedicated worker for updating heartbeat in SHM."""
        while self.running:
            try:
                # Update heartbeat namespace
                self.hippo.beat_heart()
            except Exception as e:
                print(f"Heartbeat error: {e}")
            await asyncio.sleep(0.05)  # 20Hz
    
    async def process_cycle(self):
        """Single processing cycle: Read -> Build Context -> Plan -> Validate -> Execute."""
        # Check system state
        state = self.hippo.get_state()
        
        # If there's a pending intent or input
        if state.get("pending_input"):
            user_input = state.get("last_input")
            
            # 1. Build Context (RAG / Memory)
            context = self.model_manager.build_context(user_input)
            
            # 2. Planning (Cognitive Council)
            plan = self.council.create_plan(context)
            
            # 3. Safety Validation (Layer 7)
            validation = self.safety.validate(plan)
            
            # 4. Execution (Dispatch to Motor Cortex / SynApps)
            if validation.approved:
                self.hippo.execute(plan)
            else:
                print(f"Safety rejection: {validation.reason}")
    
    async def main_loop(self):
        """Main async loop running at 20Hz."""
        while self.running:
            start_time = time.time()
            
            # Execute processing cycle
            await self.process_cycle()
            
            # Maintain 20Hz cycle
            elapsed = time.time() - start_time
            sleep_time = max(0, 0.05 - elapsed)
            await asyncio.sleep(sleep_time)
    
    def start(self):
        """Start the heartbeat loop."""
        print("Hillium Heartbeat Loop Starting...")
        self.running = True
        
        # Start main async loop
        try:
            asyncio.run(self.main_loop())
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Graceful shutdown."""
        print("\nShutting down Heartbeat Loop...")
        self.running = False


def signal_handler(signum, frame):
    """Handle SIGINT for graceful shutdown."""
    if 'hb_loop' in globals():
        globals()['hb_loop'].stop()
    exit(0)


if __name__ == "__main__":
    # Setup signal handling
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize and start loop
    hb_loop = Loop()
    hb_loop.start()
