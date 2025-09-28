<p align="center">
  <img src="https://www.hillium.ai/images/2048/19472636/logo_hillium_negro-HdJhznK4BDXT4RmEhZ9DJQ.png" alt="Hillium Logo" width="400"/>
</p>

<h1 align="center">Hillium Orchestrator Core (HOC)</h1>

<p align="center">
  <strong>The Nervous System for a Kinetic World.</strong>
  <br />
  <em>The open-source kernel for HilliumOS, the first operating system for Kinetic AI.</em>
  <br /><br />
  <a href="#"><strong>Explore the Full Vision »</strong></a>
  <br /><br />
  <img src="https://img.shields.io/badge/license-AGPL--3.0-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/status-in%20development-orange.svg" alt="Status">
  <a href="#"><img src="https://img.shields.io/badge/join-our%20discord-7289DA.svg" alt="Discord"></a>
</p>

---

**Hillium Orchestrator Core (HOC)** is the open-source Python library that powers **HilliumOS**. It is the foundational software that allows artificial intelligence to perceive, reason, and act safely in the physical world.

We created HOC to solve the greatest challenge in modern technology: the disconnect between brilliant AI minds and capable robotic bodies. HOC provides the universal nervous system to unify them.

## Key Capabilities

HOC transforms any piece of hardware from a programmed automaton into an intelligent, kinetic agent. This is made possible through our core architecture:

- 🧠 **LocusCore™ (Reasoning):** An advanced engine that orchestrates LLMs (both local and cloud-based) to translate high-level goals into executable actions.
- 🔌 **SynAppCore™ (Connectivity):** A universal standard for connectors. Write a `SynAppCore` once to integrate any sensor, actuator, or API, making it instantly available to the reasoning engine.
- 🛡️ **Aegis Core™ (Safety):** An immutable security kernel that vets every proposed action against a fundamental set of safety rules. It's the "spinal cord" that ensures all actions are safe by design.
- 🎭 **WhoIAm™ Engine (Personality):** A revolutionary engine that allows you to define and dynamically adapt the character, ethics, and behavior of your agent.

## Quick Start

Getting started with the Hillium Orchestrator Core is simple.

### 1. Installation:

```bash
pip install hillium-core
```

### 2. Your First Kinetic Agent:

Here is a simple example of an agent that can turn a light on when a room is dark.

```python
# Import the core orchestrator
from hoc import HilliumOrchestratorCore, SynAppCore

# Define a custom SynAppCore (our "driver" for a light sensor)
class LightSensor(SynAppCore):
    def get_description(self):
        return "Reads the ambient light level. Returns a value from 0 (dark) to 100 (bright)."

    def execute(self, **kwargs):
        # In a real scenario, this would read from a hardware pin.
        return 5 # Simulate a dark room

# Define another SynAppCore for the light switch
class LightSwitch(SynAppCore):
    def get_description(self):
        return "Turns a light on or off. Requires a 'state' parameter ('on' or 'off')."

    def execute(self, state: str):
        print(f"💡 Light switch has been turned {state}.")
        return {"status": f"Light is now {state}"}

# Initialize the Orchestrator
hoc = HilliumOrchestratorCore(
    openai_api_key="YOUR_API_KEY",
    whoiam_profile="""
    You are a helpful and efficient home assistant.
    Your primary goal is to maintain a comfortable environment.
    Honesty Level: 1.0
    """
)

# Register our connectors with the orchestrator
hoc.register_synappcore(name="light_sensor", synappcore=LightSensor())
hoc.register_synappcore(name="light_switch", synappcore=LightSwitch())

# Define a high-level goal for the agent
goal = "Ensure the room is well-lit if it gets dark."

# Run the orchestrator
print("Executing goal...")
result = hoc.run(goal=goal)

print("\nOrchestrator execution finished.")
print("Final Result:", result)

# Expected Output:
# >>> Executing goal...
# >>> 💡 Light switch has been turned on.
# >>> Orchestrator execution finished.
# >>> Final Result: {'status': 'Light is now on'}
```

## The Vision & The Roadmap

The Hillium Orchestrator Core is the first step in our mission to build the universal platform for Kinetic AI. Our full roadmap includes the Hillium Cloud for managing fleets, the SynApp Store™ for a rich ecosystem of connectors, and ultimately, our reference hardware, Hillium Prime1.

Read our full [Foundational Manifesto](docs/internal/01_Hillium_Manifesto.md) here.

## Contributing

We are building Hillium in the open. We believe the future of Kinetic AI will be built by a global community. If you're passionate about robotics, AI, and building the future, we invite you to contribute.

Please read our CONTRIBUTING.md guide to get started.

## Community

Join the conversation and connect with other builders:

- [Join our Discord Server](#)
- [Follow us on X/Twitter](#)
- [Read our Blog](#)

## License

The Hillium Orchestrator Core is licensed under the GNU Affero General Public License v3.0. Read the LICENSE file for more details. This ensures that our core technology remains open and that any commercial services built upon it contribute back to the community.
