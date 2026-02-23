<p align="center">
  <img src="https://www.hillium.ai/images/2048/19472636/logo_hillium_negro-HdJhznK4BDXT4RmEhZ9DJQ.png" alt="Hillium Logo" width="400"/>
</p>

<h1 align="center">HilliumOS Core</h1>

<p align="center">
  <strong>The Nervous System for Kinetic AI.</strong>
  <br />
  <em>The foundational kernel for HilliumOS - enabling embodied AI to perceive, reason, and act safely in the physical world.</em>
  <br /><br />
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/status-MVP%20Development-orange.svg" alt="Status">
  <img src="https://img.shields.io/badge/rust-1.75+-orange.svg" alt="Rust">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python">
</p>

---

## 🎯 The Mission

**HilliumOS Core** is a high-performance, biologically-inspired kernel designed for autonomous robots and kinetic systems. It provides a multi-layer architecture that combines the industrial safety of **Rust** with the cognitive flexibility of **Python**, creating a deterministic and trustworthy bridge between digital intelligence and physical motion on real hardware.

---

## 🧠 Core Architecture

HilliumOS is organized into specialized "Cores" that mimic the human nervous system:

```mermaid
graph TD
    A[Aura Engine - Perception] -->|Sensory Buffer| B[HippoServer - Memory]
    C[LoqusCore - Cognition] <-->|Context/Retrieval| B
    C -->|ActionPlan| D[Aegis Core - Safety]
    D -->|Validated Command| E[Motor Cortex - Control]
    E -->|Physical Actuation| F[Hardware/Sim]
```

### 🔭 Technological Stack (MVP Core)

Our stack is curated for maximum reliability and local-first execution, optimized for edge hardware (Jetson Orin, Mac Studio).

| Component | Technology | Primary Libraries | Usage |
|-----------|------------|-------------------|-------|
| **Perception** | Python/ONNX | `faster-whisper`, `silero-vad`, `DINOv2` | Real-time Senses & Visual Validation |
| **Cognition** | Python/LLM | `LangChain`, `sqlparse`, `duckdb` | The Cognitive Council & SQL Observability |
| **Memory** | Rust | `sled`, `qdrant`, `rkyv`, `Zero-Copy IPC` | 4-Level Hierarchy (Sensory to Episodic) |
| **Safety** | Rust | `Aegis L7`, `VisualValidator` | Hallucination Prevention & Real-time Gating |
| **Control** | Rust | `MuJoCo`, `PyO3`, `ort` | Trajectory Planning & Hardware Abstraction |

---

## 🚀 Installation & Hardware Readiness

The system is designed to run natively on Mac and Linux. We provide a bootstrap script that configures your hardware's specific accelerators (Metal for Mac, CUDA for Jetson/Linux).

### Prerequisites
- **Hardware**: Mac M1/M2/M3 or NVIDIA Jetson Orin (Recommended)
- **Rust 1.75+**
- **Python 3.11+**

### Native Bootstrap
```bash
# Clone the repository
git clone https://github.com/hillium-ai/hillium-core.git
cd hillium-core

# Universal Bootstrap (Configures Metal/CUDA automatically)
./scripts/setup_dev.sh
```

> [!NOTE]
> While we provide Docker support via `infrastructure/docker/` for isolated testing, the system is optimized for **native execution** to ensure low-latency access to hardware sensors and motor controllers.

---

## 🛠️ Detailed Documentation

- **[DEPENDENCIES.md](DEPENDENCIES.md)**: Master manifest of Python, Rust, and System requirements for real-hardware deployment.
- **[DEVELOPMENT.md](DEVELOPMENT.md)**: Guidelines for TDD, conventional commits, and CI/CD pipelines.
- **[CLAUDE.md](CLAUDE.md)**: Cultural and technical context for AI agents working on this project.

---

## 🏁 Verification & Testing

HilliumOS Core is built with a **Zero Tolerance for regressions** policy on both simulated and real hardware.

```bash
# Run the complete test suite
pytest tests/ -v
cargo test --all

# Audit the environment setup
python3 scripts/verify_env.py
```

---

## 🤝 Contributing & Community

Join the revolution in Kinetic AI. We follow the **Open Core** philosophy.

- 💬 [Join Discord](https://discord.gg/n7ChqvPWgR)
- 🌐 [Visit Hillium.ai](https://www.hillium.ai)
- 📄 [MIT License](LICENSE)

<p align="center">
  <strong>HilliumOS Core</strong><br>
  Built with ❤️ for the future of Kinetic AI
</p>
