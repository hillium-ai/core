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
  <img src="https://img.shields.io/badge/rust-1.70+-orange.svg" alt="Rust">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
</p>

---

**HilliumOS Core** is the high-performance kernel that powers embodied AI agents. Built with Rust for safety and performance, with Python bindings for cognitive processing, it provides the fundamental infrastructure for robots and kinetic AI systems to operate safely and efficiently in the real world.

## 🎯 Overview

HilliumOS Core solves the challenge of integrating advanced AI reasoning with real-time physical control by providing a biologically-inspired architecture that separates concerns while enabling ultra-low-latency communication between components.

### Core Architecture

```
┌─────────────────────────────────────────────┐
│         loqusCore (Cognitive Layer)         │
│              Python + LangChain             │
│  - Cognitive Council (multi-agent reasoning)│
│  - WhoIAm (identity and values)             │
│  - Nested Learning (self-improvement)       │
├─────────────────────────────────────────────┤
│      HippoServer (Memory + IPC Layer)       │
│                   Rust                      │
│  - Shared Memory IPC (<10µs latency)        │
│  - Working Memory Manager                   │
│  - Associative Core (fast weights)          │
├─────────────────────────────────────────────┤
│        Aegis Core (Safety Layer)            │
│                   Rust                      │
│  - 7-Layer Safety Framework                 │
│  - Value Alignment Verification             │
│  - Real-time Safety Monitoring              │
├─────────────────────────────────────────────┤
│      Motor Cortex (Control Layer)           │
│                   Rust                      │
│  - Real-time Motor Control                  │
│  - Trajectory Planning                      │
│  - Hardware Abstraction Layer               │
└─────────────────────────────────────────────┘
```

### Key Features

- 🚀 **High Performance**: Rust-based core with <10µs IPC latency
- 🛡️ **Safety First**: Multi-layer safety framework with value alignment
- 🧠 **Cognitive Flexibility**: Python-based reasoning with LLM integration
- 🤖 **Embodied AI**: Designed for physical robots and kinetic agents
- ⚡ **Real-time**: Deterministic motor control for physical actuation
- 🔧 **Cross-platform**: Supports macOS (dev) and Linux ARM64 (Jetson deployment)

## 🚀 Quick Start

### Prerequisites

- **Rust** 1.70+ (`rustup`)
- **Python** 3.10+
- **Docker** (optional, for development)

### Build from Source

```bash
# Clone the repository
git clone https://github.com/hillium-ai/core.git
cd core

# Build all Rust components
cargo build --release

# Run tests
cargo test --all

# Check code quality
cargo clippy --all -- -D warnings
cargo fmt --all --check
```

### Cross-Compilation for NVIDIA Jetson

```bash
# Add ARM64 Linux target
rustup target add aarch64-unknown-linux-gnu

# Build for Jetson Orin
cargo build --release --target aarch64-unknown-linux-gnu
```

## 📦 Repository Structure

```
hillium-core/
├── hipposerver/          # Shared memory & cognitive state (Rust)
├── loqus_core/           # Cognitive processing engine (Python)
├── motor_cortex/         # Motor control system (Rust)
├── aegis_core/           # Safety framework (Rust)
├── common/               # Shared utilities and types
└── examples/             # Usage examples and demos
```

## 🧪 Development

### Running Tests

```bash
# All tests
cargo test --all

# Specific component
cargo test -p hipposerver

# Integration tests
cargo test --all --features integration

# Benchmarks
cargo bench
```

### Code Quality Standards

This project maintains strict quality standards:

- ✅ Zero compiler warnings
- ✅ `cargo clippy` passes with `-D warnings`
- ✅ `rustfmt` enforced formatting
- ✅ Comprehensive test coverage
- ✅ All public APIs documented

## 🏗️ Technology Stack

- **Core Language**: Rust (memory safety, performance)
- **Cognitive Layer**: Python (flexibility, AI/ML ecosystem)
- **Async Runtime**: Tokio (high-performance async I/O)
- **Python Bindings**: PyO3 (zero-copy Rust ↔ Python)
- **Storage**: Sled (embedded DB), Qdrant (vectors), Neo4j (graphs)
- **Serialization**: Rkyv (zero-copy), Pydantic (validation)
- **AI/ML**: LangChain, NumPy, custom architectures

## 🎯 Target Platforms

- **Development**: macOS (Apple Silicon + Intel)
- **Production**: Linux ARM64 (NVIDIA Jetson Orin)
- **Simulation**: Linux x86_64

## 📖 Documentation

- [Architecture Overview](docs/architecture/OVERVIEW.md) *(coming soon)*
- [API Reference](docs/api/) *(coming soon)*
- [Examples](examples/) *(coming soon)*

## 🤝 Contributing

We welcome contributions from the community! Whether it's bug fixes, new features, documentation, or examples, we appreciate your help in building the future of embodied AI.

### Development Workflow

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes
4. **Test** thoroughly (`cargo test --all`)
5. **Lint** your code (`cargo clippy --all -- -D warnings && cargo fmt --all --check`)
6. **Commit** with clear messages (`git commit -m 'Add amazing feature'`)
7. **Push** to your fork (`git push origin feature/amazing-feature`)
8. **Open** a Pull Request

### Contribution Guidelines

- Follow Rust best practices and idioms
- Maintain high code quality (zero warnings policy)
- Write tests for new functionality
- Document public APIs
- Keep commits atomic and well-described

For detailed guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md) *(coming soon)*

## 🌍 Community

Join the conversation and connect with other builders:

- 💬 [Join our Discord Server](https://discord.gg/n7ChqvPWgR)
- 🐦 [Follow us on X/Twitter](https://x.com/hilliumai)
- 🌐 [Visit Hillium.ai](https://www.hillium.ai)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Repository**: https://github.com/hillium-ai/core
- **Issues**: https://github.com/hillium-ai/core/issues
- **Pull Requests**: https://github.com/hillium-ai/core/pulls
- **Website**: https://www.hillium.ai

---

<p align="center">
  <strong>HilliumOS Core v0.1.0</strong><br>
  Status: 🚧 MVP Development<br>
  Built with ❤️ for the future of Kinetic AI
</p>
