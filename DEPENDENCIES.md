# HilliumOS Technical Dependencies

This document manifest specifies the technical requirements for the HilliumOS Core kernel, optimized for high-performance execution on real-world edge hardware (NVIDIA Jetson, Mac M-Series).

## 🐍 Python Environment

The cognitive layer (`loqus_core`) requires Python 3.11+ with optimized libraries for local inference and observability.

| Category | Package | Purpose | Hardware Optimization |
|----------|---------|---------|-----------------------|
| **Observability** | `duckdb>=1.0.0` | In-process SQL engine | Thread-safe, analytical |
| **Security** | `sqlparse` | SQL validation & sanitization | Deterministic parsing |
| **Data Engine** | `polars`, `pyarrow` | High-performance telemetry storage | SIMD-accelerated Parquet |
| **Audio** | `faster-whisper`, `silero-vad` | Real-time speech-to-text | GPU/Metal acceleration |
| **Synthesis** | `piper-tts` | Ultra-low latency voice synthesis | Local-first inference |
| **Inference** | `llama-cpp-python` | LLM execution core | Metal/CUDA optimized |
| **UI** | `textual`, `watchfiles` | Development TUI & hot-reloading | Async-native |

## 🦀 Rust Toolchain

The foundational kernel (`hipposerver`, `aegis_core`) is built for memory safety and zero-copy performance.

| Component | usage |
|-----------|-------|
| **Memory** | `sled` (Working Memory), `qdrant` (Episodic) |
| **Inference** | `ort` (ONNX Runtime), `ndarray` |
| **Bindings** | `pyo3` (Rust-to-Python bridge) |

## 🐧 System Requirements (Native)

To interact with real hardware sensors and motor controllers, the following system-level drivers and utilities are required.

### macOS (Homebrew)
```bash
brew install portaudio ffmpeg cmake pkg-config ninja
```

### Linux (NVIDIA Jetson / Ubuntu)
```bash
sudo apt update && sudo apt install -y \
    portaudio19-dev \
    ffmpeg \
    cmake \
    ninja-build \
    pkg-config \
    libssl-dev \
    libopenblas-dev
```

---

## 🛠️ Automated Setup
For a faster onboarding experience on your host machine, use the provided bootstrap script:
```bash
./scripts/setup_dev.sh
```
The script will automatically detect your architecture and install the appropriate hardware-optimized versions of these dependencies.
