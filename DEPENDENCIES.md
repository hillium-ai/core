# HilliumOS Technical Dependencies

This document serves as the master record of all technical dependencies required for HilliumOS and Hillium-Core components.

## 🐍 Python Dependencies

Required for `loqus_core`, `gym`, and observability modules.

| Package | Version | Usage | WP Reference |
|---------|---------|-------|--------------|
| `duckdb` | `>=1.0.0` | In-process SQL observability | WP-029 |
| `sqlparse` | `latest` | SQL validation & security | WP-029 |
| `polars` | `latest` | High-performance Parquet export | WP-029 |
| `pandas` | `latest` | Generic data export (fallback) | WP-029 |
| `faster-whisper`| `latest` | STT (Speech-to-Text) | WP-018 |
| `silero-vad` | `latest` | VAD (Voice Activity Detection) | WP-018 |
| `piper-tts` | `latest` | TTS (Speech Synthesis) | WP-019 |
| `llama-cpp-python`| `latest` | Local LLM inference | WP-030 |
| `pyaudio` | `latest` | Real-time audio capture | WP-018 |
| `sounddevice` | `latest` | Audio playback & monitoring | WP-018 |
| `numpy` | `>=1.24.0` | Numerical & Tensor operations | Generic |
| `watchfiles` | `latest` | Real-time context awareness | Core |
| `textual` | `latest` | TUI Interface | UI |

## 🦀 Rust Dependencies (Primary)

Required for `hipposerver`, `aegis_core`, and the PyO3 bridge.

| Crate | Usage |
|-------|-------|
| `pyo3` | Python-Rust bidirectional bridge |
| `tokio` | Async runtime for HippoServer |
| `sled` | Level 2 Working Memory Database |
| `qdrant-client` | Level 3 Episodic Memory Manager |
| `ort` | ONNX Runtime for ML (DINOv2, V-JEPA) |
| `ndarray` | Multi-dimensional array operations |
| `serde` | Lean serialization (JSON/Bin) |

## 🐧 System Dependencies (Forge Host)

Required for low-level interaction and model acceleration.

| Package | Purpose |
|---------|---------|
| `portaudio19-dev` | Required for PyAudio (audio streams) |
| `libssl-dev` | Network security support |
| `ffmpeg` | Audio and video processing |
| `cmake` | Native build support for LlamaCpp/PyO3 |
| `pkg-config` | Driver and library discovery |
| `libopenblas-dev` | Numerical acceleration |

## 🛠️ Installation Command (Master)

```bash
# System
sudo apt update && sudo apt install -y portaudio19-dev ffmpeg cmake pkg-config libssl-dev libopenblas-dev

# Python
pip install duckdb sqlparse polars pandas faster-whisper silero-vad llama-cpp-python pyaudio sounddevice numpy textual
```
