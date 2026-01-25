#!/bin/bash
# HilliumOS Bootstrap Script v0.1
# Proposito: Configura el entorno de desarrollo detectando el hardware y SO.

set -e

echo "🚀 Starting HilliumOS Environment Setup..."

# 1. Deteccion de OS
OS_TYPE="$(uname)"
case "$OS_TYPE" in
    "Darwin")
        echo "🍎 Detected macOS (Metal Intelligence Mode)"
        # macOS specific setup if needed
        ;;
    "Linux")
        if [ -f /.dockerenv ]; then
            echo "🐳 Detected Docker Container (Hillium Forge)"
        else
            echo "🐧 Detected Linux Host"
        fi
        ;;
    *)
        echo "❓ Unknown OS: $OS_TYPE"
        ;;
esac

# 2. Instalacion de Dependencias del Sistema
echo "📦 Installing System Dependencies..."
if [ "$OS_TYPE" = "Linux" ]; then
    # Check if sudo is available, otherwise run directly
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    else
        SUDO=""
    fi
    $SUDO apt update && $SUDO apt install -y \
        portaudio19-dev \
        ffmpeg \
        cmake \
        ninja-build \
        pkg-config \
        libssl-dev \
        libopenblas-dev
else
    # macOS approach (brew)
    if command -v brew >/dev/null 2>&1; then
        brew install portaudio ffmpeg cmake ninja
    fi
fi

# 3. Instalacion de Dependencias Python
echo "🐍 Installing Python Dependencies..."
# Handle PEP 668 (externally-managed-environment) in Docker
PIP_ARGS=""
if [ -f /.dockerenv ]; then
    PIP_ARGS="--break-system-packages"
fi

pip install $PIP_ARGS --upgrade pip

echo "📁 Installing Core Dependencies (WP-029)..."
# These should ideally be in Dockerfile.forge, but setup_dev.sh provides self-healing
pip install $PIP_ARGS \
    duckdb>=1.0.0 \
    sqlparse \
    polars \
    pyarrow \
    pandas \
    numpy \
    pytest \
    pyyaml \
    textual \
    watchfiles

echo "🎙️ Installing Audio/STT Dependencies..."
pip install $PIP_ARGS \
    pyaudio \
    sounddevice \
    faster-whisper \
    silero-vad || echo "⚠️ Audio dependencies failed (might require local hardware access)"

echo "🧠 Installing Inference Dependencies..."
# llama-cpp-python can be tricky, try separate
pip install $PIP_ARGS llama-cpp-python || echo "⚠️ llama-cpp-python failed to build. Check builder tools."

echo "✅ HilliumOS Environment Ready!"