#!/bin/bash

# Detect platform
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Detected macOS"
    # Install Rust
    if ! command -v rustup &> /dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source $HOME/.cargo/env
    fi
    
    # Install Python dependencies
    if ! command -v python3 &> /dev/null; then
        echo "Python3 not found. Please install Python 3.10 or higher."
        exit 1
    fi
    
    python3 -m pip install --upgrade pip
    python3 -m pip install numpy pydantic langchain
    
    echo "macOS setup completed."
elif [[ "$(uname)" == "Linux" && "$(arch)" == "aarch64" ]]; then
    echo "Detected Linux ARM64"
    
    # Install cross-compilation toolchain
    if ! command -v gcc-aarch64-linux-gnu &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y gcc-aarch64-linux-gnu
    fi
    
    # Install Rust
    if ! command -v rustup &> /dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source $HOME/.cargo/env
    fi
    
    # Install Python dependencies
    if ! command -v python3 &> /dev/null; then
        echo "Python3 not found. Please install Python 3.10 or higher."
        exit 1
    fi
    
    python3 -m pip install --upgrade pip
    python3 -m pip install numpy pydantic langchain
    
    echo "Linux ARM64 setup completed."
else
    echo "Unsupported platform"
    exit 1
fi