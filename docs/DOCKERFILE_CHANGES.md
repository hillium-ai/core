# Dockerfile Changes for Forge Dependency Fix
**Date:** 2025-01-20
**Task:** Forge Build Dependencies (v18.6.1)

## Overview
Modified `infrastructure/docker/Dockerfile.forge` to include build dependencies required for compiling `llama-cpp-python` and other C++ extensions.

## Changes
### Added System Packages
The following packages were added to the `apt-get install` command:
- `ninja-build` (Required for modern CMake builds of llama.cpp)

### existing Packages Confirmed
- `build-essential`
- `cmake`
- `git`
- `libopenblas-dev`

## Pending Issues
- `llama-cpp-python` compilation still fails on ARM64/Debian 12 due to `repack.cpp` stringop-overflow warnings treated as errors or other toolchain mismatches.
- A workaround using `--no-deps` in `.levitate.yaml` is currently active to ensure `loqus_core` installs successfully.
