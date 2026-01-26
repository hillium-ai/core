# Hillium Infrastructure & Forge Configuration

## 🏭 Overview
This directory contains the infrastructure-as-code (IaC) required to replicate the **Hillium Forge** environment. 

## 🏗️ Why is this here?
To ensure **reproducibility** across different development environments without requiring the full `Hillium-Forge` repository. Any contributor or CI/CD agent can use these files to spin up a "healthy" laboratory for:
- Compiling Rust crates (`aegis_core`, `hipposerver`, etc.)
- Running Python-based cognitive safety tests.
- Simulating Edge-device hardware constraints (ARM64/Ubuntu 22.04).

## 📄 Key Files
- `Dockerfile.forge`: A synchronized snapshot of the master build engine. 
  - **Base:** Ubuntu 22.04
  - **Stack:** Python 3.11 + Rust 1.82 + Local AI Acceleration.
  - **Permissions:** Configured for UID/GID parity (defaulting to 501:20).

## 🔄 Maintenance Note
The `Dockerfile.forge` is a **downstream copy** of the master repository located at `jsaldana/Hillium-Forge`. Any structural change to the build environment should be verified there first and then synchronized here to maintain the "Forge-as-a-Service" standard.

---
*Aura Intelligence System - Infrastructure Layer*
