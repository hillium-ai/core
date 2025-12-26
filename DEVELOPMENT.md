# HilliumOS MVP v8.4 Development Guide

**Roadmap Version:** 8.4.0 "Cognitive Reasoning Upgrade"
**Last Updated:** 2025-12-16

## Quick Start with Levitate

```bash
cd /Users/jsaldana/GitLocalRepo/hillium-core
levitate
```

Then load context:
```
/file /path/to/HilliumOS/docs/MVP_v0.1/MVP_Technical_Constitution_v2.0.md
/file /path/to/HilliumOS/docs/MVP_v0.1/specs/01_CORE_MEMORY_IPC.md
```

## Version Evolution (v8.0 → v8.4)

| Version | Codename | Key Innovation |
|---------|----------|----------------|
| v8.0 | Nested Learning | 4-Level Memory + Aegis Layer 7 |
| v8.1 | Self-Improvement | Autodidactic Gym + TheTrainer |
| v8.2 | Neural Memory | Titans Memory (2M context) |
| v8.3 | Agentic Skills | Executable Skills + Sleep Cycle |
| **v8.4** | **Cognitive Reasoning** | **TheSolver + Bandit Router** |

## Current Status

- [ ] **FASE 0**: Project Scaffolding (WP-000)
- [ ] **FASE 1**: Core Memory (WP-001 to WP-007)
- [ ] **FASE 2**: PyO3 Bridge (WP-008)
- [ ] **FASE 3**: Cognitive Council (WP-009 to WP-011)
- [ ] **FASE 3B**: Aegis Layer 7 + Soft-Scoring (WP-012, WP-028, WP-030)
- [ ] **FASE 4**: Aura STT/TTS (WP-013, WP-014)
- [ ] **FASE 5**: WhoIAm + SynApps (WP-015, WP-016)
- [ ] **FASE 6**: Motor Cortex (WP-017, WP-018)
- [ ] **FASE 6+**: Autodidactic Gym (WP-019 to WP-021)
- [ ] **FASE 7**: Advanced Research (WP-022 to WP-025)
- [ ] **FASE 7-B**: Hidden Gems v8.3 (WP-026)
- [ ] **FASE 8**: Cognitive Reasoning v8.4 (WP-027 to WP-030)

**Total: 30 Work Packages**

## Project Structure

```
hillium-core/
├── Cargo.toml                    # Workspace root
├── .cargo/config.toml            # Build config
│
├── hipposerver/                  # [RUST] Sistema Nervioso
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── main.rs
│       ├── state.rs              # HippoState v8.0 (4-level memory)
│       ├── shm.rs                # Shared memory (POSIX)
│       ├── seqlock.rs            # Lock-free reads
│       ├── telemetry.rs          # CDI logging
│       └── memory/
│           ├── working_memory.rs # Level 2 (Sled DB)
│           └── associative.rs    # Level 2.5 (Fast Weights)
│
├── aegis_core/                   # [RUST] Safety (7 Layers)
│   └── src/
│       └── layer7_cognitive_safety/
│           ├── hard_constraints.rs
│           ├── core_link.rs      # Associative Core integration
│           ├── inspector_link.rs
│           └── soft_scoring.rs   # v8.4: Gradient Safety
│
├── hillium_backend/              # [RUST/PYO3] Bridge
│   ├── Cargo.toml
│   └── src/lib.rs                # HippoLink class
│
├── motor_cortex/                 # [RUST] Kinematics
│   └── src/
│       ├── kinematics.rs
│       └── trajectory.rs
│
└── loqus_core/                   # [PYTHON] Cognitive
    ├── pyproject.toml
    └── loqus_core/
        ├── memory/
        │   └── continuum.py      # 4-Level CMS
        ├── inference/
        │   └── manager.py        # Multi-model
        ├── cognition/
        │   ├── cognitive_council.py  # 5 roles v8.4
        │   ├── the_solver.py     # v8.4: Code-as-Reasoning
        │   └── bandit_router.py  # v8.4: UCB1
        ├── safety/
        │   └── cognitive_validator.py
        ├── packages/
        │   └── manager.py        # Cognitive Packages
        └── autodidactic/
            └── the_trainer.py    # v8.1: Curriculum Gen
```

## Cognitive Council v8.4 (5 Roles)

| Role | Model | Function |
|------|-------|----------|
| **Ejecutor** | Phi-3-Mini | Fast plan generation |
| **Inspector** | Qwen 2.5-7B | Plan validation |
| **Auditor** | Llama-3-8B | Meta-verification |
| **TheTrainer** | Phi-3 | Curriculum generation (v8.1) |
| **TheSolver** | Python DSL | Code-as-Reasoning (v8.4) |

## Memory Hierarchy (4 Levels)

```
┌─────────────────────────────────────────────────┐
│  L1: Sensory (SHM, <1µs)     - Audio/Video     │
│  L2: Working (Sled, <1ms)    - Conversation    │
│  L2.5: Associative (<10µs)   - Fast Weights    │
│  L3: Episodic (10-100ms)     - Titans/Qdrant   │
└─────────────────────────────────────────────────┘
```

## v8.4 Innovations

### TheSolver (WP-027)
```python
# Code-as-Reasoning: Complex logic in Python, not LLM hallucination
task → generate_python_dsl() → sandbox_execute() → self_correct()
```

### Soft-Scoring (WP-028)
```python
# Before v8.4: Binary
result = "Approved" | "Rejected"

# After v8.4: Gradient
result = SoftScore(safety=0.9, logic=0.7, efficiency=0.5)
```

### Bandit Router (WP-029)
```python
# UCB1: Upper Confidence Bound
score = mean_reward + sqrt(2 * ln(total) / count)
# Dynamic model selection: -40% token cost
```

## Build & Test

```bash
# Rust
cargo check --all
cargo build --all
cargo test --all
cargo clippy --all -- -D warnings

# Python (after PyO3 build)
cd loqus_core && poetry install && poetry run pytest
```

## Git Workflow

```bash
# Code → hillium-core (public)
git add .
git commit -m "feat(component): description [WP-XXX]"
git push origin main

# Docs → HilliumOS backup (private)
cd /path/to/HilliumOS
git push backup main
```

## Key Specs Reference

| Spec | Content | Version |
|------|---------|---------|
| SPEC-01 | Core Memory, HippoState, Sled DB | v8.0 |
| SPEC-01.5 | Associative Core, Fast Weights | v8.0 |
| SPEC-02 | Cognition, Council (5 roles), CMS | v8.4 |
| SPEC-03 | Safety, HAL | v8.0 |
| SPEC-03.5 | Layer 7, Soft-Scoring | v8.4 |
| SPEC-04 | Aura, WhoIAm, SynApps | v8.0 |
| SPEC-05 | Motor Cortex | v8.0 |

## Key ADRs

| ADR | Topic | Version |
|-----|-------|---------|
| ADR-005 | Nested Learning (4-Level Memory) | v8.0 |
| ADR-008 | Aegis Layer 7 | v8.0 |
| ADR-010 | Autodidactic Gym | v8.1 |
| ADR-012 | Titans Memory | v8.2 |
| ADR-013 | Agentic Skills | v8.3 |
| **ADR-014** | **Cognitive Reasoning** | **v8.4** |

## Success Metrics (v8.4)

| Metric | Target |
|--------|--------|
| HOR (Hallucination Rate) | <1% |
| Latency (end-to-end) | <200ms p95 |
| Fast Weights Update | <10µs p99 |
| Working Memory Query | <1ms p95 |
| Token Cost Reduction | -40% (Bandit Router) |
| Logic Precision | 100% (TheSolver) |
