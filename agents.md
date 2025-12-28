# Levitate Agents Configuration
# Place this file in your project root (`agents.md`) or in your user config directory
# (Linux: `~/.config/levitate/agents.md`, macOS: `~/Library/Application Support/levitate/agents.md`).

agents:
  # Default agent used if none specified
  default:
    model: "devstral-small-2:latest"
    temperature: 0.7
    system_prompt: |
      You are Levitate, a powerful AI coding assistant with full tool access.

      ## PROJECT CONTEXT (hillium-core)
      This repository is the PUBLIC MVP codebase. All writes must stay here.
      Docs and Work Packages live in /Users/jsaldana/GitLocalRepo/HilliumOS (read-only).
      Private roadmap and internal docs live in /Users/jsaldana/GitLocalRepo/HilliumOS-Private (read-only, do not quote or copy to public outputs).
      
      ## PROJECT STRUCTURE MAP (hillium-core)
      This is a Rust Workspace. Use the defined workspace members:
      - hipposerver/     (Main Server Node)
      - crates/          (Shared Libraries & Modules)
        - working_memory/
      - loqus_core/      (Python Logic Core)
      - motor_cortex/    (Hardware/Robot Interface)
      - hillium_backend/ (Integration Layer)
      
      DO NOT create new root folders outside of these known workspace members.

      
      Do not `cd` outside this repo unless the user explicitly asks.

      Be concise but thorough. Use tools proactively. Always use TDD (test first).
    tools: ["*"] # All tools allowed
    moe:
      enabled: true
      strategy: "critic_refine"
      experts:
        generator1: "qwen3-coder:30b"
        generator2: "deepseek-coder-v2:16b"
        critic: "deepseek-coder-v2:16b"
        refiner: "devstral-small-2:latest"
      # Context optimization to fit 96GB RAM (Deeptree defaults to huge ctx >29GB VRAM)
      expert_options:
        num_ctx: 16384

  # Specialized "Senior Developer"
  senior_dev:
    model: "qwen2.5-coder:14b"
    temperature: 0.2
    system_prompt: |
      You are a Senior Software Engineer.
      Prioritize clean, maintainable code.
      Always write tests for new functionality.
    tools: ["read_file", "write_file", "run_command", "git_*"]
    moe:
      enabled: true
      strategy: "critic_refine"
      experts:
        generator: "qwen2.5-coder:14b"
        critic: "llama3:8b"
        refiner: "qwen2.5-coder:14b"

  # Security Auditor (Read-only)
  auditor:
    model: "llama3"
    system_prompt: "Analyze the code for security vulnerabilities. OWASP Top 10."
    tools: ["read_file", "list_dir"] # No write access!
    moe:
      enabled: false

global_settings:
  theme: "hillium" # hillium (yellow/black), dracula, monokai
  approvals: "auto" # AUTO mode: tools execute without prompts. Change to "safe" for manual approval.
  trace_logging: true # Enables trace panels in the CLI. (Not telemetry by itself.)
  mcp_url: "http://localhost:3000"
  # Create a key in the MCP-Agentic dashboard and paste it here (used for /obs endpoints).
  # Authorization: Bearer <api_key>
  mcp_agentic_api_key: ""
  # Ollama base URL (local or remote). Examples:
  # - Local:  http://localhost:11434
  # - LAN:    http://ollama.hillium.ai:11434
  ollama_url: "http://ollama.hillium.ai:11434"
  # Optional: emit Levitate events into MCP-Agentic ULS via POST /api/observability/emit
  telemetry_enabled: false
  telemetry_include_user_prompts: false
  telemetry_include_tool_output: false
  local_logging: true
  log_level: "INFO"
  context_auto_summarize: true
  context_summarize_at_pct: 85
  context_keep_last_messages: 24
  # Activity phrases shown while Levitate is working.
  # `activity_phrases` overrides the pack if provided.
  activity_pack: "hillium"
  activity_interval_s: 0.6
  activity_phrases: []
  # Startup intro animation
  intro_enabled: true
  intro_duration_s: 6.0
