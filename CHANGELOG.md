# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Identity guide (RotaryQuant vs IsoQuant vs TurboQuant) in README
- MLA/DKV sub-block split constraint documentation (hard blocker for Kimi path)
- AttnRes predictor status and decision rationale
- Bootstrap install script (`scripts/bootstrap.sh`)
- 5-minute smoke test section in README
- Deployment & stability section with risk gates
- CHANGELOG.md

### Fixed
- Mermaid diagram parse errors (escaped parentheses in node labels)
- Symbol reference table rendering (`\mathfrak` replaced with plain notation)

## [0.1.0-alpha.1] — 2026-04-14

Initial public research checkpoint.

### Added
- IsoQuant (WHT + SO(4)) KV cache compression pipeline
- Fused Metal decode pipeline (4 kernels: fused_qk_dot, softmax, fused_value_accum, metal_rotate_inverse)
- Expert offloading with LRU eviction and `ensure_loaded()`
- Mixed-precision weight quantisation (4-bit dense, 2-bit routed experts, Q8_0 shared)
- Deferred prefill with bulk compression
- llama.cpp integration as `GGML_TYPE_ISOQUANT3_0` with fused Metal shader
- Mojo kernel benchmarks (matmul, softmax, RoPE)
- CLI wrapper (`isoquant-mlx`) with serve, validate, bench, convert subcommands
- Quality gate script (12-prompt automated pass/fail)
- 2-hour soak test automation
- Yum Cha conceptual primer with AI-generated illustrations
- Full technical paper (FROM_ATTENTION_TO_CONSUMER_HARDWARE.md)
- NotebookLM podcast and video companion media

### Validated
- Gemma 4-26B: 12.85 tok/s, 5.4 GB, 12/12 quality, 2h soak pass
- Nemotron-H 120B: 14.85 tok/s, 17.2 GB, 12/12 quality, 2h soak pass
- KV fidelity: delta PPL +0.0000 (Gemma 4), +0.0009 (Qwen3), +0.0012 (Nemotron)
- llama.cpp fused kernel: -0.5% prompt, -3.2% gen vs turbo3 baseline
