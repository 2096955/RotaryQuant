# Contributing to RotaryQuant / IsoQuant-MLX

## Project Status

Active research project. Core architecture (IsoQuant KV compression, expert offloading, layer-aware quantization) is stable. Validation coverage is expanding across architectures and hardware profiles.

## What We Need

### High Priority

- **Testing on 32GB physical hardware.** Current results are simulated on a 128GB machine with memory caps. We need confirmation on real 32GB M2/M3/M4 MacBooks (peak RSS, thermal throttling, swap pressure).
- **1T model pathway testing.** The expert offload + IsoQuant pipeline is designed to scale, but we have no results beyond 120B parameters.
- **Additional architectures.** DeepSeek-V3/R1, Kimi-K2.5, and other recent MoE models need wiring and validation. See `mlx-lm/mlx_lm/models/` for existing model adapters.

### Medium Priority

- **llama.cpp integration hardening.** The fused Metal kernel has known numerical divergence versus the reference Python path. Needs investigation and a tolerance-aware test harness.
- **Long-context validation beyond 4K tokens.** IsoQuant KV compression is validated at 512-4K token prefill. We need quality-gate runs at 8K, 16K, and 32K to confirm compression does not degrade downstream accuracy.

### Lower Priority

- **Energy efficiency profiling.** Requires `sudo` access for `powermetrics` on macOS. We want watts-per-token measurements across quant levels.
- **GPU profiling via Xcode Instruments.** Metal System Trace captures for the fused attention kernels to identify occupancy and bandwidth bottlenecks.
- **Mojo benchmark improvements.** Current Mojo matmul kernels are at roughly 2% of roofline. Contributions that close the gap (tiling, SIMD utilization) are welcome.

## How to Contribute

1. **Fork** the repository.
2. **Run quality validation** on your hardware:
   ```bash
   python scripts/eval_quality_gate.py --model <your-model>
   ```
3. **Open an issue** describing your hardware, model, and results.
4. **Submit a PR** with pinned artifacts: timestamped result JSONs committed under `results/` so reviewers can reproduce and compare.

Please keep PRs focused. One logical change per PR.

## Peer Review

The research paper (`docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md`) and the benchmark specification are open for technical critique. If you find errors, questionable claims, or missing baselines:

- File an issue tagged `paper-review`.
- Reference the specific section and provide your counter-evidence or suggested correction.

We take reproducibility seriously. Every claim in the paper should be traceable to a script and a result JSON in this repository.
