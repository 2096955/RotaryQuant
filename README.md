# IsoQuant: Toward Trillion-Parameter MoE Inference on Consumer Hardware

Research checkpoint -- open for peer review and contribution.

**14.85 tok/s on Nemotron-120B within a 32 GB memory budget on Apple Silicon.**

Three independent compression axes -- weight quantisation (4-bit dense, 2-bit routed experts, Q8_0 shared), IsoQuant KV compression (WHT + SO(4) rotation, 3-bit), and LRU expert offloading -- compose into a unified inference system that runs models too large for consumer hardware.

---

## Current Status

| Model | Params | tok/s | Peak Memory | Quality | Soak |
|------------|--------|-------|-------------|---------|------|
| Gemma 4 | 26B | 12.85 | 5.4 GB | 12/12 | 2 h pass |
| Nemotron-H | 120B | 14.85 | 17.2 GB | 12/12 | 2 h pass |

All measurements taken on Apple M4 Max (128 GB unified memory, 40 GPU cores, macOS 15.4). Result JSONs are pinned under `results/`.

The goal is 1T-parameter inference on 128 GB consumer hardware. The 120B result demonstrates the architecture scales.

---

## Quick Start

```bash
git clone https://github.com/2096955/RotaryQuant.git
cd RotaryQuant && pip install -e .          # isoquant CLI wrapper
cd mlx-lm && pip install -e .               # MLX inference engine
python -m mlx_lm.server --model <model> --kv-cache-type isoquant --port 8000
```

Note: this project requires an editable install from a git clone. PyPI distribution is not yet supported.

---

## Use with Claude Code

Start the local server, then point Claude Code at it:

```bash
python -m mlx_lm.server --model <model> --kv-cache-type isoquant --port 8000 &
ANTHROPIC_BASE_URL=http://localhost:8000/v1 claude code
```

---

## Repository Structure

| Directory | Contents |
|------------------|---------|
| `mlx-lm/` | MLX inference engine fork with IsoQuant KV cache + expert offload |
| `turboquant-mlx/` | KV compression library (codebooks, rotation matrices) |
| `mojo-bench/` | Mojo GPU kernel benchmarks (matmul, softmax, RoPE) |
| `scripts/` | Benchmark, comparison, validation, quality gate scripts |
| `results/` | Pinned benchmark artifacts and comparison outputs |
| `docs/` | Paper, benchmark spec, supporting documentation |
| `src/isoquant_mlx/` | CLI wrapper package (serve, validate, bench, convert) |

---

## Paper

The full technical description is in [From Attention to Consumer Hardware](docs/FROM_ATTENTION_TO_CONSUMER_HARDWARE.md). The paper covers the mathematical foundations, implementation details, benchmark methodology, and results for all three compression axes. A brief summary of the key mathematics follows.

---

## Mathematical Foundations

### The memory problem

During autoregressive generation, every past token's key and value vectors must be retained. For *L* layers, *H* heads, sequence length *T*, and head dimension *d_k*:

$$\text{KV memory} = 2 \times L \times H \times T \times d_k \times \text{bytes per element}$$

At FP16 (2 bytes), a 60-layer model with *d_k* = 128 and 8K context already demands gigabytes of KV storage alone.

### Lloyd-Max quantisation

Both TurboQuant and IsoQuant use Lloyd-Max optimal scalar quantisation. Given *b* quantisation bits, the distortion is minimised by iteratively solving:

$$D = \sum_{i=1}^{2^b} \int_{b_{i-1}}^{b_i} (x - c_i)^2 \, p(x) \, dx$$

with the nearest-neighbour condition for decision boundaries:

$$b_i = \frac{c_i + c_{i+1}}{2}$$

and the centroid condition:

$$c_i = \frac{\int_{b_{i-1}}^{b_i} x \, p(x) \, dx}{\int_{b_{i-1}}^{b_i} p(x) \, dx}$$

### IsoQuant KV compression pipeline

IsoQuant compresses keys and values through a two-stage rotation:

**Stage 1 -- Walsh-Hadamard Transform (WHT).** A global decorrelation pass that spreads information across all *d_k* dimensions. The WHT is computed in-place with *O(d_k log d_k)* operations using the butterfly factorisation.

**Stage 2 -- SO(4) block rotation.** The *d_k*-dimensional vector is partitioned into groups of 4, each rotated by paired quaternions:

$$\tilde{k}_{[i]} = \mathfrak{q}_{L,i} \otimes \hat{k}_{[i]} \otimes \bar{\mathfrak{q}}_{R,i}$$

With two independent quaternions per block, the transformation spans the full SO(4) group. The combined rotation is:

$$\tilde{k} = \Pi_{\text{SO}(4)}(H_d \cdot \hat{k})$$

where *H_d* is the normalised Walsh-Hadamard matrix. After rotation, each dimension is quantised to 3 bits using Lloyd-Max codebooks.

### Quantisation error bound

The expected squared attention-score error under IsoQuant compression is bounded by:

$$\mathbb{E}[|q^\top k - \widehat{q^\top k}|^2] \leq d_k \sigma_q^2 \|q\|_2^2$$

where *sigma_q* is the per-dimension quantisation noise variance. The key insight is that the isometric rotation preserves inner products -- rotating the query forward and computing attention in the rotated space is equivalent to computing in the original space:

$$q^\top k = (Rq)^\top (Rk)$$

This allows the inverse rotation to be applied **once** on the aggregated attention output rather than per-token, reducing decode cost from *O(T * d_k^2)* to *O(d_k^2)*.

### Amortised decode cost

The per-token decode cost for sequence length *T*:

$$\text{IsoQuant read cost} = O(T \cdot d_k) + O(d_k \log d_k)$$

$$\text{TurboQuant read cost} = O(T \cdot d_k + d_k^2)$$

IsoQuant's advantage is the constant-factor reduction of the per-query rotation cost (*d_k log d_k* vs *d_k^2*), realised through a fused 4-kernel Metal decode pipeline that operates directly on 3-bit packed data without materialising full FP16 tensors.

### Fused Metal decode pipeline

```text
[Kernel A]  fused_qk_dot:       packed 3-bit K -> unpack in-register -> centroid lookup -> dot(q,k) -> SIMD reduce -> scores
[Kernel B]  mx.softmax:         standard MLX softmax
[Kernel C]  fused_value_accum:  packed 3-bit V -> unpack in-register -> dequant -> weighted sum -> output (rotated)
[Kernel D]  metal_rotate_inverse: WHT + SO(4) structured inverse rotation (applied once on aggregated output)
```

Kernel D uses 1,408 FMAs (896 WHT butterfly + 512 SO(4) block matvecs) versus 16,384 for a dense inverse rotation.

---

## Architecture

The stack compresses three orthogonal dimensions of a large MoE model.

**Weight quantisation** reduces the static footprint: dense layers use 4-bit, routed expert weights use 2-bit, and shared parameters (embeddings, layer norms, router gates) use Q8_0.

**IsoQuant KV compression** targets the dynamic footprint that grows with context length. The WHT + SO(4) rotation pipeline followed by 3-bit Lloyd-Max codebook quantisation achieves a 5x reduction in KV cache memory with delta PPL +0.001 at 4K context.

**LRU expert offloading** keeps only the active working set of expert shards in unified memory and evicts cold experts to disk. At 120B scale, the working set is 7,544 of 20,480 total shards.

These three axes compose independently: each can be enabled or disabled without affecting the others.

---

## Gap Analysis: Proven vs Projected

| Dimension | Proven at 120B | Required for 1T | Gap |
|---------------------|----------------------------------------------|----------------------------------------------|--------------------------------------------------------------|
| Expert count | 512, topk=22 | 384, topk=8 | Different sparsity -- lower topk changes LRU dynamics |
| Working set | 7,544 of 20,480 shards | Unknown -- depends on routing entropy | Must characterise empirically |
| Memory budget | 17.2 GB of 25.6 GB target | ~110 GB of 128 GB target | Linear extrapolation holds if shard sizes scale |
| KV compression | Delta PPL +0.001 at 4K context | Same technique, longer context | Depth trend favourable but untested beyond 4K |
| Decode throughput | 14.85 tok/s | Target >5 tok/s (interactive) | Depends on expert load latency at 1T shard counts |
| Quality | 12/12 correctness harness | Must pass equivalent harness | Model-dependent, not stack-dependent |

---

## References

This project builds on and extends the following work:

- **Lloyd-Max KV quantisation (TurboQuant):** ICLR 2026. Optimal scalar quantisation applied to KV cache compression. Our baseline compressor.
- **Quaternion rotation (RotorQuant):** scrya-com, arXiv:2603.28430. Replaces the rotation step in the KV pipeline with SO(4) block quaternion rotations. IsoQuant extends this with a WHT pre-pass for global decorrelation.
- **LRU expert offloading:** Eliseev & Mazur 2023. Least-recently-used eviction of MoE expert shards to disk.
- **Mixed-precision weight quantisation:** APEX, MxMoE. Layer-aware bit allocation across dense and expert parameters.
- **Block attention residuals:** Moonshot AI, arXiv:2603.15031. Learned depth-wise attention weights for cross-layer signal compression.
- **Rate-distortion theory and Johnson-Lindenstrauss lemma:** Theoretical foundations for understanding when aggressive scalar quantisation preserves attention score ordering.

---

## Contributing

Contributions, bug reports, and peer review are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

Apache 2.0. See [LICENSE](LICENSE).
