# Copyright © 2026 Apple Inc.
"""Kimi K2.5 MLA + IsoQuant — DKV split placeholders (implement when MLX checkpoint lands).

Per roadmap Appendix H: apply IsoQuant only to **content** dimensions of the MLA latent,
never to RoPE dimensions (``content smearing`` destroys long-context position awareness).

These constants must be verified against ``config.json`` on the real checkpoint
(``kv_lora_rank``, ``qk_rope_head_dim``, etc.).
"""

from __future__ import annotations

# Typical K2.5-style split — **verify** before enabling compression.
MLA_LATENT_DIM: int = 512
DKV_CONTENT_DIM: int = 448
DKV_ROPE_DIM: int = 64

assert DKV_CONTENT_DIM + DKV_ROPE_DIM == MLA_LATENT_DIM, (
    "DKV split must partition MLA latent"
)


def content_rope_split(latent_dim: int, content_dim: int) -> tuple[int, int]:
    """Return (content_dims, rope_dims) for sanity checks."""
    rope = latent_dim - content_dim
    if rope < 0:
        raise ValueError("content_dim exceeds latent_dim")
    return content_dim, rope
