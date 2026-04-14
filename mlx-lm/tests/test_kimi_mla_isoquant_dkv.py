"""Sanity checks for Kimi MLA DKV placeholder constants (no weights required)."""

from mlx_lm.models.kimi_mla_isoquant_dkv import (
    DKV_CONTENT_DIM,
    DKV_ROPE_DIM,
    MLA_LATENT_DIM,
    content_rope_split,
)


def test_dkv_partition():
    assert DKV_CONTENT_DIM + DKV_ROPE_DIM == MLA_LATENT_DIM


def test_content_rope_split():
    c, r = content_rope_split(512, 448)
    assert (c, r) == (448, 64)
