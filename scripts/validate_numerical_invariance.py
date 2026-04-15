#!/usr/bin/env python3
"""
Phase 0: IsoQuant Numerical Invariance Test Harness

Validates that the WHT + SO(4) rotation pipeline preserves mathematical
invariants required for correct attention computation. Runs on synthetic
data with numpy only -- no model or MLX dependency needed.

Tested properties:
  1. Isometry  (norm and inner-product preservation)
  2. Cosine-similarity preservation
  3. Top-k attention index preservation
  4. Round-trip reconstruction  (R^{-1} R x == x)
  5. Quantisation error bound   (3-bit Lloyd-Max MSE)

Usage:
    python scripts/validate_numerical_invariance.py --trials 1000 \
        --json-out results/phase0_invariance.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
D_K = 128  # head dimension
SO4_BLOCK = 4  # SO(4) block size
TOP_K = 5  # for top-k preservation test
N_KEYS = 64  # number of keys in the top-k test
LLOYD_MAX_BITS = 3  # 3-bit quantisation -> 8 centroids
LLOYD_MAX_ITERS = 50  # iterations for Lloyd-Max convergence


# ---------------------------------------------------------------------------
# WHT -- recursive butterfly, normalised
# ---------------------------------------------------------------------------
def wht(x: np.ndarray) -> np.ndarray:
    """Walsh-Hadamard Transform via recursive butterfly factorisation.

    Operates on the last axis of *x* so that batched transforms work
    transparently.  The transform is unitary (orthonormal) by construction.
    """
    n = x.shape[-1]
    if n == 1:
        return x.copy()
    even = wht(x[..., ::2])
    odd = wht(x[..., 1::2])
    return np.concatenate([even + odd, even - odd], axis=-1) / np.sqrt(2)


def iwht(x: np.ndarray) -> np.ndarray:
    """Inverse WHT.  Because the normalised WHT is its own inverse
    (it is a symmetric orthogonal matrix), iwht == wht."""
    return wht(x)


# ---------------------------------------------------------------------------
# SO(4) block rotation from unit quaternion pair (q_L, q_R)
# ---------------------------------------------------------------------------
def _quaternion_to_rotation_pair(q_l: np.ndarray, q_r: np.ndarray) -> np.ndarray:
    """Build a 4x4 SO(4) matrix from two unit quaternions (q_L, q_R).

    The rotation acts on a 4-vector v as  R v  where R is decomposed via
    the double cover  SO(4) ~ (SU(2) x SU(2)) / Z_2.

    We use the explicit construction:
        R_{ij} = Tr( sigma_i  q_L  sigma_j  q_R^{-1} ) / 2
    but for efficiency we build the left- and right-multiplication matrices
    and compose them.
    """
    # Ensure unit norm
    q_l = q_l / np.linalg.norm(q_l)
    q_r = q_r / np.linalg.norm(q_r)

    # Left multiplication matrix  L(q):  q * p  represented as L(q) @ p
    def _left_matrix(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        return np.array(
            [
                [w, -x, -y, -z],
                [x, w, -z, y],
                [y, z, w, -x],
                [z, -y, x, w],
            ]
        )

    # Right multiplication matrix  R(q):  p * q  represented as R(q) @ p
    def _right_matrix(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        return np.array(
            [
                [w, -x, -y, -z],
                [x, w, z, -y],
                [y, -z, w, x],
                [z, y, -x, w],
            ]
        )

    # SO(4) = L(q_L) @ R(q_R)^T   (right-mult by conjugate of q_R)
    q_r_conj = q_r.copy()
    q_r_conj[1:] = -q_r_conj[1:]
    return _left_matrix(q_l) @ _right_matrix(q_r_conj)


def random_so4_matrix(rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly random SO(4) matrix via two random unit quaternions."""
    q_l = rng.standard_normal(4)
    q_l /= np.linalg.norm(q_l)
    q_r = rng.standard_normal(4)
    q_r /= np.linalg.norm(q_r)
    return _quaternion_to_rotation_pair(q_l, q_r)


def build_block_diagonal_rotation(
    d: int, block: int, rng: np.random.Generator
) -> np.ndarray:
    """Build a (d x d) block-diagonal orthogonal matrix from SO(4) blocks."""
    assert d % block == 0, f"d={d} must be divisible by block={block}"
    n_blocks = d // block
    R = np.zeros((d, d), dtype=np.float64)
    for i in range(n_blocks):
        start = i * block
        end = start + block
        R[start:end, start:end] = random_so4_matrix(rng)
    return R


# ---------------------------------------------------------------------------
# Full rotation pipeline:  x  ->  WHT(x)  ->  block-SO(4)
# ---------------------------------------------------------------------------
@dataclass
class RotationPipeline:
    """Encapsulates a fixed WHT + SO(4) block rotation for a given d_k."""

    d_k: int
    block_size: int
    R_so4: np.ndarray  # (d_k, d_k) block-diagonal SO(4)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply WHT then block-SO(4) rotation."""
        h = wht(x)
        return h @ self.R_so4.T  # matrix-vector or matrix-matrix

    def inverse(self, y: np.ndarray) -> np.ndarray:
        """Inverse: SO(4)^T then inverse WHT."""
        h = y @ self.R_so4  # R^T is R_so4 since R is orthogonal
        return iwht(h)


def make_pipeline(
    d_k: int, block_size: int, rng: np.random.Generator
) -> RotationPipeline:
    R = build_block_diagonal_rotation(d_k, block_size, rng)
    return RotationPipeline(d_k=d_k, block_size=block_size, R_so4=R)


# ---------------------------------------------------------------------------
# Lloyd-Max quantiser  (3-bit, 8 centroids)
# ---------------------------------------------------------------------------
def lloyd_max_quantise(
    data: np.ndarray,
    n_centroids: int = 2**LLOYD_MAX_BITS,
    n_iters: int = LLOYD_MAX_ITERS,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run iterative Lloyd-Max on *data* (1-D).

    Returns (quantised_data, centroids, assignments).
    """
    flat = data.ravel().astype(np.float64)

    # Initialise centroids via equally-spaced percentiles
    percentiles = np.linspace(0, 100, n_centroids + 2)[1:-1]
    centroids = np.percentile(flat, percentiles)

    for _ in range(n_iters):
        # Assign each point to nearest centroid
        dists = np.abs(flat[:, None] - centroids[None, :])  # (N, C)
        assignments = np.argmin(dists, axis=1)

        # Update centroids
        new_centroids = np.empty_like(centroids)
        for c in range(n_centroids):
            mask = assignments == c
            if mask.any():
                new_centroids[c] = flat[mask].mean()
            else:
                new_centroids[c] = centroids[c]

        if np.allclose(centroids, new_centroids, atol=1e-12):
            centroids = new_centroids
            break
        centroids = new_centroids

    # Final assignment + reconstruction
    dists = np.abs(flat[:, None] - centroids[None, :])
    assignments = np.argmin(dists, axis=1)
    quantised = centroids[assignments].reshape(data.shape)
    return quantised, centroids, assignments


# ---------------------------------------------------------------------------
# Test result bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class TestResult:
    name: str
    passed: bool
    metric_name: str
    metric_value: float
    threshold: float
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Test 1: Isometry (norm + inner-product preservation)
# ---------------------------------------------------------------------------
def test_isometry(
    pipe: RotationPipeline, n_trials: int, rng: np.random.Generator
) -> TestResult:
    """Verify ||Rx|| == ||x||  and  <Rx, Ry> == <x, y>  within tolerance."""
    norm_errors = np.empty(n_trials)
    ip_errors = np.empty(n_trials)

    for t in range(n_trials):
        x = rng.standard_normal(pipe.d_k)
        y = rng.standard_normal(pipe.d_k)

        rx = pipe.forward(x)
        ry = pipe.forward(y)

        # Norm preservation
        norm_x = np.linalg.norm(x)
        norm_rx = np.linalg.norm(rx)
        norm_errors[t] = abs(norm_rx - norm_x) / max(norm_x, 1e-15)

        # Inner product preservation
        ip_orig = np.dot(x, y)
        ip_rot = np.dot(rx, ry)
        ip_errors[t] = abs(ip_rot - ip_orig) / max(abs(ip_orig), 1e-15)

    max_norm_err = float(np.max(norm_errors))
    max_ip_err = float(np.max(ip_errors))
    mean_norm_err = float(np.mean(norm_errors))
    mean_ip_err = float(np.mean(ip_errors))

    threshold = 1e-10
    passed = max_norm_err < threshold and max_ip_err < threshold

    return TestResult(
        name="isometry",
        passed=passed,
        metric_name="max_relative_error",
        metric_value=max(max_norm_err, max_ip_err),
        threshold=threshold,
        details={
            "max_norm_error": max_norm_err,
            "mean_norm_error": mean_norm_err,
            "max_inner_product_error": max_ip_err,
            "mean_inner_product_error": mean_ip_err,
            "n_trials": n_trials,
        },
    )


# ---------------------------------------------------------------------------
# Test 2: Cosine similarity preservation
# ---------------------------------------------------------------------------
def test_cosine_similarity(
    pipe: RotationPipeline, n_trials: int, rng: np.random.Generator
) -> TestResult:
    """Verify cos(Q, K) == cos(RQ, RK) within tolerance."""
    errors = np.empty(n_trials)

    for t in range(n_trials):
        q = rng.standard_normal(pipe.d_k)
        k = rng.standard_normal(pipe.d_k)

        cos_orig = np.dot(q, k) / (np.linalg.norm(q) * np.linalg.norm(k))

        rq = pipe.forward(q)
        rk = pipe.forward(k)
        cos_rot = np.dot(rq, rk) / (np.linalg.norm(rq) * np.linalg.norm(rk))

        errors[t] = abs(cos_rot - cos_orig)

    max_err = float(np.max(errors))
    mean_err = float(np.mean(errors))
    threshold = 1e-10

    return TestResult(
        name="cosine_similarity_preservation",
        passed=max_err < threshold,
        metric_name="max_absolute_error",
        metric_value=max_err,
        threshold=threshold,
        details={
            "max_error": max_err,
            "mean_error": mean_err,
            "n_trials": n_trials,
        },
    )


# ---------------------------------------------------------------------------
# Test 3: Top-k attention index preservation
# ---------------------------------------------------------------------------
def test_topk_preservation(
    pipe: RotationPipeline,
    n_trials: int,
    rng: np.random.Generator,
    n_keys: int = N_KEYS,
    top_k: int = TOP_K,
) -> TestResult:
    """Generate Q and N keys, check that top-k attention indices are
    identical before and after rotating all keys (and query)."""
    mismatches = 0
    score_diffs = []

    for _ in range(n_trials):
        q = rng.standard_normal(pipe.d_k)
        K = rng.standard_normal((n_keys, pipe.d_k))
        scale = 1.0 / np.sqrt(pipe.d_k)

        # Original scores
        scores_orig = (K @ q) * scale
        topk_orig = set(np.argsort(scores_orig)[-top_k:])

        # Rotated scores -- rotate both Q and K to stay in the same basis
        rq = pipe.forward(q)
        RK = np.array([pipe.forward(k) for k in K])
        scores_rot = (RK @ rq) * scale
        topk_rot = set(np.argsort(scores_rot)[-top_k:])

        if topk_orig != topk_rot:
            mismatches += 1

        # Track max score difference
        score_diffs.append(float(np.max(np.abs(scores_orig - scores_rot))))

    mismatch_rate = mismatches / n_trials
    max_score_diff = float(np.max(score_diffs))
    threshold = 0.0  # exact match expected for isometric rotation

    return TestResult(
        name="topk_preservation",
        passed=mismatch_rate <= threshold,
        metric_name="mismatch_rate",
        metric_value=mismatch_rate,
        threshold=threshold,
        details={
            "mismatches": mismatches,
            "n_trials": n_trials,
            "n_keys": n_keys,
            "top_k": top_k,
            "max_score_diff": max_score_diff,
            "mean_score_diff": float(np.mean(score_diffs)),
        },
    )


# ---------------------------------------------------------------------------
# Test 4: Round-trip reconstruction
# ---------------------------------------------------------------------------
def test_roundtrip(
    pipe: RotationPipeline, n_trials: int, rng: np.random.Generator
) -> TestResult:
    """Apply rotation then inverse, verify x_reconstructed approx x_original."""
    errors = np.empty(n_trials)

    for t in range(n_trials):
        x = rng.standard_normal(pipe.d_k)
        rx = pipe.forward(x)
        x_recon = pipe.inverse(rx)
        errors[t] = np.linalg.norm(x_recon - x) / max(np.linalg.norm(x), 1e-15)

    max_err = float(np.max(errors))
    mean_err = float(np.mean(errors))
    threshold = 1e-10

    return TestResult(
        name="roundtrip_reconstruction",
        passed=max_err < threshold,
        metric_name="max_relative_reconstruction_error",
        metric_value=max_err,
        threshold=threshold,
        details={
            "max_error": max_err,
            "mean_error": mean_err,
            "n_trials": n_trials,
        },
    )


# ---------------------------------------------------------------------------
# Test 5: Quantisation error bound (3-bit Lloyd-Max)
# ---------------------------------------------------------------------------
def test_quantisation_error_bound(
    pipe: RotationPipeline,
    n_trials: int,
    rng: np.random.Generator,
) -> TestResult:
    """Apply rotation, quantise to 3-bit via Lloyd-Max, measure MSE.

    Theoretical bound for Lloyd-Max with 2^b centroids on a Gaussian source:
        MSE_LM(b) ~ sigma^2 * c(b)
    where c(3) ~ 0.0344  (from rate-distortion theory).

    After WHT + SO(4) rotation of i.i.d. Gaussian data the marginal
    distribution remains Gaussian with approximately the same variance,
    so the same bound applies.  We use a generous 2x safety factor.

    We also compare the MSE of rotated+quantised vs unrotated+quantised
    to verify that rotation does not inflate quantisation error.
    """
    n_centroids = 2**LLOYD_MAX_BITS  # 8
    # Theoretical MSE fraction for 3-bit Lloyd-Max on unit-variance Gaussian
    # (from optimal Lloyd-Max tables: ~3.44% of variance)
    lloyd_max_fraction_3bit = 0.0344
    safety_factor = 2.5  # generous margin for finite samples

    mse_rotated_list = []
    mse_unrotated_list = []
    variance_list = []

    for _ in range(n_trials):
        x = rng.standard_normal(pipe.d_k)
        variance = float(np.var(x))
        variance_list.append(variance)

        # Quantise original (unrotated)
        q_orig, _, _ = lloyd_max_quantise(x, n_centroids=n_centroids)
        mse_unrotated = float(np.mean((x - q_orig) ** 2))
        mse_unrotated_list.append(mse_unrotated)

        # Rotate then quantise
        rx = pipe.forward(x)
        q_rot, _, _ = lloyd_max_quantise(rx, n_centroids=n_centroids)
        mse_rotated = float(np.mean((rx - q_rot) ** 2))
        mse_rotated_list.append(mse_rotated)

    mean_mse_rotated = float(np.mean(mse_rotated_list))
    mean_mse_unrotated = float(np.mean(mse_unrotated_list))
    mean_variance = float(np.mean(variance_list))

    # Theoretical upper bound
    theoretical_bound = mean_variance * lloyd_max_fraction_3bit * safety_factor

    # Two checks:
    #  (a) MSE of rotated data is within theoretical bound
    #  (b) MSE of rotated data is not significantly worse than unrotated
    bound_ok = mean_mse_rotated < theoretical_bound
    # Allow 20% relative inflation at most
    inflation_ratio = mean_mse_rotated / max(mean_mse_unrotated, 1e-15)
    inflation_ok = inflation_ratio < 1.20

    passed = bound_ok and inflation_ok

    return TestResult(
        name="quantisation_error_bound",
        passed=passed,
        metric_name="mean_mse_rotated",
        metric_value=mean_mse_rotated,
        threshold=theoretical_bound,
        details={
            "mean_mse_rotated": mean_mse_rotated,
            "mean_mse_unrotated": mean_mse_unrotated,
            "mean_variance": mean_variance,
            "theoretical_bound": theoretical_bound,
            "inflation_ratio": inflation_ratio,
            "inflation_ok": inflation_ok,
            "bound_ok": bound_ok,
            "lloyd_max_fraction_3bit": lloyd_max_fraction_3bit,
            "safety_factor": safety_factor,
            "n_centroids": n_centroids,
            "n_trials": n_trials,
        },
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_all_tests(n_trials: int, seed: int = 42) -> list[TestResult]:
    rng = np.random.default_rng(seed)
    pipe = make_pipeline(D_K, SO4_BLOCK, rng)

    print("IsoQuant Phase 0 Numerical Invariance Tests")
    print(f"  d_k={D_K}  SO(4) block={SO4_BLOCK}  trials={n_trials}  seed={seed}")
    print(f"  WHT size={D_K}  top_k={TOP_K}  n_keys={N_KEYS}")
    print("=" * 72)

    tests = [
        ("1. Isometry check", lambda: test_isometry(pipe, n_trials, rng)),
        (
            "2. Cosine similarity preservation",
            lambda: test_cosine_similarity(pipe, n_trials, rng),
        ),
        (
            "3. Top-k attention preservation",
            lambda: test_topk_preservation(pipe, n_trials, rng),
        ),
        ("4. Round-trip reconstruction", lambda: test_roundtrip(pipe, n_trials, rng)),
        (
            "5. Quantisation error bound (3-bit)",
            lambda: test_quantisation_error_bound(pipe, min(n_trials, 200), rng),
        ),
    ]

    results: list[TestResult] = []

    for label, fn in tests:
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0

        status = "PASS" if result.passed else "FAIL"
        print(f"\n{label}")
        print(
            f"  [{status}]  {result.metric_name} = {result.metric_value:.6e}  "
            f"(threshold: {result.threshold:.6e})  [{elapsed:.3f}s]"
        )

        # Print relevant detail lines
        for k, v in result.details.items():
            if isinstance(v, float):
                print(f"         {k}: {v:.6e}")
            else:
                print(f"         {k}: {v}")

        results.append(result)

    print("\n" + "=" * 72)
    n_passed = sum(1 for r in results if r.passed)
    n_total = len(results)
    overall = (
        "ALL PASSED"
        if n_passed == n_total
        else f"FAILURES: {n_total - n_passed}/{n_total}"
    )
    print(f"Summary: {n_passed}/{n_total} tests passed  --  {overall}")
    print("=" * 72)

    return results


def results_to_json(results: list[TestResult], n_trials: int, seed: int) -> dict:
    """Build a JSON-serialisable summary."""
    test_entries = []
    for r in results:
        test_entries.append(
            {
                "name": r.name,
                "passed": r.passed,
                "metric_name": r.metric_name,
                "metric_value": r.metric_value,
                "threshold": r.threshold,
                "details": r.details,
            }
        )

    return {
        "harness": "IsoQuant Phase 0 Numerical Invariance",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "parameters": {
            "d_k": D_K,
            "so4_block_size": SO4_BLOCK,
            "n_trials": n_trials,
            "seed": seed,
            "top_k": TOP_K,
            "n_keys": N_KEYS,
            "lloyd_max_bits": LLOYD_MAX_BITS,
        },
        "tests": test_entries,
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "all_passed": all(r.passed for r in results),
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="IsoQuant Phase 0: validate WHT + SO(4) numerical invariants",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1000,
        help="Number of random trials per test (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Path to write JSON results (e.g. results/phase0_invariance.json)",
    )
    args = parser.parse_args()

    results = run_all_tests(n_trials=args.trials, seed=args.seed)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = results_to_json(results, args.trials, args.seed)
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nJSON results written to: {out_path}")

    # Exit code reflects test outcome
    if not all(r.passed for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
