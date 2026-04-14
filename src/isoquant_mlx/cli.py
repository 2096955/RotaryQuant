"""CLI entry point for isoquant-mlx.

Delegates to mlx-lm and project scripts via subprocess.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from isoquant_mlx import __version__

# Resolve repo root: cli.py -> isoquant_mlx/ -> src/ -> repo_root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _run(cmd: list[str]) -> int:
    """Run a subprocess, forwarding stdio. Returns the exit code."""
    result = subprocess.run(cmd)
    return result.returncode


def cmd_serve(args: argparse.Namespace) -> int:
    """Start an MLX-LM server with IsoQuant KV compression."""
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.server",
        "--model",
        args.model,
        "--port",
        str(args.port),
        "--kv-cache-type",
        "isoquant",
    ]
    if args.expert_offload:
        cmd.append("--expert-offload")
    return _run(cmd)


def cmd_validate(args: argparse.Namespace) -> int:
    """Run the quality gate validation suite."""
    script = SCRIPTS_DIR / "eval_quality_gate.py"
    if not script.exists():
        print(
            f"Error: validation script not found at {script}\n"
            f"Expected scripts/ directory at repo root: {REPO_ROOT}",
            file=sys.stderr,
        )
        return 1
    cmd = [sys.executable, str(script), "--model", args.model]
    return _run(cmd)


def cmd_bench(args: argparse.Namespace) -> int:
    """Run the MoE offload benchmark."""
    script = SCRIPTS_DIR / "benchmark_moe_offload.py"
    if not script.exists():
        print(
            f"Error: benchmark script not found at {script}\n"
            f"Expected scripts/ directory at repo root: {REPO_ROOT}",
            file=sys.stderr,
        )
        return 1
    cmd = [
        sys.executable,
        str(script),
        "--model",
        args.model,
        "--profile",
        args.profile,
    ]
    return _run(cmd)


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert a model with IsoQuant KV quantization."""
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.convert",
        "--model",
        args.model,
        "--kv-bits",
        str(args.kv_bits),
    ]
    if args.output:
        cmd.extend(["--output", args.output])
    return _run(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="isoquant",
        description="IsoQuant: KV compression + expert offloading for Apple Silicon LLM inference",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"isoquant-mlx {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # serve
    p_serve = subparsers.add_parser(
        "serve", help="Start an MLX-LM server with IsoQuant KV compression"
    )
    p_serve.add_argument("model", help="HuggingFace model ID or local path")
    p_serve.add_argument(
        "--port", type=int, default=8000, help="Server port (default: 8000)"
    )
    p_serve.add_argument(
        "--expert-offload",
        action="store_true",
        help="Enable expert offloading for MoE models",
    )
    p_serve.set_defaults(func=cmd_serve)

    # validate
    p_validate = subparsers.add_parser("validate", help="Run quality gate validation")
    p_validate.add_argument("model", help="HuggingFace model ID or local path")
    p_validate.set_defaults(func=cmd_validate)

    # bench
    p_bench = subparsers.add_parser("bench", help="Run MoE offload benchmark")
    p_bench.add_argument("model", help="HuggingFace model ID or local path")
    p_bench.add_argument(
        "--profile", default="B", help="Benchmark profile (default: B)"
    )
    p_bench.set_defaults(func=cmd_bench)

    # convert
    p_convert = subparsers.add_parser(
        "convert", help="Convert model with IsoQuant KV quantization"
    )
    p_convert.add_argument("model", help="HuggingFace model ID or local path")
    p_convert.add_argument(
        "--kv-bits", type=int, default=3, help="KV cache quantization bits (default: 3)"
    )
    p_convert.add_argument("--output", default=None, help="Output directory")
    p_convert.set_defaults(func=cmd_convert)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    rc = args.func(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
