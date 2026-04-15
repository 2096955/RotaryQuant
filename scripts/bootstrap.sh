#!/usr/bin/env bash
# bootstrap.sh — single-command install for RotaryQuant / IsoQuant
# Usage: bash scripts/bootstrap.sh
set -euo pipefail

echo "==> Installing isoquant-mlx CLI wrapper..."
pip install -e .

echo "==> Installing mlx-lm inference engine (with test deps)..."
cd mlx-lm && pip install -e ".[test]" && cd ..

echo "==> Verifying install..."
python -c "from isoquant_mlx import __version__; print(f'isoquant-mlx {__version__}')"
python -c "import mlx_lm; print('mlx-lm OK')"

echo ""
echo "Done. Run a smoke test with:"
echo "  python -m mlx_lm.generate --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \\"
echo "    --prompt 'Hello, world' --kv-cache-type isoquant --max-tokens 50"
