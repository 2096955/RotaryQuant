"""Smoke test for isoquant_mlx package."""

import subprocess
import sys


def test_version():
    from isoquant_mlx import __version__

    assert __version__ == "0.1.0a1"


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "isoquant_mlx.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "isoquant" in result.stdout.lower()
