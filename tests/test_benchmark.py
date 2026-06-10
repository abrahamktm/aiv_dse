"""Smoke test for the empirical benchmark script.

The full benchmark is too slow for CI (runs hundreds of synthesis iterations).
This test runs a tiny version end-to-end to confirm the script wires up
correctly and produces the expected output shape.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_benchmark_runs_to_completion():
    """Tiny benchmark: 1 run per strategy, 3 iters each, just verify it doesn't crash."""
    # Inherit the parent environment so the subprocess can find installed
    # site-packages (pydantic, optuna, etc.), then layer PYTHONPATH on top.
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    result = subprocess.run(
        [sys.executable, "scripts/benchmark.py", "--runs", "1", "--max-iters", "3"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"benchmark failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    # The summary table should include all three strategies and the header
    assert "Strategy" in result.stdout
    assert "shadow" in result.stdout
    assert "bayesian" in result.stdout
    assert "llm" in result.stdout
