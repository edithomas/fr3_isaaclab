"""Pytest configuration for the ensemble_gate package tests.

Adds the repository root to ``sys.path`` so ``import ensemble_gate`` works
without installing the package (which would pull in Isaac Lab dependencies).
"""
import os
import sys

# Repository root (parent of this tests/ directory) so that
# `import ensemble_gate` resolves to the top-level package.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import json  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def tmp_jsonl(tmp_path):
    """Return a helper that writes records to a JSONL file and returns path."""

    def _make(records, name="episodes.jsonl"):
        path = tmp_path / name
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        return str(path)

    return _make


@pytest.fixture
def run_cli():
    """Return a helper that runs the CLI as a subprocess and captures output."""
    import subprocess

    def _run(*args, env=None, cwd=ROOT):
        full_env = dict(os.environ)
        if env:
            full_env.update(env)
        full_env.setdefault("PYTHONPATH", ROOT)
        # Prefer `python -m ensemble_gate` so the package path is resolved
        # via PYTHONPATH=ROOT.
        proc = subprocess.run(
            [sys.executable, "-m", "ensemble_gate", *args],
            capture_output=True,
            text=True,
            env=full_env,
            cwd=cwd,
        )
        return proc.returncode, proc.stdout, proc.stderr

    return _run
