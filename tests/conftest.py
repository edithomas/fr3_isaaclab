"""Pytest configuration for the FR3 experimental tooling test suite.

Two test categories:

* **static** (default, run locally on macOS / CI without a GPU): exercise the
  pure-logic helpers under ``fr3_isaaclab_project.diagnostics`` and the
  non-Isaac-Sim parts of the scripts. These are the regression tests for the
  load-bearing behaviours.

* **runtime** (marker ``@pytest.mark.runtime``): launch a real Isaac Sim
  ``SimulationApp``. They are skipped automatically unless
  ``RUN_FR3_RUNTIME_TESTS=1`` is set, because they require the remote
  one-L40 container (Isaac Sim 4.5 / Isaac Lab v2.1.0) and cannot run on macOS.
"""
from __future__ import annotations

import json
import os
import sys

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list):
    """Skip runtime tests unless explicitly opted in."""
    if os.environ.get("RUN_FR3_RUNTIME_TESTS") == "1":
        return
    skip_runtime = pytest.mark.skip(
        reason="runtime test: needs Isaac Sim container; set RUN_FR3_RUNTIME_TESTS=1 to run"
    )
    for item in items:
        if "runtime" in item.keywords:
            item.add_marker(skip_runtime)


def pytest_configure(config: pytest.Config):
    config.addinivalue_line("markers", "runtime: needs a live Isaac Sim container")


# --------------------------------------------------------------------------- #
# ensemble_gate test support
# --------------------------------------------------------------------------- #
# Make the top-level ``ensemble_gate`` package importable without installing
# the project (which would pull in Isaac Lab dependencies). The repository root
# is the parent of this tests/ directory.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@pytest.fixture
def tmp_jsonl(tmp_path):
    """Write records to a JSONL file and return its path."""

    def _make(records, name="episodes.jsonl"):
        path = tmp_path / name
        with open(path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        return str(path)

    return _make


@pytest.fixture
def run_cli():
    """Run the ensemble_gate CLI as a subprocess; return (rc, stdout, stderr)."""
    import subprocess

    def _run(*args, env=None, cwd=_ROOT):
        full_env = dict(os.environ)
        if env:
            full_env.update(env)
        full_env.setdefault("PYTHONPATH", _ROOT)
        proc = subprocess.run(
            [sys.executable, "-m", "ensemble_gate", *args],
            capture_output=True,
            text=True,
            env=full_env,
            cwd=cwd,
        )
        return proc.returncode, proc.stdout, proc.stderr

    return _run
