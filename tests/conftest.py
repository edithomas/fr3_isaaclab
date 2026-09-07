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

import os

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
