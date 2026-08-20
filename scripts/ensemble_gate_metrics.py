#!/usr/bin/env python3
"""Convenience shim so the CLI can be run as ``scripts/ensemble_gate_metrics.py``.

Adds the repository root to ``sys.path`` and delegates to
``ensemble_gate.cli.main``.  This shim intentionally imports nothing from
Isaac Lab / Isaac Sim; the ``ensemble_gate`` package is pure standard-library
Python.
"""
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ensemble_gate.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
