"""Stub ``isaaclab.app.AppLauncher`` for offline exit-code tests.

This stub lets ``scripts/check_fr3_asset.py`` (and similar scripts) be run under
``PYTHONPATH=tests/stubs`` on a machine **without Isaac Sim**, so the script's
exit-code contract can be regression-tested locally (must-fail-first for the old
swallowed-error behaviour). It launches nothing and creates a fake
``simulation_app`` whose ``close()`` is a no-op (or hangable via env).
"""
from __future__ import annotations

import argparse
import os
from types import SimpleNamespace


class _FakeApp:
    """Fake SimulationApp. ``close()`` can be made to hang for T-05 tests."""

    def __init__(self) -> None:
        self._close_calls = 0

    def close(self) -> None:
        self._close_calls += 1
        # FAKE_CLOSE_HANG=1 => close() blocks forever (to test timeout masking).
        if os.environ.get("FAKE_CLOSE_HANG") == "1":
            import time
            while True:
                time.sleep(0.5)


class AppLauncher:
    def __init__(self, args_cli=None, **kwargs):
        # Accept either a namespace (script path) or kwargs (headless=True).
        self.args = args_cli
        self.app = _FakeApp()

    @staticmethod
    def add_app_launcher_args(parser: argparse.ArgumentParser) -> None:
        # Register only the args the scripts pass on the CLI so argparse parse
        # succeeds; none of them affect the stub.
        for flag, kwargs in (
            ("--headless", {"action": "store_true", "default": False}),
            ("--device", {"type": str, "default": None}),
            ("--enable_cameras", {"action": "store_true", "default": False}),
            ("--livestream", {"type": int, "default": 0}),
            ("--experience", {"type": str, "default": None}),
        ):
            try:
                parser.add_argument(flag, **kwargs)
            except argparse.ArgumentError:
                pass
