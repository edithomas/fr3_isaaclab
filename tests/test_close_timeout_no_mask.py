"""T-05 regression: ``close_with_exit_code`` must NOT mask a failure code to 0.

The old repo ``_close_with_timeout`` pattern force-exits with ``os._exit(0)``
when ``simulation_app.close()`` hangs, which converts an in-flight
``sys.exit(nonzero)`` into a false SUCCESS (the "hang-mask" false-pass class).
This test runs a subprocess that sets a failure code and then hangs close(),
asserting the process dies with the FAILURE code (never 0).
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SOURCE_DIR = os.path.join(PROJECT_ROOT, "source", "fr3_isaaclab_project")


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([SOURCE_DIR, env.get("PYTHONPATH", "")])
    return env


HANG_SCRIPT = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, %r)
    from fr3_isaaclab_project.diagnostics.sim_close import close_with_exit_code

    class _HangApp:
        def close(self):
            import time
            while True:
                time.sleep(0.2)  # hang forever

    INTENDED = 7  # a failure code
    close_with_exit_code(_HangApp(), exit_code=INTENDED, timeout_sec=1)
    # If close() returned (it won't), still exit with the intended code.
    sys.exit(INTENDED)
    """ % SOURCE_DIR
)


class TestCloseTimeoutNoMask:
    def test_hung_close_preserves_failure_code(self):
        proc = subprocess.run(
            [sys.executable, "-c", HANG_SCRIPT],
            env=_env(), capture_output=True, text=True, timeout=30,
        )
        # The process must die with the INTENDED failure code, never 0.
        assert proc.returncode == 7, (
            f"expected 7 (preserved failure code), got {proc.returncode}\n"
            f"stdout={proc.stdout[-800:]}\nstderr={proc.stderr[-800:]}"
        )
        assert "force-exiting with code 7" in (proc.stdout + proc.stderr)

    def test_clean_close_returns_caller_code(self):
        """On a clean close the caller's code is returned via normal flow."""
        script = textwrap.dedent(
            """
            import sys
            sys.path.insert(0, %r)
            from fr3_isaaclab_project.diagnostics.sim_close import close_with_exit_code

            class _CleanApp:
                def close(self):
                    pass

            close_with_exit_code(_CleanApp(), exit_code=0, timeout_sec=5)
            sys.exit(0)
            """ % SOURCE_DIR
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            env=_env(), capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 0

    def test_record_script_does_not_use_os_exit_zero_mask(self):
        """Static guard: record_fr3_trajectory.py must not contain the old
        ``os._exit(0)`` hang-mask (it should use close_with_exit_code)."""
        path = os.path.join(PROJECT_ROOT, "scripts", "record_fr3_trajectory.py")
        with open(path) as f:
            src = f.read()
        assert "os._exit(0)" not in src, (
            "record_fr3_trajectory.py still masks failures with os._exit(0)"
        )
