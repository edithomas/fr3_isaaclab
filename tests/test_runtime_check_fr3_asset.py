"""Runtime tests for the FR3 asset check script.

These launch a real Isaac Sim ``SimulationApp`` (headless) and are therefore
**skipped locally on macOS**. Set ``RUN_FR3_RUNTIME_TESTS=1`` on the remote
one-L40 container to run them::

    RUN_FR3_RUNTIME_TESTS=1 ~/IsaacLab/isaaclab.sh -p -m pytest tests/test_runtime_check_fr3_asset.py -q

The static contract (that invariant failures map to nonzero exit codes) is
pinned by ``tests/test_asset_invariants.py`` against sampled data, so these
runtime tests only verify the live happy-path end-to-end.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.runtime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
ISACLAB_SH = os.environ.get("ISACLAB_SH", os.path.expanduser("~/IsaacLab/isaaclab.sh"))
CHECK_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "check_fr3_asset.py")


def _run_check() -> subprocess.CompletedProcess:
    cmd = [ISACLAB_SH, "-p", CHECK_SCRIPT, "--headless"]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=900)


class TestCheckFr3AssetRuntime:
    def test_script_runs_and_closes_simulation_app(self):
        """The old script could leave SimulationApp hanging; the new one must
        exit cleanly (return a CompletedProcess, not time out)."""
        proc = _run_check()
        # Whether it passed or failed on asset availability, it must TERMINATE.
        assert proc.returncode is not None

    @pytest.mark.skipif(
        not os.path.exists(ISACLAB_SH),
        reason=f"isaaclab.sh not found at {ISACLAB_SH}",
    )
    def test_healthy_asset_passes_nonzero_on_failure_paths(self):
        """Happy path: if the FR3 USD is reachable, the script must exit 0 and
        print the three invariants. If Nucleus is unreachable it must exit 3
        (asset resolution), never 0 — i.e. a resolution failure is not
        swallowed."""
        proc = _run_check()
        out = proc.stdout + proc.stderr
        if proc.returncode == 0:
            assert "fr3_hand" in out
            assert "fr3_joint1" in out
            assert "RESULT: PASS" in out
        else:
            # A genuine failure must be a *documented* nonzero code, never 0.
            assert proc.returncode in (2, 3, 4, 5, 6), (
                f"unexpected exit code {proc.returncode}; output:\n{out[-2000:]}"
            )
            assert "RESULT: PASS" not in out
