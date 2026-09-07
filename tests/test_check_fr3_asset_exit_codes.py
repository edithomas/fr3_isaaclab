"""Must-fail-first exit-code regression tests for ``check_fr3_asset.py``.

These run the **actual** script as a subprocess under a stub ``isaaclab``
package + ``usd-core`` (``pxr``), so the script's full code path — including
``pxr`` USD traversal, the ``IsA``/``HasAPI`` fix, and the invariant layer — is
exercised locally on macOS **without Isaac Sim**. They pin the contract that
the old script violated (every failure returned exit 0).

Branches (from the test-review report T-01..T-04):
* T-01 asset-resolution failure (retrieve raises)      -> exit 3
* T-02 USD stage open failure (bad path)               -> exit 4
* T-03 unexpected exception                            -> exit != 0
* T-04 healthy applied-API stage                        -> exit 0
       (HasAPI detects rigid bodies + fr3_hand; IsA still detects joints)
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
CHECK_SCRIPT = os.path.join(PROJECT_ROOT, "scripts", "check_fr3_asset.py")
STUBS_DIR = os.path.join(os.path.dirname(__file__), "stubs")
SOURCE_DIR = os.path.join(PROJECT_ROOT, "source", "fr3_isaaclab_project")

# Exit codes defined in check_fr3_asset.py
EXIT_OK = 0
EXIT_STARTUP = 2
EXIT_ASSET_RESOLVE = 3
EXIT_STAGE_OPEN = 4
EXIT_INVARIANT = 5
EXIT_UNEXPECTED = 6

_HAS_PXR = True
try:  # noqa: SIM105
    import pxr  # noqa: F401
except Exception:  # noqa: BLE001
    _HAS_PXR = False


def _env(extras=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [STUBS_DIR, SOURCE_DIR, env.get("PYTHONPATH", "")]
    )
    if extras:
        env.update(extras)
    return env


def _build_healthy_usd(path: str) -> str:
    """Build a minimal 'URDF-import-style' FR3 USD with applied APIs."""
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.CreateNew(path)
    root = stage.DefinePrim("/fr3", "Xform")
    UsdPhysics.ArticulationRootAPI.Apply(root)
    # 7 arm links + fr3_hand, each a rigid body via applied API
    for name in (
        "fr3_link0", "fr3_link1", "fr3_link2", "fr3_link3",
        "fr3_link4", "fr3_link5", "fr3_link6", "fr3_link7",
        "fr3_hand", "fr3_leftfinger", "fr3_rightfinger", "fr3_hand_tcp",
    ):
        p = stage.DefinePrim(f"/fr3/{name}", "Xform")
        UsdPhysics.RigidBodyAPI.Apply(p)
    # 7 revolute arm joints (PhysicsRevoluteJoint is-a UsdPhysics.Joint)
    for j in range(1, 8):
        stage.DefinePrim(f"/fr3/fr3_joint{j}", "PhysicsRevoluteJoint")
    # 2 finger joints
    stage.DefinePrim("/fr3/fr3_finger_joint1", "PhysicsRevoluteJoint")
    stage.DefinePrim("/fr3/fr3_finger_joint2", "PhysicsRevoluteJoint")
    stage.GetRootLayer().Save()
    return path


def _run(env_extras, timeout=60):
    proc = subprocess.run(
        [sys.executable, CHECK_SCRIPT, "--headless"],
        env=_env(env_extras),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc


class TestCheckFr3AssetExitCodes:
    def test_t01_asset_resolution_failure_exits_nonzero(self, tmp_path):
        """retrieve_file_path raises -> exit 3 (old script returned 0)."""
        env = {"FAKE_RETRIEVE_MODE": "raise", "FAKE_USD_PATH": str(tmp_path / "x.usd")}
        proc = _run(env)
        assert proc.returncode == EXIT_ASSET_RESOLVE, (
            f"expected {EXIT_ASSET_RESOLVE}, got {proc.returncode}\n{proc.stdout[-1500:]}"
        )
        assert "Unable to resolve FR3 USD" in proc.stdout

    def test_t02_stage_open_failure_exits_nonzero(self, tmp_path):
        """Stage.Open on a nonexistent path -> exit 4 (old script returned 0)."""
        env = {"FAKE_RETRIEVE_MODE": "ok",
               "FAKE_USD_PATH": str(tmp_path / "does_not_exist.usd")}
        proc = _run(env)
        assert proc.returncode == EXIT_STAGE_OPEN, (
            f"expected {EXIT_STAGE_OPEN}, got {proc.returncode}\n{proc.stdout[-1500:]}"
        )

    def test_t03_unexpected_exception_exits_nonzero(self, tmp_path):
        """An unexpected exception (here: a broken diagnostics import) must NOT
        be swallowed to exit 0. The old script returned 0 on tracebacks."""
        bad_pkg = tmp_path / "fr3_isaaclab_project"
        bad_pkg.mkdir()
        (bad_pkg / "__init__.py").write_text("")
        (bad_pkg / "diagnostics").mkdir()
        (bad_pkg / "diagnostics" / "__init__.py").write_text("raise RuntimeError('broken diagnostics')")
        e = dict(os.environ)
        # Shadow the real source dir with the broken package.
        e["PYTHONPATH"] = os.pathsep.join([STUBS_DIR, str(tmp_path), e.get("PYTHONPATH", "")])
        proc = subprocess.run(
            [sys.executable, CHECK_SCRIPT, "--headless"],
            env=e, capture_output=True, text=True, timeout=60,
        )
        # Top-level import failures escape to Python's default handler (exit 1);
        # the in-body __main__ except maps to EXIT_UNEXPECTED (6). Both are
        # nonzero, which is the must-fail-first contract.
        assert proc.returncode != 0, (
            f"expected nonzero, got {proc.returncode}\n{proc.stdout[-1500:]}"
        )

    @pytest.mark.skipif(not _HAS_PXR, reason="usd-core (pxr) not installed")
    def test_t04_healthy_applied_api_stage_exits_zero(self, tmp_path):
        """Healthy FR3 stage with applied APIs -> exit 0; HasAPI finds bodies,
        IsA finds joints (the fix this PR preserves)."""
        usd = _build_healthy_usd(str(tmp_path / "fr3.usd"))
        env = {"FAKE_RETRIEVE_MODE": "ok", "FAKE_USD_PATH": usd}
        proc = _run(env)
        assert proc.returncode == EXIT_OK, (
            f"expected 0, got {proc.returncode}\n{proc.stdout[-2500:]}"
        )
        out = proc.stdout
        assert "fr3_hand" in out
        assert "fr3_joint1" in out
        assert "RESULT: PASS" in out

    @pytest.mark.skipif(not _HAS_PXR, reason="usd-core (pxr) not installed")
    def test_t04b_missing_ee_body_fails(self, tmp_path):
        """If fr3_hand is absent the invariant fails -> exit 5 (not 0)."""
        from pxr import Usd, UsdPhysics

        path = str(tmp_path / "no_hand.usd")
        stage = Usd.Stage.CreateNew(path)
        root = stage.DefinePrim("/fr3", "Xform")
        UsdPhysics.ArticulationRootAPI.Apply(root)
        for name in ("fr3_link0", "fr3_link1"):  # no fr3_hand
            p = stage.DefinePrim(f"/fr3/{name}", "Xform")
            UsdPhysics.RigidBodyAPI.Apply(p)
        for j in range(1, 8):
            stage.DefinePrim(f"/fr3/fr3_joint{j}", "PhysicsRevoluteJoint")
        stage.GetRootLayer().Save()

        env = {"FAKE_RETRIEVE_MODE": "ok", "FAKE_USD_PATH": path}
        proc = _run(env)
        assert proc.returncode == EXIT_INVARIANT, (
            f"expected {EXIT_INVARIANT}, got {proc.returncode}\n{proc.stdout[-1500:]}"
        )
        assert "RESULT: FAIL" in proc.stdout
