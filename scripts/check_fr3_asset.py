"""Check the FR3 USD asset: joint names, body names, end-effector candidates.

Usage::

    ~/IsaacLab/isaaclab.sh -p scripts/check_fr3_asset.py --headless

The script resolves the FR3 USD from the Isaac Sim Nucleus, scrapes its joint /
rigid-body / articulation-root prims, and runs the FR3 asset invariants from
:mod:`fr3_isaaclab_project.diagnostics.invariants`:

* the end-effector body ``fr3_hand`` is present,
* the seven revolute arm joints ``fr3_joint1`` … ``fr3_joint7`` are present,
* at least one applied rigid body exists.

Exit codes (the old script swallowed errors and always returned 0; this version
returns nonzero on every failure path so regressions are detectable in CI):

    0  all invariants held
    2  startup / argument error
    3  asset resolution failure (Nucleus unreachable / fr3.usd missing)
    4  USD stage open failure
    5  one or more asset invariants failed
    6  unexpected exception

The :class:`~isaacsim.app.SimulationApp` is always closed in ``finally``.

Note: requires access to the Isaac Sim Nucleus asset server to download the FR3
USD. If the asset is not reachable the script prints a clear error and exits 3.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check FR3 USD asset structure.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from pxr import Usd, UsdPhysics

from fr3_isaaclab_project.diagnostics import (
    AssetReport,
    classify_asset,
    EXPECTED_ARM_JOINTS,
    EXPECTED_EE_BODY,
)

# Exit codes — distinct so CI/log scrapers can branch on the failure class.
EXIT_OK = 0
EXIT_STARTUP = 2
EXIT_ASSET_RESOLVE = 3
EXIT_STAGE_OPEN = 4
EXIT_INVARIANT = 5
EXIT_UNEXPECTED = 6


def _scrape_stage(stage) -> tuple[list[str], list[str], list[str]]:
    """Traverse the USD stage and return (joint_names, body_names, root_paths).

    Rigid-body and articulation-root detection uses ``prim.HasAPI(...)`` rather
    than ``prim.IsA(...)``: ``UsdPhysics.RigidBodyAPI`` /
    ``UsdPhysics.ArticulationRootAPI`` are *applied* APIs, so ``IsA`` returns
    False for applied-only prims and silently drops real links. This is the bug
    fixed on branch ``origin/fix/fr3-asset-check`` (commit 5a261f7) and
    preserved here.
    """
    joint_names: list[str] = []
    body_names: list[str] = []
    root_paths: list[str] = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            root_paths.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body_names.append(prim.GetName())
        if prim.IsA(UsdPhysics.Joint):
            joint_names.append(prim.GetName())
    return joint_names, body_names, root_paths


def check_fr3_asset() -> int:
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path

    usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/FR3/fr3.usd"

    print(f"\n{'='*70}", flush=True)
    print("FR3 Asset Check", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"USD path requested: {usd_path}", flush=True)

    # --- resolve the path (exit 3 on failure, never swallow) ---
    try:
        resolved = retrieve_file_path(usd_path)
        print(f"Resolved path: {resolved}", flush=True)
    except Exception as e:  # noqa: BLE001 — report any resolution failure
        print("\n[ERROR] Unable to resolve FR3 USD path.", flush=True)
        print(f"  Exception: {e}", flush=True)
        print("\nPossible causes:", flush=True)
        print("  1. Isaac Sim Nucleus / asset server is not reachable", flush=True)
        print("  2. FR3 USD asset is not yet available on Nucleus", flush=True)
        print("  3. Network connectivity issues on this server", flush=True)
        print("\nAlternative: run scripts/generate_fr3_usd.py or place fr3.usd manually", flush=True)
        return EXIT_ASSET_RESOLVE

    # --- open the stage (exit 4 on failure) ---
    try:
        stage = Usd.Stage.Open(resolved)
    except Exception as e:  # noqa: BLE001 — pxr raises Tf.ErrorException on bad paths
        print(f"\n[ERROR] Failed to open USD stage: {resolved}\n  Exception: {e}", flush=True)
        return EXIT_STAGE_OPEN
    if not stage:
        print("[ERROR] Failed to open USD stage (empty stage).", flush=True)
        return EXIT_STAGE_OPEN

    # --- scrape the stage ---
    joint_names, body_names, root_paths = _scrape_stage(stage)

    print(f"\n--- Articulation Roots ({len(root_paths)}) ---", flush=True)
    for p in root_paths:
        print(f"  {p}", flush=True)

    print(f"\n--- Rigid Bodies / Links ({len(body_names)}) ---", flush=True)
    for name in body_names:
        print(f"  {name}", flush=True)

    print(f"\n--- Joints ({len(joint_names)}) ---", flush=True)
    for name in joint_names:
        print(f"  {name}", flush=True)

    # --- run invariants (the load-bearing check) ---
    report: AssetReport = classify_asset(joint_names, body_names, root_paths)

    print(f"\n{'='*70}", flush=True)
    print("INVARIANTS", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  End-effector body '{EXPECTED_EE_BODY}': "
          f"{'FOUND' if report.ee_body_found else 'MISSING'}", flush=True)
    print(f"  Arm joints ({len(EXPECTED_ARM_JOINTS)} expected): "
          f"{report.arm_joint_names}", flush=True)
    if report.missing_arm_joints:
        print(f"  MISSING arm joints: {report.missing_arm_joints}", flush=True)
    if report.extra_arm_joints:
        print(f"  UNEXPECTED extra arm joints: {report.extra_arm_joints}", flush=True)
    print(f"  Rigid bodies: {report.num_rigid_bodies}", flush=True)
    print(f"  Gripper joints: {report.gripper_joint_names}", flush=True)
    if report.ee_candidates:
        print(f"  EE body candidates (keyword match): {report.ee_candidates}", flush=True)

    print(f"\n{'='*70}", flush=True)
    if report.ok:
        print("RESULT: PASS — all FR3 asset invariants held.", flush=True)
        print(f"{'='*70}\n", flush=True)
        return EXIT_OK

    print("RESULT: FAIL — invariant violations:", flush=True)
    for v in report.violations:
        print(f"  - {v}", flush=True)
    print(f"{'='*70}\n", flush=True)
    return EXIT_INVARIANT


if __name__ == "__main__":
    exit_code = EXIT_UNEXPECTED
    try:
        exit_code = check_fr3_asset()
    except Exception:
        import traceback
        print("\n[FATAL] Unexpected exception in check_fr3_asset():", flush=True)
        traceback.print_exc()
        exit_code = EXIT_UNEXPECTED
    finally:
        # ALWAYS close the SimulationApp, even on the success path. The old
        # script relied on a bare except + finally that masked the real exit
        # code; here we preserve ``exit_code`` so failures stay nonzero.
        try:
            simulation_app.close()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(exit_code)
