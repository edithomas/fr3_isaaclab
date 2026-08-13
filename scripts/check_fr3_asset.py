"""Check FR3 USD asset: print joint names, body names, end-effector candidates.

Usage:
    ~/IsaacLab/isaaclab.sh -p scripts/check_fr3_asset.py --headless

Note: requires access to the Isaac Sim Nucleus asset server to download
the FR3 USD. If the asset is not reachable the script prints a clear
error message and exits.
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


def check_fr3_asset():
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path

    usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/FR3/fr3.usd"

    print(f"\n{'='*70}", flush=True)
    print(f"FR3 Asset Check", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"USD path requested: {usd_path}", flush=True)

    # --- resolve the path ---
    try:
        resolved = retrieve_file_path(usd_path)
        print(f"Resolved path: {resolved}", flush=True)
    except Exception as e:
        print(f"\n[ERROR] Unable to resolve FR3 USD path.", flush=True)
        print(f"  Exception: {e}", flush=True)
        print(f"\nPossible causes:", flush=True)
        print(f"  1. Isaac Sim Nucleus / asset server is not reachable", flush=True)
        print(f"  2. FR3 USD asset is not yet available on Nucleus", flush=True)
        print(f"  3. Network connectivity issues on this server", flush=True)
        print(f"\nAlternative: Manually place fr3.usd and update assets/fr3.py", flush=True)
        return

    # --- open the stage ---
    stage = Usd.Stage.Open(resolved)
    if not stage:
        print("[ERROR] Failed to open USD stage.", flush=True)
        return

    # --- collect articulation roots ---
    print(f"\n--- Articulation Roots ---", flush=True)
    articulations = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulations.append(prim)
            print(f"  {prim.GetPath()}", flush=True)

    # --- collect rigid bodies (links) ---
    print(f"\n--- Rigid Bodies (Links) ---", flush=True)
    body_names = []
    ee_candidates = []
    ee_keywords = [
        "hand", "tcp", "link8", "link_8", "ee", "end_effector",
        "flange", "tool0", "gripper", "gripper_center",
    ]
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            name = prim.GetName()
            path = str(prim.GetPath())
            body_names.append((name, path))
            print(f"  {name:32s} path={path}", flush=True)
            name_lower = name.lower()
            if any(kw in name_lower for kw in ee_keywords):
                ee_candidates.append((name, path))

    # --- collect joints ---
    print(f"\n--- Joints ---", flush=True)
    joint_names = []
    arm_joint_names = []
    gripper_joint_names = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            name = prim.GetName()
            path = str(prim.GetPath())
            joint_names.append((name, path))
            print(f"  {name:32s} path={path}", flush=True)
            if "finger" in name.lower() or "gripper" in name.lower():
                gripper_joint_names.append(name)
            else:
                arm_joint_names.append(name)

    # --- summary ---
    print(f"\n{'='*70}", flush=True)
    print(f"SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  Total joints:         {len(joint_names)}", flush=True)
    print(f"  Arm joints:           {len(arm_joint_names)}", flush=True)
    print(f"  Gripper joints:       {len(gripper_joint_names)}", flush=True)
    print(f"  Total rigid bodies:   {len(body_names)}", flush=True)

    if arm_joint_names:
        # derive common prefix
        from collections import Counter
        prefixes = [j.rstrip("0123456789_") for j in arm_joint_names]
        common = Counter(prefixes).most_common(1)[0][0] if prefixes else ""
        print(f"\n  >>> Suggested arm_joint_names regex: \"{common}.*\"", flush=True)
        print(f"  >>> Arm joint list:                   {arm_joint_names}", flush=True)
    else:
        print("\n  [WARN] No arm joints detected.", flush=True)

    if gripper_joint_names:
        print(f"  >>> Gripper joint names:              {gripper_joint_names}", flush=True)

    if ee_candidates:
        print(f"\n  >>> End-effector body candidates:", flush=True)
        for name, path in ee_candidates:
            print(f"        {name:32s} {path}", flush=True)
    else:
        print(f"\n  >>> [WARN] No end-effector body candidates found.", flush=True)
        print(f"  >>> All body names for manual review:", flush=True)
        for name, path in body_names:
            print(f"        {name}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("Next steps:", flush=True)
    print("  1. Verify FR3_INIT_JOINT_POS keys match joint names in assets/fr3.py", flush=True)
    print("  2. Update FR3_SHOULDER_PATTERN / FR3_FOREARM_PATTERN regexes", flush=True)
    print("  3. Update EE_BODY in fr3_reach_env_cfg.py", flush=True)
    print("  4. Update the action joint_names in fr3_reach_env_cfg.py", flush=True)
    print(f"{'='*70}\n", flush=True)


if __name__ == "__main__":
    try:
        check_fr3_asset()
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
