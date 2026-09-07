"""Generate FR3 USD from URDF using Isaac Sim's URDF importer.

Since the FR3 USD is not yet available on the Isaac Sim Nucleus, this
script imports the FR3 URDF (which IS shipped with Isaac Sim 4.5.0) and
substitutes meshes from the Panda description (FR3 and Panda share identical
mechanical geometry).

Usage:
    ~/IsaacLab/isaaclab.sh -p scripts/generate_fr3_usd.py --headless

Exit codes: 0 = USD generated; 1 = URDF/import failure; 6 = unexpected.

Output:  ~/fr3_isaaclab_project/assets/fr3/fr3.usd
"""

import argparse
import os
import shutil
import sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Generate FR3 USD from URDF.")
parser.add_argument("--isaacsim-root", type=str,
                    default=os.environ.get("ISAACSIM_ROOT",
                                           "/public/home/s2353600/.local/lib/python3.10/site-packages/isaacsim"),
                    help="pip isaacsim install root (overridable via ISAACSIM_ROOT env)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.kit.commands
import omni.usd
from pxr import Sdf, UsdPhysics, Gf


def generate_fr3_usd():
    # Paths
    isaacsim_root = args_cli.isaacsim_root
    ext_path = f"{isaacsim_root}/extscache/isaacsim.asset.importer.urdf-2.3.10+106.4.0.lx64.r.cp310"
    panda_data = f"{ext_path}/data/urdf/robots/franka_description"

    fr3_urdf_source = f"{isaacsim_root}/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/FR3/fr3.urdf"

    output_dir = os.path.expanduser("~/fr3_isaaclab_project/assets/fr3")
    os.makedirs(output_dir, exist_ok=True)
    output_usd = os.path.join(output_dir, "fr3.usd")

    if not os.path.exists(fr3_urdf_source):
        print(f"[ERROR] FR3 URDF not found at: {fr3_urdf_source}")
        print(f"        Set --isaacsim-root or ISAACSIM_ROOT env to your pip isaacsim install.")
        return False

    # Create a temp directory for the URDF + meshes
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="fr3_import_")
    print(f"Temp workspace: {tmpdir}")

    # Copy FR3 URDF to temp
    shutil.copy(fr3_urdf_source, os.path.join(tmpdir, "fr3.urdf"))

    # Create meshes/fr3/{visual,collision} symlinks pointing to Panda meshes
    os.makedirs(os.path.join(tmpdir, "meshes", "fr3", "visual"), exist_ok=True)
    os.makedirs(os.path.join(tmpdir, "meshes", "fr3", "collision"), exist_ok=True)

    # Panda visual meshes -> FR3 visual meshes
    panda_visual = os.path.join(panda_data, "meshes", "visual")
    for mesh in os.listdir(panda_visual):
        src = os.path.join(panda_visual, mesh)
        dst = os.path.join(tmpdir, "meshes", "fr3", "visual", mesh)
        if os.path.isfile(src):
            shutil.copy(src, dst)

    # Panda collision meshes -> FR3 collision meshes
    panda_collision = os.path.join(panda_data, "meshes", "collision")
    for mesh in os.listdir(panda_collision):
        src = os.path.join(panda_collision, mesh)
        dst = os.path.join(tmpdir, "meshes", "fr3", "collision", mesh)
        if os.path.isfile(src):
            shutil.copy(src, dst)

    # Open a new stage
    omni.usd.get_context().new_stage()

    # Import the FR3 URDF
    status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    import_config.merge_fixed_joints = False
    import_config.fix_base = True
    import_config.make_default_prim = True
    import_config.create_physics_scene = True

    print("Importing FR3 URDF...")
    try:
        omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=os.path.join(tmpdir, "fr3.urdf"),
            import_config=import_config,
        )
    except Exception as e:
        print(f"[ERROR] URDF import failed: {e}")
        return False

    # Save the stage as USD
    stage = omni.usd.get_context().get_stage()

    # Add physics scene if not already present
    scene_path = Sdf.Path("/physicsScene")
    if not stage.GetPrimAtPath(scene_path):
        scene = UsdPhysics.Scene.Define(stage, scene_path)
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(9.81)

    # Export
    print(f"Exporting to: {output_usd}")
    stage.Export(output_usd)
    print(f"FR3 USD successfully generated at: {output_usd}")

    # Clean up temp dir
    shutil.rmtree(tmpdir, ignore_errors=True)

    return True


if __name__ == "__main__":
    success = False
    exit_code = 6  # unexpected
    try:
        success = generate_fr3_usd()
        exit_code = 0 if success else 1
        if success:
            print("\nSUCCESS: FR3 USD generated.")
            print("You can now update assets/fr3.py to use the local path:")
            print("  FR3_USD_PATH = os.path.expanduser('~/fr3_isaaclab_project/assets/fr3/fr3.usd')")
    except Exception:
        import traceback
        traceback.print_exc()
        exit_code = 6
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
        sys.stdout.flush()
        sys.stderr.flush()
        sys.exit(exit_code)
