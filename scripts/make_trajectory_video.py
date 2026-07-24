#!/usr/bin/env python
"""
Offline trajectory animation from trajectory.json — NO Isaac Sim, NO Vulkan, NO GUI.

Generates a 4-panel animation:
  Top-left:    Simplified FR3 stick-model (approximate FK, no mesh)
  Top-right:   EE end-effector XYZ trajectory (3D view)
  Bottom-left: 7 actuated joint positions vs time (rolling)
  Bottom-right: Reward / action norm curve

Input:  outputs/record/record_*/trajectory.json
Output: outputs/videos/fr3_trajectory.mp4 (or .gif fallback)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np

# ===========================================================================
# Approximate FR3 Forward Kinematics (stick-model only, NOT physically accurate)
# ===========================================================================
# FR3 is a 7-DOF arm. Joint axes alternate roughly Z-Y-Y-Y-Y-Y-Z.
# Link lengths are approximate from FR3 datasheet / URDF.
# Joints 1-7 are actuated; joints 8,9 are fixed (fr3_joint8, fr3_hand_joint).

FR3_LINK_LENGTHS = np.array([0.333, 0.316, 0.0, 0.384, 0.0, 0.0, 0.107])  # meters
FR3_JOINT_AXES = [
    np.array([0, 0, 1]),   # j1: Z
    np.array([0, 1, 0]),   # j2: Y
    np.array([0, 1, 0]),   # j3: Y
    np.array([0, 1, 0]),   # j4: Y
    np.array([0, 1, 0]),   # j5: Y
    np.array([0, 1, 0]),   # j6: Y
    np.array([0, 0, 1]),   # j7: Z
]


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula — 3x3 rotation matrix."""
    axis = axis / np.linalg.norm(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    v = 1 - c
    x, y, z = axis
    return np.array([
        [c + x*x*v,     x*y*v - z*s,   x*z*v + y*s],
        [x*y*v + z*s,   c + y*y*v,     y*z*v - x*s],
        [x*z*v - y*s,   y*z*v + x*s,   c + z*z*v],
    ])


def fr3_fk(joint_angles: np.ndarray) -> np.ndarray:
    """
    Approximate FK for FR3 7-DOF arm.
    joint_angles: shape (7,) or (9,) — if 9, use only first 7.
    Returns: (8, 3) array of joint positions (base + 7 joint origins).
    """
    angles = joint_angles[:7]  # Use only actuated joints

    positions = np.zeros((8, 3))  # base(0) + 7 links
    R = np.eye(3)  # cumulative rotation
    p = np.array([0.0, 0.0, 0.0])  # current position

    for i in range(7):
        positions[i] = p.copy()
        Ri = rotation_matrix(FR3_JOINT_AXES[i], angles[i])
        R = R @ Ri
        # Move along Z of current frame by link length
        p = p + R @ np.array([0, 0, FR3_LINK_LENGTHS[i]])

    positions[7] = p.copy()  # end-effector

    return positions


# ===========================================================================
# Plot generation
# ===========================================================================
def make_animation(
    records: list[dict],
    output_path: str,
    fps: int = 10,
    dpi: int = 100,
) -> bool:
    """Generate 4-panel trajectory animation and save to output_path."""

    num_steps = len(records)
    # Pre-compute FK for all steps
    all_joint_pos = np.array([r["joint_pos"] for r in records])  # (N, 9)
    all_fk = np.array([fr3_fk(r["joint_pos"]) for r in records])  # (N, 8, 3)
    ee_traj = all_fk[:, 7, :]  # (N, 3) — end-effector position
    all_rewards = np.array([r["reward"] for r in records])
    all_actions = np.array([r["prev_action"] for r in records])  # (N, 7)
    action_norms = np.linalg.norm(all_actions, axis=1)

    # --- Create figure ---
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("FR3 Trajectory Visualization (Headless, No-Vulkan)", fontsize=14, fontweight="bold")

    # Subplot layout: 2x2
    ax_3d = fig.add_subplot(2, 2, 1, projection="3d")  # Stick model
    ax_ee = fig.add_subplot(2, 2, 2, projection="3d")  # EE trajectory
    ax_joints = fig.add_subplot(2, 2, 3)                # Joint positions
    ax_reward = fig.add_subplot(2, 2, 4)                # Reward + action norm

    # Determine axis limits
    fk_all = all_fk.reshape(-1, 3)
    x_min, x_max = fk_all[:, 0].min() - 0.1, fk_all[:, 0].max() + 0.1
    y_min, y_max = fk_all[:, 1].min() - 0.1, fk_all[:, 1].max() + 0.1
    z_min, z_max = fk_all[:, 2].min() - 0.1, fk_all[:, 2].max() + 0.1

    joint_min, joint_max = all_joint_pos[:, :7].min() - 0.1, all_joint_pos[:, :7].max() + 0.1

    colors = plt.cm.tab10(np.linspace(0, 1, 7))

    def _update(frame_idx: int):
        ax_3d.clear()
        ax_ee.clear()
        ax_joints.clear()
        ax_reward.clear()

        fk = all_fk[frame_idx]  # (8, 3)

        # ---- Top-Left: Stick model ----
        ax_3d.plot(fk[:, 0], fk[:, 1], fk[:, 2], "ko-", linewidth=2, markersize=4)
        ax_3d.scatter(*fk[7], c="red", s=60, marker="o", label="EE", zorder=5)
        ax_3d.scatter(*fk[0], c="black", s=40, marker="s", label="Base", zorder=5)
        # Trace EE
        if frame_idx > 0:
            ax_3d.plot(ee_traj[:frame_idx+1, 0],
                       ee_traj[:frame_idx+1, 1],
                       ee_traj[:frame_idx+1, 2], "r--", alpha=0.3, linewidth=0.8)
        ax_3d.set_xlim(x_min, x_max)
        ax_3d.set_ylim(y_min, y_max)
        ax_3d.set_zlim(z_min, z_max)
        ax_3d.set_xlabel("X (m)")
        ax_3d.set_ylabel("Y (m)")
        ax_3d.set_zlabel("Z (m)")
        ax_3d.set_title(f"FR3 Stick Model — Step {frame_idx+1}/{num_steps}")
        ax_3d.legend(loc="upper left", fontsize=7)
        ax_3d.view_init(elev=25, azim=45 + frame_idx * 0.5)  # slowly rotate

        # ---- Top-Right: EE 3D trajectory ----
        ax_ee.plot(ee_traj[:frame_idx+1, 0],
                   ee_traj[:frame_idx+1, 1],
                   ee_traj[:frame_idx+1, 2], "b-", linewidth=1.5, alpha=0.8)
        ax_ee.scatter(*ee_traj[frame_idx], c="red", s=50, marker="o", zorder=5)
        ax_ee.set_xlim(ee_traj[:, 0].min() - 0.05, ee_traj[:, 0].max() + 0.05)
        ax_ee.set_ylim(ee_traj[:, 1].min() - 0.05, ee_traj[:, 1].max() + 0.05)
        ax_ee.set_zlim(ee_traj[:, 2].min() - 0.05, ee_traj[:, 2].max() + 0.05)
        ax_ee.set_xlabel("X (m)"); ax_ee.set_ylabel("Y (m)"); ax_ee.set_zlabel("Z (m)")
        ax_ee.set_title("End-Effector XYZ Trajectory")

        # ---- Bottom-Left: Joint positions ----
        steps = np.arange(max(0, frame_idx - 20), frame_idx + 1)
        for j in range(7):
            ax_joints.plot(steps, all_joint_pos[steps, j],
                           color=colors[j], label=f"j{j+1}", linewidth=1.2, alpha=0.8)
        ax_joints.axvline(x=frame_idx, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
        ax_joints.set_xlim(max(0, frame_idx - 20), max(frame_idx + 2, 22))
        ax_joints.set_ylim(joint_min, joint_max)
        ax_joints.set_xlabel("Step"); ax_joints.set_ylabel("Joint position (rad)")
        ax_joints.set_title("7 Actuated Joint Positions")
        ax_joints.legend(loc="upper right", fontsize=6, ncol=4)
        ax_joints.grid(True, alpha=0.3)

        # ---- Bottom-Right: Reward + Action norm ----
        ax2 = ax_reward.twinx()
        ax_reward.plot(range(frame_idx + 1), all_rewards[:frame_idx+1], "b-", linewidth=1.5, label="Reward")
        ax2.plot(range(frame_idx + 1), action_norms[:frame_idx+1], "r-", linewidth=1.5, alpha=0.6, label="Action norm")
        ax_reward.set_xlim(0, num_steps)
        ax_reward.set_xlabel("Step"); ax_reward.set_ylabel("Reward", color="b")
        ax2.set_ylabel("Action Norm", color="r")
        ax_reward.set_title("Reward & Action Magnitude")
        ax_reward.legend(loc="upper left", fontsize=7)
        ax2.legend(loc="upper right", fontsize=7)
        ax_reward.grid(True, alpha=0.3)
        ax_reward.tick_params(axis="y", labelcolor="b")
        ax2.tick_params(axis="y", labelcolor="r")

        fig.tight_layout()

    # Create animation
    ani = anim.FuncAnimation(fig, _update, frames=num_steps, repeat=True, interval=1000//fps)

    # ---- Try saving ----
    saved = False

    # 1. Try ffmpeg (system or imageio)
    try:
        ani.save(output_path, fps=fps, writer="ffmpeg", dpi=dpi)
        saved = True
        print(f"  Saved MP4 (ffmpeg): {output_path}")
    except Exception as e:
        print(f"  [INFO] ffmpeg not available ({e})")

    # 2. Try pillow (GIF)
    if not saved:
        gif_path = output_path.replace(".mp4", ".gif")
        try:
            ani.save(gif_path, fps=fps, writer="pillow", dpi=dpi)
            saved = True
            print(f"  Saved GIF (pillow): {gif_path}")
        except Exception as e:
            print(f"  [INFO] pillow GIF failed ({e})")

    # 3. Try saving as PNG frame sequence
    if not saved:
        frames_dir = os.path.join(os.path.dirname(output_path), "frames")
        os.makedirs(frames_dir, exist_ok=True)
        for i in range(num_steps):
            _update(i)
            frame_path = os.path.join(frames_dir, f"frame_{i:04d}.png")
            fig.savefig(frame_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved {num_steps} PNG frames to: {frames_dir}")
        # Write a simple HTML viewer
        html_path = os.path.join(frames_dir, "viewer.html")
        with open(html_path, "w") as f:
            f.write(f"""<!DOCTYPE html><html><head><title>FR3 Trajectory Frames</title>
<style>body{{background:#111;text-align:center;color:#fff;font-family:monospace}}
img{{max-width:90%;margin:5px auto;display:block}}</style></head><body>
<h1>FR3 Trajectory — Frame Sequence</h1><p>{num_steps} frames</p>
{"".join(f'<img src="frame_{i:04d}.png" alt="frame {i}"><p>Frame {i}</p>' for i in range(num_steps))}
</body></html>""")
        print(f"  Saved viewer: {html_path}")
        saved = True

    plt.close(fig)
    return saved


# ===========================================================================
# Main
# ===========================================================================
def main():
    import argparse
    p = argparse.ArgumentParser(description="Generate offline FR3 trajectory animation")
    p.add_argument("--input", type=str, required=True, help="Path to trajectory.json")
    p.add_argument("--output", type=str, default=None, help="Output video path (.mp4 or .gif)")
    p.add_argument("--fps", type=int, default=10, help="Frames per second")
    p.add_argument("--dpi", type=int, default=100, help="Output resolution DPI")
    args = p.parse_args()

    # Load data
    with open(args.input, "r") as f:
        data = json.load(f)

    records = data["records"]
    num_steps = len(records)
    num_joints = len(records[0]["joint_pos"])
    print(f"Loaded: {args.input}")
    print(f"  Steps: {num_steps}")
    print(f"  Joint dim: {num_joints} (7 actuated + {num_joints - 7} fixed)")
    print(f"  Step time: {data['step_time_sec']:.4f}s")

    # Output path
    if args.output is None:
        out_dir = "outputs/videos"
        os.makedirs(out_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join(out_dir, f"fr3_trajectory_{timestamp}.mp4")
    else:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Output: {args.output}")
    print(f"FPS: {args.fps}, DPI: {args.dpi}")

    saved = make_animation(records, args.output, fps=args.fps, dpi=args.dpi)

    if saved:
        # Determine actual output path
        if os.path.exists(args.output):
            actual = args.output
        elif os.path.exists(args.output.replace(".mp4", ".gif")):
            actual = args.output.replace(".mp4", ".gif")
        else:
            actual = os.path.join(os.path.dirname(args.output), "frames")

        size_str = ""
        if os.path.isfile(actual):
            size_mb = os.path.getsize(actual) / 1e6
            size_str = f" ({size_mb:.1f} MB)"

        print(f"\nDone. Output: {actual}{size_str}")

        # Write report
        report_path = os.path.join(os.path.dirname(actual) if os.path.isfile(actual) else actual, "..", "video_report.md")
        report_path = os.path.normpath(report_path)
        if os.path.isfile(actual):
            report_path = os.path.join(os.path.dirname(actual), "video_report.md")

        fmt = os.path.splitext(actual)[1]
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"""# FR3 Trajectory Video Report

- Source data: `{args.input}`
- Steps: {num_steps}
- Format: `{fmt}`
- FPS: {args.fps}

## Video Content (4 panels)

| Panel | Content |
|---|---|
| Top-left | FR3 stick-model FK animation (approximate, 7-DOF) |
| Top-right | End-effector XYZ 3D trajectory |
| Bottom-left | 7 actuated joint positions (rolling window) |
| Bottom-right | Reward + action norm curves |

## Notes

- Generated **offline** from trajectory state data — no Isaac Sim, no Vulkan, no GUI required.
- FK model uses approximate link lengths and simplified joint axes (Z-Y-Y-Y-Y-Y-Z).
- Joints 1-7 are actuated; 2 fixed joints (fr3_joint8, fr3_hand_joint) are ignored.
- Isaac Sim native camera video is unavailable due to Vulkan `ERROR_INCOMPATIBLE_DRIVER` in this container.
- This animation demonstrates FR3 joint motion, EE trajectory, and reward signals — suitable for reports and demos.
""")
        print(f"Report: {report_path}")
    else:
        print("[FATAL] Could not save animation in any format.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
