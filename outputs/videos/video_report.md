# FR3 Trajectory Video Report

- Source data: `outputs/record/record_20260724_030014/trajectory.json`
- Steps: 50
- Format: `.gif`
- FPS: 10

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
