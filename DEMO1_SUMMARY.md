# Demo 1: FR3 轨迹录制与离线可视化

**日期**: 2026-07-24 | **耗时**: ~5 分钟

## 概述

在 headless 容器中运行 FR3-Reach-v0 环境，录制 50 步仿真轨迹数据，然后用离线脚本生成 4 面板动画 GIF。

全程**不需要 GPU 图形渲染、不需要 GUI、不需要训练模型**。

## 执行步骤

### Step 1: 录制轨迹 (`record_fr3_trajectory.py`)

```bash
~/IsaacLab/isaaclab.sh -p scripts/record_fr3_trajectory.py
```

**做了什么**：
1. `AppLauncher(headless=True)` 启动 Isaac Sim
2. `gym.make("FR3-Reach-v0", cfg=FR3ReachEnvCfg)` 创建环境（1 env）
3. `env.reset()` 初始化，FR3 机械臂处于默认关节位姿
4. 循环 50 步：`env.step(torch.zeros(1,7))` — **零力矩动作**，机械臂仅受重力
5. 每步记录 joint_pos(9)、joint_vel(9)、pose_command(7)、actions(7)、reward

**输出**: `outputs/record/record_20260724_030014/`
- `trajectory.json` (74 KB) — 完整轨迹数据
- `joint_positions.png` — 7 关节位置曲线
- `joint_velocities.png` — 7 关节速度曲线
- `ee_pose_and_actions.png` — EE 指令 vs 动作
- `rewards.png` — 每步 reward

### Step 2: 生成动画 (`make_trajectory_video.py`)

```bash
python3 scripts/make_trajectory_video.py \
    --input outputs/record/record_20260724_030014/trajectory.json \
    --fps 10 --dpi 120
```

**做了什么**（完全离线，不依赖 Isaac Sim）：
1. 读取 trajectory.json 中的 joint_pos 数据
2. 用近似 FK（Z-Y-Y-Y-Y-Y-Z 关节轴 + 近似连杆长度）计算 FR3 杆件 3D 位姿
3. 生成 4 面板 matplotlib 动画

**输出**: `outputs/videos/fr3_trajectory_20260724_030040.gif` (3.7 MB, 50 帧, 10 FPS)

## 动画面板说明

| 面板 | 内容 | 技术细节 |
|------|------|----------|
| 左上 | FR3 杆件模型 FK 动画 | 7-DOF 近似运动学，红色球=EE，红色虚线=EE轨迹 |
| 右上 | 末端执行器 XYZ 3D 轨迹 | EE 在空间中的运动路径 |
| 左下 | 7 关节位置 vs 时间 | 滑动窗口（20步），展示重力漂移 |
| 右下 | Reward + Action 幅值 | Reward 变化曲线 + 动作 L2 范数 |

## 核心发现：重力漂移

由于使用零动作（不给力矩），机械臂在重力下自由塌落：

| 关节 | 初始 (rad) | 最终 (rad) | 漂移 (rad) | 漂移速率 (rad/s) |
|------|-----------|-----------|-----------|-----------------|
| joint1 | 0.000 | -0.008 | -0.008 | -0.005 |
| joint2 | -0.266 | 0.259 | **+0.525** | 0.336 |
| joint3 | -0.009 | 0.005 | +0.014 | 0.009 |
| joint4 | 0.884 | -0.225 | **-1.110** | -0.711 |
| joint5 | -0.006 | -0.003 | +0.003 | 0.002 |
| joint6 | -0.844 | -0.072 | **+0.772** | 0.495 |
| joint7 | -0.081 | 0.002 | +0.082 | 0.053 |

- 关节 4 漂移最大：**1.11 rad (63°)**
- 关节 6 漂移：**0.77 rad (44°)**
- 关节 2 漂移：**0.53 rad (30°)**
- 这就是为什么需要 RL 训练 — 策略必须学会输出合适的力矩来抵抗重力并追踪目标

## 学到的关键概念

### Isaac Lab 环境生命周期
```
AppLauncher(headless=True)     → 启动 Isaac Sim
gym.make("FR3-Reach-v0", cfg)  → 创建环境
env.reset()                     → 初始化 → obs (shape: [1, 32])
env.step(action)                → 物理步进 → obs, reward, done, info
env.close()                     → 清理
```

### Observation 结构 (dim=32)
```
joint_pos[0:9]  + joint_vel[9:18] + pose_command[18:25] + actions[25:32]
  (9 DOF)           (9 DOF)            (7: pos3+ori4)       (7 joints)
```

### Reward 组件 (5 项，仅 EE 相关有非零值)
| 组件 | Weight | 说明 |
|------|--------|------|
| `end_effector_position_tracking` | -0.2 | EE 位置误差 |
| `end_effector_position_tracking_fine_grained` | 0.1 | EE 位置精确追踪 |
| `end_effector_orientation_tracking` | -0.1 | EE 姿态误差 |
| `action_rate` | -0.0001 | 动作平滑惩罚 |
| `joint_vel` | -0.0001 | 关节速度惩罚 |

### 环境配置关键参数
| 参数 | 值 |
|------|-----|
| Physics dt | 0.0167s (60Hz) |
| Rendering dt | 0.0333s (30Hz) |
| Action space | 7-DOF joint position targets |
| Env spacing | 2.5m |
| Device | cuda:0 |

## 前置修复

在运行 Demo 1 之前，修复了 Isaac Sim 无法启动的问题：

- **问题**: `ImportError: libhdx.so: cannot open shared object file`
- **原因**: pip 版 Isaac Sim 的 `python.sh` 未将 `extscache/*/bin/` 加入 `LD_LIBRARY_PATH`
- **修复**: 修改 `~/.local/lib/python3.10/site-packages/isaacsim/python.sh`，遍历所有 extcache bin 目录加入 `LD_LIBRARY_PATH`
- **详见**: `ENV_FIX.md`（待创建）

## 相关文件

```
outputs/
├── record/record_20260724_030014/
│   ├── trajectory.json          # 50 步仿真数据
│   ├── joint_positions.png      # 关节位置图
│   ├── joint_velocities.png     # 关节速度图
│   ├── ee_pose_and_actions.png  # EE 轨迹图
│   ├── rewards.png              # Reward 图
│   └── record_report.md         # 录制报告
└── videos/
    ├── fr3_trajectory_20260724_030040.gif  # 4 面板动画 (3.7 MB)
    └── video_report.md                     # 视频报告

scripts/
├── record_fr3_trajectory.py     # 轨迹录制脚本
└── make_trajectory_video.py     # 离线动画生成脚本
```
