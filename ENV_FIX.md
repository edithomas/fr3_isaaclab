# 环境修复记录

**日期**: 2026-07-23 | **问题**: Isaac Sim 无法启动

## 错误

```python
ImportError: libhdx.so: cannot open shared object file: No such file or directory
```

发生在 `isaaclab.app.AppLauncher._load_extensions()` 中的 `import omni.physx.bindings._physx`。

## 根因

pip 版 Isaac Sim (4.5.0-rc.36) 的 `python.sh` 只将 `omni/` 加入了 `LD_LIBRARY_PATH`，但 Python C 扩展 `_physx.cpython-310-x86_64-linux-gnu.so` 依赖的 153 个 `.so` 库（`libhdx.so`、`libhd.so`、`libusd.so` 等）位于 `isaacsim/extscache/omni.usd.libs-*/bin/`，不在动态链接器的搜索路径中。

此外还缺少 `libSM.so.6`、`libICE.so.6` 等系统库。

## 修复

### 1. 安装缺失系统库

```bash
sudo apt-get install -y libsm6 libice6
```

### 2. 修改 python.sh 添加扩展库路径

文件: `~/.local/lib/python3.10/site-packages/isaacsim/python.sh`

```bash
# 原始 (仅添加 omni/)
export LD_LIBRARY_PATH="$KIT_PATH:$LD_LIBRARY_PATH"

# 修复后 (添加所有 ext 库目录)
LIB_PATHS="$KIT_PATH"
LIB_PATHS="$LIB_PATHS:$KIT_PATH/kernel/plugins"  # libpython3.10.so

# 遍历所有 extscache 和 extsPhysics 的 bin/ 目录
for bin_dir in "$ISAACSIM_PATH"/extscache/*/bin; do
    [ -d "$bin_dir" ] && LIB_PATHS="$LIB_PATHS:$bin_dir"
done
for bin_dir in "$ISAACSIM_PATH"/extsPhysics/*/bin; do
    [ -d "$bin_dir" ] && LIB_PATHS="$LIB_PATHS:$bin_dir"
done

export LD_LIBRARY_PATH="$LIB_PATHS:$LD_LIBRARY_PATH"
```

原文件已备份到 `python.sh.bak`。

## 验证

```bash
# 列出 FR3 环境
~/IsaacLab/isaaclab.sh -p scripts/list_envs.py --headless

# 验证 FR3 USD 资产
~/IsaacLab/isaaclab.sh -p scripts/check_fr3_asset.py --headless

# 物理仿真验证
bash scripts/run_headless.sh
```

三项均通过 ✅

## 注意

`python.sh` 位于 pip 安装目录，`pip install --upgrade isaacsim` 可能覆盖此修复。如果升级 Isaac Sim 后问题复现，需要重新应用此修复。
