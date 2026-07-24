#!/usr/bin/env bash
set -euo pipefail

cd ~/fr3_isaaclab_project

# Resolve Isaac Sim lib directory for LD_LIBRARY_PATH
ISAAC_SIM_LIB_DIR=$(dirname "$(find ~/.local/lib/python3.10/site-packages/isaacsim/extscache/omni.usd.libs-*/bin/libhdx.so 2>/dev/null | head -1)")

export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export LIVESTREAM=2

# Add libhdx.so and other PhysX deps to library path
if [ -n "$ISAAC_SIM_LIB_DIR" ] && [ -d "$ISAAC_SIM_LIB_DIR" ]; then
    export LD_LIBRARY_PATH="$ISAAC_SIM_LIB_DIR:$LD_LIBRARY_PATH"
fi

# Start Xorg with NVIDIA driver (headless server rendering)
XORG_PID=""
if [ -z "${DISPLAY:-}" ]; then
    export DISPLAY=:99
    # Create Xorg config for headless NVIDIA
    XORG_CONF_DIR="/tmp/xorg.conf.d.$$"
    mkdir -p "$XORG_CONF_DIR"
    cat > "$XORG_CONF_DIR/nvidia-headless.conf" << 'XORGEOF'
Section "Device"
    Identifier     "NVIDIA"
    Driver         "nvidia"
    ModulePath     "/usr/lib/x86_64-linux-gnu/nvidia/xorg"
    Option         "AllowEmptyInitialConfiguration" "True"
    Option         "UseDisplayDevice" "none"
EndSection
XORGEOF
    Xorg :99 -configdir "$XORG_CONF_DIR" -logfile /tmp/xorg."$$".log &
    XORG_PID=$!
    sleep 2
    echo "[INFO] Started Xorg on :99 (PID: $XORG_PID)"
fi

cleanup() {
    if [ -n "$XORG_PID" ]; then
        kill "$XORG_PID" 2>/dev/null || true
        echo "[INFO] Stopped Xorg"
    fi
    if [ -n "${XORG_CONF_DIR:-}" ]; then
        rm -rf "$XORG_CONF_DIR" 2>/dev/null || true
    fi
}
trap cleanup EXIT

~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task FR3-Reach-v0 \
  --num_envs 8 \
  --max_iterations 100 \
  --headless
