#!/usr/bin/env bash
# ============================================================================
# run_record.sh — FR3 Isaac Lab trajectory recording & visualization.
#
# Runs record_fr3_trajectory.py to:
#   1. Start Isaac Sim headless
#   2. Create FR3-Reach-v0, run 50 simulation steps
#   3. Record joint positions, velocities, EE pose, rewards
#   4. Generate matplotlib PNG plots + JSON trajectory data
#   5. Write a summary report.md
#
# All output goes to outputs/record/record_<timestamp>/
#
# NOTE: No Vulkan/GPU rendering required — state-space only.
#       Video recording (gym.RecordVideo) requires Vulkan and is skipped.
# ============================================================================
set -euo pipefail

ISAACLAB_PATH="${ISAACLAB_PATH:-/public/home/s2353600/IsaacLab}"
ISAACLAB_SH="${ISAACLAB_PATH}/isaaclab.sh"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TIMEOUT_SEC=600   # 10 minutes hard limit
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/outputs/logs"
LOG_FILE="${LOG_DIR}/record_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

echo "============================================" | tee "${LOG_FILE}"
echo "FR3 Isaac Lab — RECORD (trajectory visualization)" | tee -a "${LOG_FILE}"
echo "Started:     $(date)" | tee -a "${LOG_FILE}"
echo "Timeout:     ${TIMEOUT_SEC}s (10 min)" | tee -a "${LOG_FILE}"
echo "Log:         ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "============================================" | tee -a "${LOG_FILE}"

cd "${PROJECT_ROOT}"

# ---- Fix library path ----
ISAACSIM_LIBS="/public/home/s2353600/.local/lib/python3.10/site-packages/isaacsim/extscache"
if [ -d "${ISAACSIM_LIBS}" ]; then
    EXTCACHE_PATHS=$(find "${ISAACSIM_LIBS}" -maxdepth 2 -type d -name "bin" 2>/dev/null | tr '\n' ':')
    if [ -n "${EXTCACHE_PATHS}" ]; then
        export LD_LIBRARY_PATH="${EXTCACHE_PATHS}${LD_LIBRARY_PATH:-}"
        echo "[INFO] Added extcache bin dirs to LD_LIBRARY_PATH" | tee -a "${LOG_FILE}"
    fi
fi

# ---- Check IsaacLab ----
if [ ! -f "${ISAACLAB_SH}" ]; then
    echo "[FATAL] IsaacLab script not found: ${ISAACLAB_SH}" | tee -a "${LOG_FILE}"
    exit 1
fi
echo "[OK] IsaacLab script found" | tee -a "${LOG_FILE}"

# ---- Check matplotlib ----
echo "" | tee -a "${LOG_FILE}"
echo "[INFO] Checking matplotlib..." | tee -a "${LOG_FILE}"
set +e
${ISAACLAB_SH} -p -c "import matplotlib; print(f'  matplotlib {matplotlib.__version__} OK')" 2>&1 | tee -a "${LOG_FILE}" || {
    echo "[WARN] matplotlib not available, installing..." | tee -a "${LOG_FILE}"
    ${ISAACLAB_SH} -p -m pip install matplotlib --quiet 2>&1 | tee -a "${LOG_FILE}"
}
set -e

# ===================================================================
# MAIN: record_fr3_trajectory.py
# ===================================================================
echo "" | tee -a "${LOG_FILE}"
echo "----------------------------------------" | tee -a "${LOG_FILE}"
echo "RECORD: record_fr3_trajectory.py (50 steps)" | tee -a "${LOG_FILE}"
echo "  Headless state-space trajectory recording" | tee -a "${LOG_FILE}"
echo "  No Vulkan/GPU rendering needed" | tee -a "${LOG_FILE}"
echo "----------------------------------------" | tee -a "${LOG_FILE}"

set +e
timeout --signal=TERM --kill-after=30 ${TIMEOUT_SEC} \
    ${ISAACLAB_SH} -p scripts/record_fr3_trajectory.py \
    2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}
set -e

# ===================================================================
# Check for output files
# ===================================================================
RECORD_DIR=$(ls -dt "${PROJECT_ROOT}/outputs/record"/record_* 2>/dev/null | head -1)

# ===================================================================
# DIAGNOSE
# ===================================================================
echo "" | tee -a "${LOG_FILE}"
echo "============================================" | tee -a "${LOG_FILE}"
echo "Exit code:   ${EXIT_CODE}" | tee -a "${LOG_FILE}"
echo "Finished:    $(date)" | tee -a "${LOG_FILE}"

# Check for errors in log
HAS_ERROR=false
if grep -qE "Traceback \(most recent call last\)|ImportError|ModuleNotFoundError|FATAL" "${LOG_FILE}"; then
    HAS_ERROR=true
fi

case ${EXIT_CODE} in
    0)
        if ${HAS_ERROR}; then
            echo "RESULT: FAILED (Python error but exit code 0)" | tee -a "${LOG_FILE}"
            EXIT_CODE=1
        else
            echo "RESULT: SUCCESS" | tee -a "${LOG_FILE}"
        fi
        echo "" | tee -a "${LOG_FILE}"

        # Check output artifacts
        if [ -n "${RECORD_DIR}" ]; then
            echo "Output dir:  ${RECORD_DIR}" | tee -a "${LOG_FILE}"
            echo "" | tee -a "${LOG_FILE}"
            echo "=== Generated files ===" | tee -a "${LOG_FILE}"
            find "${RECORD_DIR}" -type f | sort | while read f; do
                size=$(stat -c%s "$f" 2>/dev/null || echo "?")
                echo "  $(basename "$f") (${size} bytes)" | tee -a "${LOG_FILE}"
            done
        fi

        # Check for videos
        VIDEO_COUNT=$(find "${RECORD_DIR:-/dev/null}" -name "*.mp4" 2>/dev/null | wc -l)
        GIF_COUNT=$(find "${RECORD_DIR:-/dev/null}" -name "*.gif" 2>/dev/null | wc -l)
        PNG_COUNT=$(find "${RECORD_DIR:-/dev/null}" -name "*.png" 2>/dev/null | wc -l)
        JSON_COUNT=$(find "${RECORD_DIR:-/dev/null}" -name "*.json" 2>/dev/null | wc -l)
        MD_COUNT=$(find "${RECORD_DIR:-/dev/null}" -name "*.md" 2>/dev/null | wc -l)

        echo "" | tee -a "${LOG_FILE}"
        echo "=== Artifact summary ===" | tee -a "${LOG_FILE}"
        echo "  MP4 videos:  ${VIDEO_COUNT}" | tee -a "${LOG_FILE}"
        echo "  GIF anims:   ${GIF_COUNT}" | tee -a "${LOG_FILE}"
        echo "  PNG plots:   ${PNG_COUNT}" | tee -a "${LOG_FILE}"
        echo "  JSON data:   ${JSON_COUNT}" | tee -a "${LOG_FILE}"
        echo "  Reports:     ${MD_COUNT}" | tee -a "${LOG_FILE}"

        if [ "${PNG_COUNT}" -gt 0 ] || [ "${JSON_COUNT}" -gt 0 ]; then
            echo "" | tee -a "${LOG_FILE}"
            echo "VERDICT: Record phase PASSED — state-space trajectory data captured." | tee -a "${LOG_FILE}"
            if [ "${VIDEO_COUNT}" -eq 0 ]; then
                echo "NOTE: No MP4 video (Vulkan GPU rendering unavailable in this container)." | tee -a "${LOG_FILE}"
                echo "      PNG plots + JSON trajectory provide demo-ready state visualization." | tee -a "${LOG_FILE}"
            fi
        fi

        # Stage diagnosis
        echo "" | tee -a "${LOG_FILE}"
        echo "=== Stage check ===" | tee -a "${LOG_FILE}"
        grep -q "Env created" "${LOG_FILE}" && echo "  [OK] Environment created" | tee -a "${LOG_FILE}" || echo "  [FAIL] Environment creation" | tee -a "${LOG_FILE}"
        grep -q "Env reset" "${LOG_FILE}" && echo "  [OK] Environment reset" | tee -a "${LOG_FILE}" || echo "  [FAIL] Environment reset" | tee -a "${LOG_FILE}"
        grep -q "Completed.*steps" "${LOG_FILE}" && echo "  [OK] Simulation stepping" | tee -a "${LOG_FILE}" || echo "  [FAIL] Simulation stepping" | tee -a "${LOG_FILE}"
        grep -q "trajectory.json" "${LOG_FILE}" && echo "  [OK] JSON trajectory saved" | tee -a "${LOG_FILE}" || echo "  [FAIL] JSON trajectory" | tee -a "${LOG_FILE}"
        grep -q "joint_positions.png" "${LOG_FILE}" && echo "  [OK] Joint position plot" | tee -a "${LOG_FILE}" || echo "  [FAIL] Joint position plot" | tee -a "${LOG_FILE}"
        grep -q "ee_pose_and_actions.png" "${LOG_FILE}" && echo "  [OK] EE pose plot" | tee -a "${LOG_FILE}" || echo "  [FAIL] EE pose plot" | tee -a "${LOG_FILE}"
        grep -q "rewards.png" "${LOG_FILE}" && echo "  [OK] Reward plot" | tee -a "${LOG_FILE}" || echo "  [FAIL] Reward plot" | tee -a "${LOG_FILE}"
        grep -q "record_report.md" "${LOG_FILE}" && echo "  [OK] Report generated" | tee -a "${LOG_FILE}" || echo "  [FAIL] Report generation" | tee -a "${LOG_FILE}"
        ;;
    124)
        echo "RESULT: TIMEOUT (>${TIMEOUT_SEC}s)" | tee -a "${LOG_FILE}"
        echo "" | tee -a "${LOG_FILE}"
        echo "=== Last 100 lines ===" | tee -a "${LOG_FILE}"
        tail -100 "${LOG_FILE}"
        echo "" | tee -a "${LOG_FILE}"
        echo "=== Stage diagnosis ===" | tee -a "${LOG_FILE}"
        grep -q "AppLauncher" "${LOG_FILE}" && echo "STAGE: AppLauncher started" | tee -a "${LOG_FILE}" || echo "STAGE: STUCK at AppLauncher" | tee -a "${LOG_FILE}"
        grep -q "Env created" "${LOG_FILE}" && echo "STAGE: Env created" | tee -a "${LOG_FILE}" || echo "STAGE: STUCK at env creation" | tee -a "${LOG_FILE}"
        grep -q "Completed.*steps" "${LOG_FILE}" && echo "STAGE: Steps completed" | tee -a "${LOG_FILE}" || echo "STAGE: STUCK during stepping" | tee -a "${LOG_FILE}"
        grep -q "Saved:" "${LOG_FILE}" && echo "STAGE: Plot generation done" | tee -a "${LOG_FILE}" || echo "STAGE: STUCK at plot generation" | tee -a "${LOG_FILE}"
        ;;
    137)
        echo "RESULT: KILLED (force-kill after grace period)" | tee -a "${LOG_FILE}"
        tail -100 "${LOG_FILE}"
        ;;
    *)
        echo "RESULT: FAILED (exit code ${EXIT_CODE})" | tee -a "${LOG_FILE}"
        echo "" | tee -a "${LOG_FILE}"
        echo "=== Last 100 lines ===" | tee -a "${LOG_FILE}"
        tail -100 "${LOG_FILE}"
        ;;
esac

echo "" | tee -a "${LOG_FILE}"
echo "Full log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "Record dir: ${RECORD_DIR:-none}" | tee -a "${LOG_FILE}"
echo "============================================" | tee -a "${LOG_FILE}"

exit ${EXIT_CODE}
