#!/usr/bin/env bash
# ============================================================================
# run_headless.sh — FR3 Isaac Lab minimal headless physics stepping validation.
#
# Runs rsl_rl/train.py with 1 env × 1 iteration to verify:
#   1. Isaac Sim SimulationApp starts
#   2. FR3-Reach-v0 env is created via gym.make()
#   3. Physics simulation stepping works (PhysX on CUDA)
#   4. One PPO training iteration completes
#   5. Summary/logs are written, env closes cleanly
#
# VALIDATION MODE (minimal):
#   - 8-minute hard timeout (kill & save tail-log if stuck)
#   - 1 env, 1 iteration, headless, no video
#   - No real VLM (this is RL training, no VLM involved)
# ============================================================================
set -euo pipefail

ISAACLAB_PATH="${ISAACLAB_PATH:-/public/home/s2353600/IsaacLab}"
ISAACLAB_SH="${ISAACLAB_PATH}/isaaclab.sh"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TIMEOUT_SEC=480   # 8 minutes hard limit
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/headless_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

echo "============================================" | tee "${LOG_FILE}"
echo "FR3 Isaac Lab — HEADLESS (physics stepping validation)" | tee -a "${LOG_FILE}"
echo "Started:     $(date)" | tee -a "${LOG_FILE}"
echo "Timeout:     ${TIMEOUT_SEC}s (8 min)" | tee -a "${LOG_FILE}"
echo "Log:         ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "============================================" | tee -a "${LOG_FILE}"

cd "${PROJECT_ROOT}"

# ---- Fix library path for Isaac Sim shared libs ----
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

# ---- Check project installed ----
echo "" | tee -a "${LOG_FILE}"
echo "[INFO] Checking project installation..." | tee -a "${LOG_FILE}"
set +e
if ! ${ISAACLAB_SH} -p -c "import fr3_isaaclab_project" 2>&1 | tee -a "${LOG_FILE}"; then
    echo "[WARN] Project not installed. Installing..." | tee -a "${LOG_FILE}"
    ${ISAACLAB_SH} -p -m pip install -e source/fr3_isaaclab_project --quiet 2>&1 | tee -a "${LOG_FILE}"
fi
set -e

# ===================================================================
# MAIN: rsl_rl/train.py with minimal params
# ===================================================================
echo "" | tee -a "${LOG_FILE}"
echo "----------------------------------------" | tee -a "${LOG_FILE}"
echo "HEADLESS: rsl_rl/train.py (1 env × 1 iter)" | tee -a "${LOG_FILE}"
echo "  --task FR3-Reach-v0" | tee -a "${LOG_FILE}"
echo "  --headless" | tee -a "${LOG_FILE}"
echo "  --num_envs 1" | tee -a "${LOG_FILE}"
echo "  --max_iterations 1" | tee -a "${LOG_FILE}"
echo "----------------------------------------" | tee -a "${LOG_FILE}"

set +e  # Don't exit on timeout signal
timeout --signal=TERM --kill-after=30 ${TIMEOUT_SEC} \
    ${ISAACLAB_SH} -p scripts/rsl_rl/train.py \
        --task FR3-Reach-v0 \
        --headless \
        --num_envs 1 \
        --max_iterations 1 \
        "$@" \
    2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}
set -e

# ===================================================================
# DIAGNOSE
# ===================================================================
echo "" | tee -a "${LOG_FILE}"
echo "============================================" | tee -a "${LOG_FILE}"
echo "Exit code:   ${EXIT_CODE}" | tee -a "${LOG_FILE}"
echo "Finished:    $(date)" | tee -a "${LOG_FILE}"

# Check log for errors
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
        # Check key milestones
        if grep -q "AppLauncher.*Using device" "${LOG_FILE}"; then
            echo "STAGE 1: Isaac Sim AppLauncher started OK" | tee -a "${LOG_FILE}"
        else
            echo "STAGE 1: MISSING — AppLauncher didn't reach device init" | tee -a "${LOG_FILE}"
        fi
        if grep -qE "gym\.make|Creating environment|FR3-Reach" "${LOG_FILE}"; then
            echo "STAGE 2: FR3-Reach-v0 environment created OK" | tee -a "${LOG_FILE}"
        else
            echo "STAGE 2: MISSING — env creation not confirmed in log" | tee -a "${LOG_FILE}"
        fi
        if grep -qE "Learning iteration|iteration.*1/1|learn\(\)" "${LOG_FILE}"; then
            echo "STAGE 3: Training iteration completed" | tee -a "${LOG_FILE}"
        fi
        if grep -qE "Logging experiment|log_root_path|logs/rsl_rl" "${LOG_FILE}"; then
            echo "STAGE 4: Training log directory created" | tee -a "${LOG_FILE}"
        fi
        if grep -qE "Exact experiment name" "${LOG_FILE}"; then
            echo "STAGE 5: Experiment name printed" | tee -a "${LOG_FILE}"
        fi
        if grep -qE "env\.close|simulation_app" "${LOG_FILE}"; then
            echo "STAGE 6: Cleanup initiated" | tee -a "${LOG_FILE}"
        fi
        ;;
    124)
        echo "RESULT: TIMEOUT (>${TIMEOUT_SEC}s)" | tee -a "${LOG_FILE}"
        echo "" | tee -a "${LOG_FILE}"
        echo "=== Last 100 lines ===" | tee -a "${LOG_FILE}"
        tail -100 "${LOG_FILE}"
        echo "" | tee -a "${LOG_FILE}"
        echo "=== Stage diagnosis ===" | tee -a "${LOG_FILE}"
        grep -q "AppLauncher" "${LOG_FILE}" && echo "STAGE: AppLauncher started" | tee -a "${LOG_FILE}" || echo "STAGE: STUCK at AppLauncher startup" | tee -a "${LOG_FILE}"
        grep -q "FR3-Reach" "${LOG_FILE}" && echo "STAGE: FR3 env reached" | tee -a "${LOG_FILE}" || echo "STAGE: STUCK before env creation" | tee -a "${LOG_FILE}"
        grep -q "gym.make\|Creating environment" "${LOG_FILE}" && echo "STAGE: env creation started" | tee -a "${LOG_FILE}"
        grep -q "iteration\|learn" "${LOG_FILE}" && echo "STAGE: training started" | tee -a "${LOG_FILE}" || echo "STAGE: STUCK before training iteration" | tee -a "${LOG_FILE}"
        ;;
    137)
        echo "RESULT: KILLED (force-kill after grace period)" | tee -a "${LOG_FILE}"
        tail -100 "${LOG_FILE}"
        ;;
    *)
        echo "RESULT: FAILED (exit code ${EXIT_CODE})" | tee -a "${LOG_FILE}"
        echo "" | tee -a "${LOG_FILE}"
        echo "=== Stage diagnosis ===" | tee -a "${LOG_FILE}"
        grep -q "AppLauncher" "${LOG_FILE}" && echo "STAGE: AppLauncher started" | tee -a "${LOG_FILE}" || echo "STAGE: STUCK before AppLauncher" | tee -a "${LOG_FILE}"
        grep -q "FR3-Reach" "${LOG_FILE}" && echo "STAGE: FR3 env reached" | tee -a "${LOG_FILE}" || echo "STAGE: STUCK before env creation" | tee -a "${LOG_FILE}"
        grep -q "iteration\|learn" "${LOG_FILE}" && echo "STAGE: training started" | tee -a "${LOG_FILE}" || echo "STAGE: STUCK before training iteration" | tee -a "${LOG_FILE}"
        echo "" | tee -a "${LOG_FILE}"
        echo "=== Last 100 lines ===" | tee -a "${LOG_FILE}"
        tail -100 "${LOG_FILE}"
        ;;
esac

echo "" | tee -a "${LOG_FILE}"
echo "Full log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "============================================" | tee -a "${LOG_FILE}"

exit ${EXIT_CODE}
