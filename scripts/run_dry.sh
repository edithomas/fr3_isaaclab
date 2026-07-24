#!/usr/bin/env bash
# ============================================================================
# run_dry.sh — FR3 Isaac Lab minimal dry-run validation.
#
# Runs list_envs.py --headless to verify:
#   1. Isaac Sim AppLauncher starts correctly
#   2. GPU/CUDA/PhysX is accessible
#   3. FR3 environments are gym-registered
#
# This is the FASTEST end-to-end validation (no simulation stepping).
#
# VALIDATION MODE:
#   - 5-minute hard timeout (kill & save tail-log if stuck)
#   - Headless only, no GUI, no real VLM
#   - No training / no simulation steps
# ============================================================================
set -euo pipefail

ISAACLAB_PATH="${ISAACLAB_PATH:-/public/home/s2353600/IsaacLab}"
ISAACLAB_SH="${ISAACLAB_PATH}/isaaclab.sh"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TIMEOUT_SEC=300   # 5 minutes hard limit
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/dry_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

echo "============================================" | tee "${LOG_FILE}"
echo "FR3 Isaac Lab — DRY RUN (minimal validation)" | tee -a "${LOG_FILE}"
echo "Started:     $(date)" | tee -a "${LOG_FILE}"
echo "Timeout:     ${TIMEOUT_SEC}s (5 min)" | tee -a "${LOG_FILE}"
echo "Log:         ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "Project:     ${PROJECT_ROOT}" | tee -a "${LOG_FILE}"
echo "IsaacLab:    ${ISAACLAB_PATH}" | tee -a "${LOG_FILE}"
echo "============================================" | tee -a "${LOG_FILE}"

# ---- Fix library path for Isaac Sim shared libs ----
# Isaac Sim bundles libhdx.so etc. in extscache but doesn't always add them to LD_LIBRARY_PATH
ISAACSIM_LIBS="/public/home/s2353600/.local/lib/python3.10/site-packages/isaacsim/extscache"
if [ -d "${ISAACSIM_LIBS}" ]; then
    # Collect all extcache bin directories
    EXTCACHE_PATHS=$(find "${ISAACSIM_LIBS}" -maxdepth 2 -type d -name "bin" 2>/dev/null | tr '\n' ':')
    if [ -n "${EXTCACHE_PATHS}" ]; then
        export LD_LIBRARY_PATH="${EXTCACHE_PATHS}${LD_LIBRARY_PATH:-}"
        echo "[INFO] Added extcache bin dirs to LD_LIBRARY_PATH" | tee -a "${LOG_FILE}"
    fi
fi

# ---- Check IsaacLab exists ----
if [ ! -f "${ISAACLAB_SH}" ]; then
    echo "[FATAL] IsaacLab script not found: ${ISAACLAB_SH}" | tee -a "${LOG_FILE}"
    echo "Set ISAACLAB_PATH or install IsaacLab." | tee -a "${LOG_FILE}"
    exit 1
fi
echo "[OK] IsaacLab script found" | tee -a "${LOG_FILE}"

# ---- Check project installed ----
echo "" | tee -a "${LOG_FILE}"
echo "[INFO] Checking project installation..." | tee -a "${LOG_FILE}"
cd "${PROJECT_ROOT}"
if ! ${ISAACLAB_SH} -p -c "import fr3_isaaclab_project" 2>&1 | tee -a "${LOG_FILE}"; then
    echo "[WARN] Project not installed. Installing..." | tee -a "${LOG_FILE}"
    ${ISAACLAB_SH} -p -m pip install -e source/fr3_isaaclab_project --quiet 2>&1 | tee -a "${LOG_FILE}"
fi

# ---- Check CUDA (non-fatal) ----
echo "" | tee -a "${LOG_FILE}"
echo "[INFO] Checking CUDA..." | tee -a "${LOG_FILE}"
set +e
${ISAACLAB_SH} -p -c "
import torch
print(f'  PyTorch:        {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  Device:         {torch.cuda.get_device_name(0)}')
    print(f'  Memory:         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  [WARN] CUDA NOT available — simulation may fail')
" 2>&1 | tee -a "${LOG_FILE}" || echo "[WARN] CUDA check had issues (non-fatal)" | tee -a "${LOG_FILE}"
set -e

# ===================================================================
# MAIN DRY RUN: list_envs.py --headless
# ===================================================================
echo "" | tee -a "${LOG_FILE}"
echo "----------------------------------------" | tee -a "${LOG_FILE}"
echo "DRY RUN: list_envs.py --headless" | tee -a "${LOG_FILE}"
echo "----------------------------------------" | tee -a "${LOG_FILE}"

set +e  # Don't exit on timeout signal
timeout --signal=TERM --kill-after=30 ${TIMEOUT_SEC} \
    ${ISAACLAB_SH} -p scripts/list_envs.py --headless \
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

# Check log for Python tracebacks even if exit code is 0 (isaaclab.sh may swallow errors)
HAS_PYTHON_ERROR=false
if grep -qE "Traceback \(most recent call last\)|ImportError|ModuleNotFoundError|libhdx\.so" "${LOG_FILE}"; then
    HAS_PYTHON_ERROR=true
fi

case ${EXIT_CODE} in
    0)
        if ${HAS_PYTHON_ERROR}; then
            echo "RESULT: FAILED (Python error but exit code 0 — isaaclab.sh swallowed error)" | tee -a "${LOG_FILE}"
            EXIT_CODE=1
        else
            echo "RESULT: SUCCESS" | tee -a "${LOG_FILE}"
        fi
        echo "" | tee -a "${LOG_FILE}"
        # Verify FR3 envs are actually listed
        if grep -q "FR3-Reach" "${LOG_FILE}"; then
            echo "VALIDATION: FR3-Reach-v0 environment registered OK" | tee -a "${LOG_FILE}"
        elif ${HAS_PYTHON_ERROR}; then
            echo "VALIDATION: FAILED — Python error prevented FR3 env registration" | tee -a "${LOG_FILE}"
        else
            echo "WARNING: No FR3-Reach env found in output — check registration" | tee -a "${LOG_FILE}"
        fi
        ;;
    124)
        echo "RESULT: TIMEOUT (>${TIMEOUT_SEC}s)" | tee -a "${LOG_FILE}"
        echo "" | tee -a "${LOG_FILE}"
        echo "=== Last 100 lines ===" | tee -a "${LOG_FILE}"
        tail -100 "${LOG_FILE}"
        echo "" | tee -a "${LOG_FILE}"
        echo "=== Stage diagnosis ===" | tee -a "${LOG_FILE}"
        if grep -q "Isaac Sim" "${LOG_FILE}"; then
            echo "STAGE: Isaac Sim started" | tee -a "${LOG_FILE}"
        else
            echo "STAGE: STUCK at Isaac Sim AppLauncher startup" | tee -a "${LOG_FILE}"
        fi
        if grep -q "FR3-Reach" "${LOG_FILE}"; then
            echo "STAGE: FR3 envs registered" | tee -a "${LOG_FILE}"
        else
            echo "STAGE: STUCK before FR3 env registration" | tee -a "${LOG_FILE}"
        fi
        if grep -q "Simulation App" "${LOG_FILE}"; then
            echo "STAGE: SimulationApp created" | tee -a "${LOG_FILE}"
        fi
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
echo "============================================" | tee -a "${LOG_FILE}"

exit ${EXIT_CODE}
