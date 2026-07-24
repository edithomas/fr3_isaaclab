#!/usr/bin/env bash
# ============================================================================
# make_video.sh — Generate offline FR3 trajectory animation from trajectory.json.
#
# Finds latest outputs/record/record_*/trajectory.json and creates a 4-panel
# animation (mp4/gif) using matplotlib only — no Isaac Sim, no Vulkan, no GUI.
# ============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_ROOT}/outputs/logs"
LOG_FILE="${LOG_DIR}/make_video_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

echo "============================================" | tee "${LOG_FILE}"
echo "FR3 Trajectory Video — Offline Animation Generator" | tee -a "${LOG_FILE}"
echo "Started: $(date)" | tee -a "${LOG_FILE}"
echo "Log:     ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "============================================" | tee -a "${LOG_FILE}"

cd "${PROJECT_ROOT}"

# ---- Find latest trajectory.json ----
LATEST_RECORD=$(ls -dt outputs/record/record_*/ 2>/dev/null | head -1)
if [ -z "${LATEST_RECORD}" ]; then
    echo "[FATAL] No outputs/record/record_*/ directory found." | tee -a "${LOG_FILE}"
    echo "Run bash scripts/run_record.sh first to generate trajectory data." | tee -a "${LOG_FILE}"
    exit 1
fi

TRAJ_JSON="${LATEST_RECORD}/trajectory.json"
if [ ! -f "${TRAJ_JSON}" ]; then
    echo "[FATAL] trajectory.json not found in ${LATEST_RECORD}" | tee -a "${LOG_FILE}"
    exit 1
fi

echo "[INFO] Using data: ${TRAJ_JSON}" | tee -a "${LOG_FILE}"

# ---- Ensure matplotlib/imageio ----
echo "" | tee -a "${LOG_FILE}"
echo "[INFO] Checking Python packages..." | tee -a "${LOG_FILE}"
MISSING_PKGS=""
python3 -c "import matplotlib" 2>/dev/null || MISSING_PKGS="${MISSING_PKGS} matplotlib"
python3 -c "import imageio" 2>/dev/null || MISSING_PKGS="${MISSING_PKGS} imageio"

if [ -n "${MISSING_PKGS}" ]; then
    echo "[INFO] Installing:${MISSING_PKGS}" | tee -a "${LOG_FILE}"
    pip install --quiet${MISSING_PKGS} 2>&1 | tee -a "${LOG_FILE}" || {
        echo "[WARN] pip install failed, continuing with what's available..." | tee -a "${LOG_FILE}"
    }
fi

# ---- Output dir ----
VIDEO_OUT="${PROJECT_ROOT}/outputs/videos"
mkdir -p "${VIDEO_OUT}"

# ---- Run video generator ----
echo "" | tee -a "${LOG_FILE}"
echo "----------------------------------------" | tee -a "${LOG_FILE}"
echo "Generating animation..." | tee -a "${LOG_FILE}"
echo "----------------------------------------" | tee -a "${LOG_FILE}"

# Use system Python (not IsaacLab) since we only need matplotlib, no Isaac Sim
set +e
python3 scripts/make_trajectory_video.py \
    --input "${TRAJ_JSON}" \
    --output "${VIDEO_OUT}/fr3_trajectory_${TIMESTAMP}.mp4" \
    --fps 8 \
    --dpi 120 \
    2>&1 | tee -a "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}
set -e

# ---- Check output ----
echo "" | tee -a "${LOG_FILE}"
echo "============================================" | tee -a "${LOG_FILE}"
echo "Exit code: ${EXIT_CODE}" | tee -a "${LOG_FILE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "RESULT: SUCCESS" | tee -a "${LOG_FILE}"

    # Find generated files
    NEWEST_VIDEO=$(ls -t "${VIDEO_OUT}"/fr3_trajectory_*.mp4 "${VIDEO_OUT}"/fr3_trajectory_*.gif 2>/dev/null | head -1)
    FRAMES_DIR="${VIDEO_OUT}/frames"

    echo "" | tee -a "${LOG_FILE}"
    echo "=== Output artifacts ===" | tee -a "${LOG_FILE}"

    if [ -n "${NEWEST_VIDEO}" ] && [ -f "${NEWEST_VIDEO}" ]; then
        SIZE=$(stat -c%s "${NEWEST_VIDEO}" 2>/dev/null || echo "?")
        SIZE_MB=$(echo "scale=1; ${SIZE}/1048576" | bc 2>/dev/null || echo "?")
        echo "  Video/GIF: ${NEWEST_VIDEO} (${SIZE_MB} MB)" | tee -a "${LOG_FILE}"
    fi

    if [ -d "${FRAMES_DIR}" ]; then
        FRAME_COUNT=$(find "${FRAMES_DIR}" -name "*.png" 2>/dev/null | wc -l)
        echo "  Frames:    ${FRAMES_DIR}/ (${FRAME_COUNT} PNGs)" | tee -a "${LOG_FILE}"
    fi

    VIDEO_REPORT=$(ls -t "${VIDEO_OUT}"/video_report.md 2>/dev/null | head -1)
    if [ -n "${VIDEO_REPORT}" ]; then
        echo "  Report:    ${VIDEO_REPORT}" | tee -a "${LOG_FILE}"
    fi

    echo "" | tee -a "${LOG_FILE}"
    echo "VERDICT: Trajectory animation generated successfully." | tee -a "${LOG_FILE}"
    echo "  This is an OFFLINE animation from state data — NO Vulkan/GPU rendering needed." | tee -a "${LOG_FILE}"
else
    echo "RESULT: FAILED (exit code ${EXIT_CODE})" | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"
    echo "=== Last 50 lines ===" | tee -a "${LOG_FILE}"
    tail -50 "${LOG_FILE}" | tee -a "${LOG_FILE}"
fi

echo "" | tee -a "${LOG_FILE}"
echo "Full log: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "Data source: ${TRAJ_JSON}" | tee -a "${LOG_FILE}"
echo "============================================" | tee -a "${LOG_FILE}"

exit ${EXIT_CODE}
