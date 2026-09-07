#!/usr/bin/env bash
# ============================================================================
# captain_first_run.sh — FR3 first-run pipeline for the one-L40 no-GUI container.
#
# Stages (each gates the next on its exit code; no grep-based verdicts):
#   1. Semantic diagnostics:
#        - scripts/check_fr3_asset.py          (asset invariants, nonzero-exit)
#        - scripts/diag_fr3_observation_triangulation.py  (D1)
#        - scripts/diag_fr3_reward.py                    (D2)
#        - scripts/diag_fr3_pd_gravity.py                 (D3)
#   2. 8192-env throughput probe (rsl_rl/train.py, 1 iter, 8192 envs)
#   3. 50-iteration pilot (rsl_rl/train.py, 4096 envs)
#   4. Evaluator (zero baseline) on a fixed seed
#
# Environment:
#   ISAACLAB_PATH   default ~/IsaacLab   (the isaaclab.sh checkout)
#   NUM_ENVS_PROBE  default 8192
#   NUM_ENVS_PILOT  default 4096
#   PILOT_ITERS     default 50
#   EVAL_NUM_ENVS   default 16
#   EVAL_SEED       default 0
#
# Run on the container:
#   bash scripts/captain_first_run.sh
# ============================================================================
set -euo pipefail

ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/IsaacLab}"
ISAACLAB_SH="${ISAACLAB_PATH}/isaaclab.sh"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs/captain_first_run"
mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%d_%H%M%S)

NUM_ENVS_PROBE="${NUM_ENVS_PROBE:-8192}"
NUM_ENVS_PILOT="${NUM_ENVS_PILOT:-4096}"
PILOT_ITERS="${PILOT_ITERS:-50}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-16}"
EVAL_SEED="${EVAL_SEED:-0}"

fail() { echo "[FAIL] $*" >&2; exit 1; }
[ -f "${ISAACLAB_SH}" ] || fail "isaaclab.sh not found: ${ISAACLAB_SH} (set ISAACLAB_PATH)"

# Ensure the project is importable.
"${ISAACLAB_SH}" -p -c "import fr3_isaaclab_project" >/dev/null 2>&1 \
  || "${ISAACLAB_SH}" -p -m pip install -e "${PROJECT_ROOT}/source/fr3_isaaclab_project" >/dev/null

run() {  # run <label> <log_name> <cmd...>
  local label="$1"; shift
  local log="$1"; shift
  echo "========================================"
  echo "[STAGE] ${label}"
  echo "----------------------------------------"
  set +e; timeout --signal=TERM --kill-after=30 1800 "$@" >"${LOG_DIR}/${log}" 2>&1
  local rc=$?; set -e
  tail -n 20 "${LOG_DIR}/${log}"
  echo "[STAGE ${label}] exit=${rc}  log=${LOG_DIR}/${log}"
  [ "${rc}" -eq 0 ] || fail "stage '${label}' failed (exit ${rc}); see ${LOG_DIR}/${log}"
}

cd "${PROJECT_ROOT}"

# ---- 1. Semantic diagnostics ----
echo "######## 1. SEMANTIC DIAGNOSTICS ########"
run "check_fr3_asset"     "01_check_${TS}.log" \
  "${ISAACLAB_SH}" -p scripts/check_fr3_asset.py --headless
run "D1_observation"      "02_d1_obs_${TS}.log" \
  "${ISAACLAB_SH}" -p scripts/diag_fr3_observation_triangulation.py --headless
run "D2_reward"           "03_d2_reward_${TS}.log" \
  "${ISAACLAB_SH}" -p scripts/diag_fr3_reward.py --headless
run "D3_pd_gravity"       "04_d3_pdgrav_${TS}.log" \
  "${ISAACLAB_SH}" -p scripts/diag_fr3_pd_gravity.py --headless

# ---- 2. 8192-env throughput probe ----
echo "######## 2. THROUGHPUT PROBE (${NUM_ENVS_PROBE} envs, 1 iter) ########"
run "throughput_probe"    "05_probe_${TS}.log" \
  "${ISAACLAB_SH}" -p scripts/rsl_rl/train.py --task FR3-Reach-v0 --headless \
    --num_envs "${NUM_ENVS_PROBE}" --max_iterations 1

# ---- 3. 50-iteration pilot ----
echo "######## 3. PILOT (${NUM_ENVS_PILOT} envs, ${PILOT_ITERS} iters) ########"
run "pilot_train"         "06_pilot_${TS}.log" \
  "${ISAACLAB_SH}" -p scripts/rsl_rl/train.py --task FR3-Reach-v0 --headless \
    --num_envs "${NUM_ENVS_PILOT}" --max_iterations "${PILOT_ITERS}"

# ---- 4. Evaluator (zero baseline, fixed seed) ----
echo "######## 4. EVALUATOR (zero baseline, seed ${EVAL_SEED}) ########"
run "eval_zero"           "07_eval_zero_${TS}.log" \
  "${ISAACLAB_SH}" -p scripts/eval_fr3_reach.py --headless \
    --policy zero --num_envs "${EVAL_NUM_ENVS}" --seed "${EVAL_SEED}" --max_steps 100

echo "========================================"
echo "captain_first_run: ALL STAGES PASSED"
echo "logs: ${LOG_DIR}"
echo "========================================"
