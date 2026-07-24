#!/usr/bin/env bash
set -euo pipefail

cd ~/fr3_isaaclab_project

export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export LIVESTREAM=2

CHECKPOINT="$1"
if [ -z "$CHECKPOINT" ]; then
  echo "Usage: bash scripts/run_fr3_visual_play.sh /path/to/model.pt"
  echo ""
  echo "Available checkpoints:"
  find logs/rsl_rl -name "*.pt" 2>/dev/null | sort | tail -n 10 || echo "  (none found)"
  exit 1
fi

~/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task FR3-Reach-Play-v0 \
  --num_envs 8 \
  --checkpoint "$CHECKPOINT"
