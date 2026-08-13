# FR3 Experimental Tooling

Headless diagnostics, recorder, and evaluator for the FR3-Reach-v0 task, built
against **Isaac Lab v2.1.0**, **Isaac Sim 4.5.0**, **RSL-RL 2.3.1**. All runtime
scripts launch via `~/IsaacLab/isaaclab.sh -p <script> --headless`.

## Scripts

| Script | Purpose | Exit-code contract |
|---|---|---|
| `scripts/check_fr3_asset.py` | FR3 USD invariants (fr3_hand, arm joints, rigid bodies) | 0 pass / 3 asset-resolve / 4 stage-open / 5 invariant / 6 unexpected |
| `scripts/diag_fr3_observation_triangulation.py` | **D1** — policy obs == `joint_pos - default` | 0 verified / 7 inconclusive / 6 unexpected |
| `scripts/diag_fr3_reward.py` | **D2** — `net == Σ(get_active_iterable_terms)·dt` | 0 verified / 7 violated / 6 unexpected |
| `scripts/diag_fr3_pd_gravity.py` | **D3** — PD/gravity A/B/C evidence | 0 evidence / 7 inconclusive / 6 unexpected |
| `scripts/record_fr3_trajectory.py` | State-space recorder (absolute + relative + actual EE) | 0 ok / 6 unexpected |
| `scripts/eval_fr3_reach.py` | Finite-step evaluator (zero/random/checkpoint) | 0 ok / 7 inconclusive / 6 unexpected |
| `scripts/generate_fr3_usd.py` | Local FR3 USD from URDF (when Nucleus unreachable) | 0 ok / 1 failure / 6 unexpected |
| `scripts/captain_first_run.sh` | diagnostics → 8192 probe → 50-iter pilot → evaluator | 0 all stages passed |

## Pure-logic test harness (runs locally without Isaac Sim)

```
source/fr3_isaaclab_project/fr3_isaaclab_project/diagnostics/   # no Isaac Sim imports
tests/                                                          # pytest
```

Run locally (macOS/CI, no GPU):

```bash
PYTHONPATH=source/fr3_isaaclab_project python3 -m pytest tests/ -q
```

`tests/test_check_fr3_asset_exit_codes.py` and `tests/test_close_timeout_no_mask.py`
run the **actual** scripts under a stub `isaaclab` package + `usd-core` (`pxr`),
so the full exit-code contract — including the `IsA→HasAPI` fix this PR preserves
from `origin/fix/fr3-asset-check` (5a261f7) — is exercised without a GPU.

Runtime tests (`@pytest.mark.runtime`) need the container:

```bash
RUN_FR3_RUNTIME_TESTS=1 ~/IsaacLab/isaaclab.sh -p -m pytest tests/ -q
```

## Corrected semantics (this PR)

1. **Zero joint-position action is NOT zero torque / free fall.**
   `JointPositionAction(use_default_offset=True, scale=0.5)` sets the position
   target to `default_joint_pos`, held by implicit PD (`stiffness=80,
   damping=4`). D3 proves this: PD-on legs (A/B) hold; PD-off (C) collapses.
2. **Recorded `joint_pos` observation is relative.** `mdp.joint_pos_rel` returns
   `joint_pos - default_joint_pos` (±0.01 noise). The prior F3/H7/P0-1 "runtime
   default mismatch" is disproved; D1 verifies the relative reconstruction.
3. **`info["log"]` is NOT per-step reward decomposition.** It is the per-episode
   logging dict populated only on reset steps (`Episode_Reward/<name>`). Per-step
   terms come from `reward_manager.get_active_iterable_terms` (rate form,
   `func*weight`); the net-reward convention is `net == Σ(rates)·step_dt` (D2).

## Official source (version-pinned)

* Isaac Lab v2.1.0 tag — `https://github.com/isaac-sim/IsaacLab/tree/v2.1.0`
  * RewardManager: `source/isaaclab/isaaclab/managers/reward_manager.py`
    — `https://github.com/isaac-sim/IsaacLab/blob/v2.1.0/source/isaaclab/isaaclab/managers/reward_manager.py`
  * ManagerBasedRLEnv.step (5-tuple return, `_reset_idx` populates `extras["log"]`):
    `source/isaaclab/isaaclab/envs/manager_based_rl_env.py`
  * joint_pos_rel / joint_vel_rel / generated_commands / last_action:
    `source/isaaclab/isaaclab/envs/mdp/observations.py`
  * JointPositionAction (use_default_offset, scale, set_joint_position_target):
    `source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py`
  * ArticulationData (joint_pos, default_joint_pos, body_state_w, body_pos_w):
    `source/isaaclab/isaaclab/assets/articulation/articulation_data.py`
  * Reach env cfg (PolicyCfg: joint_pos_rel + Unoise ±0.01):
    `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/reach_env_cfg.py`
  * Reach reward funcs (combine_frame_transforms, quat_mul, quat_error_magnitude):
    `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/reach/mdp/rewards.py`
  * RSL-RL vec env wrapper:
    `source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py`
* RSL-RL v2.3.1 — `https://github.com/leggedrobotics/rsl_rl/tree/v2.3.1`
  * OnPolicyRunner.get_inference_policy / load:
    `rsl_rl/runners/on_policy_runner.py`

## Captains' workflow

See `scripts/captain_first_run.sh` (one-L40, no-GUI). Diagnostics write JSON to
`outputs/diagnostics/`; the evaluator writes `outputs/eval/eval.json`.
