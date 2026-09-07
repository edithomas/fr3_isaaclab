"""Reward integration convention for Isaac Lab v2.1.0 — pure Python.

Encodes the *net-reward integration convention* of
:class:`~isaaclab.managers.RewardManager` (Isaac Lab v2.1.0,
``source/isaaclab/isaaclab/managers/reward_manager.py``) so the headless reward
diagnostic and its local unit tests can verify it without a GPU:

* ``RewardManager.compute(dt)`` sets, for each active term ``i``::

      step_reward_i = func_i(env) * weight_i            # rate form (no dt)
      reward_buf   += step_reward_i * dt                # integrated

* :meth:`RewardManager.get_active_iterable_terms(env_idx)` returns ``[(name,
  [step_reward_i])]`` — the per-step term values in **rate form**, i.e. the
  weighted term value *before* the ``dt`` integration.

Therefore the convention the diagnostic must prove is::

      net_reward == sum(active_term_rates) * step_dt   (within float tolerance)

and ``info["log"]`` (``extras["log"]``) is **not** a per-step reward
decomposition: it is the per-episode logging dict populated only inside
``ManagerBasedRLEnv._reset_idx`` (``Episode_Reward/<name>`` = episodic sum /
``max_episode_length_s``). Treating ``info["log"]`` as per-step terms is the
regression this module exists to catch.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RewardConventionResult:
    """Result of verifying the net-reward integration convention."""

    net_reward: float
    reconstructed: float
    abs_err: float
    rel_err: float
    ok: bool


def reconstruct_net_reward(term_rates: list[float], step_dt: float) -> float:
    """Reconstruct the net reward from per-step term *rates*.

    ``term_rates`` are the values returned by
    ``RewardManager.get_active_iterable_terms`` (i.e. ``func*weight``, before
    ``dt``). Per the v2.1.0 source, ``reward_buf`` accumulates
    ``step_reward_i * dt`` for every term.
    """
    if step_dt <= 0.0:
        raise ValueError(f"step_dt must be > 0, got {step_dt}")
    return sum(float(r) for r in term_rates) * float(step_dt)


def verify_reward_convention(
    net_reward: float,
    term_rates: list[float],
    step_dt: float,
    *,
    abs_tol: float = 1e-5,
    rel_tol: float = 1e-4,
) -> RewardConventionResult:
    """Verify ``net_reward == sum(term_rates) * step_dt``.

    A failure here means the recorded "per-step reward terms" were *not* the
    rate-form values from ``get_active_iterable_terms`` — i.e. someone likely
    grabbed ``info["log"]`` (episode-level logging) and mislabeled it.
    """
    reconstructed = reconstruct_net_reward(term_rates, step_dt)
    abs_err = abs(net_reward - reconstructed)
    scale = max(abs(net_reward), abs(reconstructed), 1.0)
    rel_err = abs_err / scale
    ok = abs_err <= abs_tol or rel_err <= rel_tol
    return RewardConventionResult(
        net_reward=float(net_reward),
        reconstructed=float(reconstructed),
        abs_err=abs_err,
        rel_err=rel_err,
        ok=ok,
    )


# Documented names — used by docs/tests to assert the convention text.
INFO_LOG_IS_PER_STEP = False
"""``info['log']`` is episode-level logging, not per-step reward terms."""

REWARD_TERM_RATE_DOC = (
    "RewardManager.get_active_iterable_terms(env_idx) returns [(name, [rate])] "
    "where rate = func*weight (the dt-free weighted term value). Net reward = "
    "sum(rates) * step_dt."
)
