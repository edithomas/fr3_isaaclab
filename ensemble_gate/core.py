"""Offline ensemble-disagreement safety-gate metrics for FR3 sim-to-sim research.

WARNING -- RESEARCH INSTRUMENT, NOT A SAFETY GUARANTEE
=====================================================
This module is an **offline sim-to-sim research instrument**.  It consumes
recorded ensemble action trajectories from simulation rollouts and reports
threshold-based gating statistics (ensemble disagreement, tie-correct AUROC,
failure recall / false-abort / coverage, confusion counts, per-domain
metrics).

It is **NOT** a real-world safety guarantee, it performs **no** runtime
monitoring, and it is **NOT** a sim-to-real validation.  A good score on this
instrument does not mean a policy is safe to deploy on hardware.  All
decisions derived from its output must be reviewed by a qualified engineer
before any deployment.

Design constraints honoured by this module
-------------------------------------------
* Pure standard-library Python (no NumPy / SciPy / scikit-learn / Isaac Sim).
* Reads a documented JSONL episode format (see ``docs/ensemble_gate.md``).
* Threshold is selected **only** on the calibration-ID subset, under an
  ID false-abort ceiling, maximizing failure recall with deterministic
  tie-breaking; evaluation metrics are computed with that frozen threshold
  (no re-tuning on held-out data).
* ECE is reported **only** when calibrated probabilities are supplied in the
  input (no fitting is performed, so no calibration leakage).
* Output is deterministic machine-readable JSON with schema version, source
  hash, parameters, counts, metrics, and explicit undefined-metric reasons.
"""
from __future__ import annotations

import hashlib
import json
import math
from json import JSONDecodeError
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SCHEMA_VERSION = "1.0.0"
TOOL_NAME = "ensemble_gate_metrics"
TOOL_VERSION = "1.0.0"

DEFAULT_CEILING = 0.1          # 10% ID false-abort ceiling on calibration
DEFAULT_DDOF = 1               # sample variance (denominator P-1)
DEFAULT_ECE_BINS = 10
SPLITS = ("calibration", "evaluation")
ID_DOMAIN = "id"
REQUIRED_FIELDS = ("episode_id", "split", "domain", "failed", "actions")

# Label convention: the positive class is a FAILURE (failed == True -> 1).
# "Aborted" means the gate flagged the episode (score >= threshold).
LABEL_FAILURE = 1
LABEL_SUCCESS = 0

# A small epsilon for the false-abort feasibility test, to avoid rejecting a
# false-abort rate that is arithmetically equal to the ceiling but differs by
# floating-point rounding (e.g. 1/10 vs the ceiling 0.1).
_FEASIBILITY_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #

class Episode:
    """A validated rollout episode."""

    __slots__ = (
        "episode_id", "split", "domain", "failed",
        "actions", "probability", "lineno",
    )

    def __init__(
        self,
        episode_id: Any,
        split: str,
        domain: str,
        failed: bool,
        actions: Dict[str, List[List[float]]],
        probability: Optional[float],
        lineno: int,
    ) -> None:
        self.episode_id = episode_id
        self.split = split
        self.domain = domain
        self.failed = failed
        self.actions = actions
        self.probability = probability
        self.lineno = lineno

    @property
    def label(self) -> int:
        return LABEL_FAILURE if self.failed else LABEL_SUCCESS


class SourceMeta:
    """Provenance information about the input file."""

    __slots__ = (
        "file", "file_sha256", "byte_size",
        "total_lines", "episode_count", "policy_set", "action_dim",
    )

    def __init__(
        self,
        file: str,
        file_sha256: str,
        byte_size: int,
        total_lines: int,
        episode_count: int,
        policy_set: Optional[List[str]],
        action_dim: Optional[int],
    ) -> None:
        self.file = file
        self.file_sha256 = file_sha256
        self.byte_size = byte_size
        self.total_lines = total_lines
        self.episode_count = episode_count
        self.policy_set = policy_set
        self.action_dim = action_dim

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "file_sha256": self.file_sha256,
            "byte_size": self.byte_size,
            "total_lines": self.total_lines,
            "episode_count": self.episode_count,
            "policy_set": self.policy_set,
            "action_dim": self.action_dim,
        }


# --------------------------------------------------------------------------- #
# Loading + structural validation
# --------------------------------------------------------------------------- #

def load_episodes(
    path: str,
) -> Tuple[List[Episode], Optional[SourceMeta], List[str]]:
    """Read a JSONL episode file, parse and structurally validate it.

    Returns ``(episodes, source_meta, errors)``.  When the file cannot be
    read at all, ``source_meta`` is ``None``.  Structural validation errors
    are collected per-line and returned in ``errors``; the caller decides
    whether to abort (malformed evidence -> non-zero exit).
    """
    errors: List[str] = []
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        return [], None, [f"cannot read input file {path!r}: {exc}"]

    sha = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    total_lines = len(lines)

    episodes: List[Episode] = []
    seen_ids: set = set()
    policy_set: Optional[frozenset] = None
    action_dim: Optional[int] = None

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except JSONDecodeError as exc:
            errors.append(f"line {i}: invalid JSON: {exc.msg}")
            continue
        if not isinstance(obj, dict):
            errors.append(f"line {i}: JSON value is not an object")
            continue
        ep, rec_errors = _parse_record(obj, i, policy_set, action_dim, seen_ids)
        errors.extend(rec_errors)
        if ep is not None:
            episodes.append(ep)
            if policy_set is None:
                policy_set = frozenset(ep.actions.keys())
                # first vector of first policy (guaranteed non-empty by validation)
                first_pol = next(iter(ep.actions.keys()))
                action_dim = len(ep.actions[first_pol][0])

    source = SourceMeta(
        file=path,
        file_sha256=sha,
        byte_size=len(raw),
        total_lines=total_lines,
        episode_count=len(episodes),
        policy_set=sorted(policy_set) if policy_set else None,
        action_dim=action_dim,
    )
    return episodes, source, errors


def _parse_record(
    obj: dict,
    lineno: int,
    expected_pol_set: Optional[frozenset],
    expected_dim: Optional[int],
    seen_ids: set,
) -> Tuple[Optional[Episode], List[str]]:
    """Validate a single JSON object into an :class:`Episode` (or errors)."""
    errs: List[str] = []

    for field in REQUIRED_FIELDS:
        if field not in obj:
            errs.append(f"line {lineno}: missing required field {field!r}")
    if errs:
        return None, errs

    episode_id = obj["episode_id"]
    if isinstance(episode_id, bool) or not isinstance(episode_id, (str, int)):
        errs.append(f"line {lineno}: episode_id must be a string or integer")
    elif episode_id in seen_ids:
        errs.append(f"line {lineno}: duplicate episode_id {episode_id!r}")
    else:
        seen_ids.add(episode_id)

    split = obj["split"]
    if split not in SPLITS:
        errs.append(
            f"line {lineno}: split must be one of {list(SPLITS)}, got {split!r}"
        )

    domain = obj["domain"]
    if not isinstance(domain, str) or not domain.strip():
        errs.append(f"line {lineno}: domain must be a non-empty string")

    failed = obj["failed"]
    if not isinstance(failed, bool):
        errs.append(
            f"line {lineno}: failed must be a boolean, got {type(failed).__name__}"
        )

    actions = obj["actions"]
    if not isinstance(actions, dict):
        errs.append(
            f"line {lineno}: actions must be an object mapping policy -> trajectory"
        )
        actions = {}
    elif len(actions) < 2:
        errs.append(
            f"line {lineno}: actions must contain at least two policies, "
            f"got {len(actions)}"
        )

    # Optional calibrated probability (P(failure)).  If present it must be a
    # finite number in [0, 1]; otherwise it is a structural (hard) error.
    probability: Optional[float] = None
    if "probability" in obj:
        prob = obj["probability"]
        if isinstance(prob, bool) or not isinstance(prob, (int, float)):
            errs.append(f"line {lineno}: probability must be a number")
        elif not math.isfinite(prob) or prob < 0 or prob > 1:
            errs.append(
                f"line {lineno}: probability must be finite and in [0, 1], "
                f"got {prob}"
            )
        else:
            probability = float(prob)

    # --- actions structure + rectangularity + finiteness ------------------ #
    pol_T: Dict[str, int] = {}
    pol_D: Dict[str, int] = {}
    structurally_ok = True
    for pname, traj in actions.items():
        if not isinstance(traj, list) or len(traj) == 0:
            errs.append(
                f"line {lineno}: policy {pname!r} trajectory must be a "
                f"non-empty list"
            )
            structurally_ok = False
            continue
        dim0: Optional[int] = None
        for vec in traj:
            if not isinstance(vec, list) or len(vec) == 0:
                errs.append(
                    f"line {lineno}: policy {pname!r} has a non-list or empty "
                    f"action vector"
                )
                structurally_ok = False
                dim0 = None
                break
            if dim0 is None:
                dim0 = len(vec)
            elif len(vec) != dim0:
                errs.append(
                    f"line {lineno}: policy {pname!r} has ragged action "
                    f"vectors (dim {len(vec)} != {dim0})"
                )
                structurally_ok = False
                dim0 = None
                break
            for x in vec:
                if isinstance(x, bool) or not isinstance(x, (int, float)):
                    errs.append(
                        f"line {lineno}: policy {pname!r} has non-numeric "
                        f"action value {x!r}"
                    )
                    structurally_ok = False
                elif not math.isfinite(x):
                    errs.append(
                        f"line {lineno}: policy {pname!r} has non-finite "
                        f"action value {x!r}"
                    )
                    structurally_ok = False
        if dim0 is not None:
            pol_D[pname] = dim0
            pol_T[pname] = len(traj)

    if pol_T:
        ts = set(pol_T.values())
        ds = set(pol_D.values())
        if len(ts) > 1:
            errs.append(
                f"line {lineno}: policies disagree on trajectory length {pol_T}"
            )
        if len(ds) > 1:
            errs.append(
                f"line {lineno}: policies disagree on action dimension {pol_D}"
            )

    if not structurally_ok or errs:
        return None, errs

    dim = next(iter(pol_D.values()))
    cur_set = frozenset(actions.keys())
    if expected_pol_set is not None and cur_set != expected_pol_set:
        errs.append(
            f"line {lineno}: policy set {sorted(cur_set)} differs from earlier "
            f"episodes {sorted(expected_pol_set)}"
        )
    if expected_dim is not None and dim != expected_dim:
        errs.append(
            f"line {lineno}: action dimension {dim} differs from earlier "
            f"episodes ({expected_dim})"
        )
    if errs:
        return None, errs

    # Normalize actions to plain floats (ints -> floats for stable scoring).
    norm = {
        p: [[float(x) for x in vec] for vec in traj]
        for p, traj in actions.items()
    }
    return Episode(episode_id, split, domain, failed, norm, probability, lineno), []


# --------------------------------------------------------------------------- #
# Disagreement score
# --------------------------------------------------------------------------- #

def episode_disagreement(episode: Episode, ddof: int = DEFAULT_DDOF) -> float:
    """Predeclared episode disagreement score from ensemble action variance.

    For an episode with ``P`` policies, ``T`` timesteps and ``D`` action dims::

        for each (t, d):
            values   = [ a_policy(t, d) for each policy ]      # P values
            mean     = mean(values)
            var(t,d) = sum((v - mean)^2) / (P - ddof)          # sample variance
        per_timestep(t) = mean over d of var(t, d)             # average over dims
        score            = mean over t of per_timestep(t)      # average over time

    Aggregation choices (explicit in output provenance):
      * across policies: sample variance with ``ddof`` degrees of freedom
        (default ``ddof=1``, i.e. denominator ``P-1``);
      * across action dims: arithmetic mean;
      * across timesteps: arithmetic mean.

    Only the *ranking* of scores matters for AUROC and threshold selection, so
    a global monotone rescaling would not change the metrics; the chosen
    normalization is documented for reproducibility.
    """
    policies = list(episode.actions.keys())
    p_count = len(policies)
    traj0 = episode.actions[policies[0]]
    t_count = len(traj0)
    d_count = len(traj0[0])
    denom = p_count - ddof
    per_t: List[float] = []
    for t in range(t_count):
        var_sum = 0.0
        for d in range(d_count):
            vals = [episode.actions[p][t][d] for p in policies]
            mean = sum(vals) / p_count
            ss = sum((v - mean) ** 2 for v in vals)
            var = ss / denom if denom > 0 else 0.0
            var_sum += var
        per_t.append(var_sum / d_count)
    return sum(per_t) / t_count


# --------------------------------------------------------------------------- #
# Tie-correct AUROC (Mann-Whitney U with average ranks, no scipy/sklearn)
# --------------------------------------------------------------------------- #

def _average_ranks(values: List[float]) -> List[float]:
    """Return 1-indexed average ranks, with ties receiving the mean rank."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        # 1-indexed positions of the tie block are (i+1) .. (j+1)
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def auroc(scores: List[float], labels: List[int]) -> Dict[str, Any]:
    """Tie-correct AUROC (area under the ROC curve).

    ``labels`` use ``1`` = failure (positive), ``0`` = success (negative).
    Uses the Mann-Whitney U statistic with average ranks for ties, which is
    the standard trapezoidal (tie-correct) AUC.  Ties are detected by exact
    float equality of the disagreement scores.
    """
    n = len(scores)
    n_pos = sum(1 for y in labels if y == LABEL_FAILURE)
    n_neg = n - n_pos
    if n == 0:
        return _metric(False, reason="no records", n=0, n_pos=0, n_neg=0)
    if n_pos == 0:
        return _metric(
            False, reason="no positive (failure) examples", n=n, n_pos=0, n_neg=n_neg
        )
    if n_neg == 0:
        return _metric(
            False, reason="no negative (success) examples", n=n, n_pos=n_pos, n_neg=0
        )
    ranks = _average_ranks(scores)
    sum_pos = sum(ranks[i] for i in range(n) if labels[i] == LABEL_FAILURE)
    u = sum_pos - n_pos * (n_pos + 1) / 2.0
    auc = u / (n_pos * n_neg)
    return _metric(True, value=auc, reason=None, n=n, n_pos=n_pos, n_neg=n_neg)


# --------------------------------------------------------------------------- #
# Confusion / rates at a (frozen) threshold
# --------------------------------------------------------------------------- #

def confusion_at(
    scores: List[float], labels: List[int], threshold: float
) -> Dict[str, int]:
    """Confusion counts with abort = (score >= threshold), positive = failure."""
    tp = fp = tn = fn = 0
    for s, y in zip(scores, labels):
        aborted = s >= threshold
        if y == LABEL_FAILURE:
            if aborted:
                tp += 1
            else:
                fn += 1
        else:
            if aborted:
                fp += 1
            else:
                tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def rates_at(
    scores: List[float], labels: List[int], threshold: float
) -> Dict[str, Any]:
    """Failure recall, false-abort rate, coverage, abort rate + confusion."""
    n = len(scores)
    n_pos = sum(1 for y in labels if y == LABEL_FAILURE)
    n_neg = n - n_pos
    conf = confusion_at(scores, labels, threshold)
    tp, fp, tn, fn = conf["tp"], conf["fp"], conf["tn"], conf["fn"]
    recall = _metric(
        n_pos > 0,
        value=(tp / n_pos) if n_pos > 0 else None,
        reason=None if n_pos > 0 else "no failures (positive) examples",
    )
    far = _metric(
        n_neg > 0,
        value=(fp / n_neg) if n_neg > 0 else None,
        reason=None if n_neg > 0 else "no successes (negative) examples",
    )
    coverage = _metric(
        n > 0,
        value=((tn + fn) / n) if n > 0 else None,
        reason=None if n > 0 else "no records",
    )
    abort_rate = _metric(
        n > 0,
        value=((tp + fp) / n) if n > 0 else None,
        reason=None if n > 0 else "no records",
    )
    return {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "confusion": conf,
        "failure_recall": recall,
        "false_abort_rate": far,
        "coverage": coverage,
        "abort_rate": abort_rate,
    }


# --------------------------------------------------------------------------- #
# Threshold selection (calibration ID only, under a false-abort ceiling)
# --------------------------------------------------------------------------- #

def select_threshold(
    scores: List[float],
    labels: List[int],
    ceiling: float,
    eps: float = _FEASIBILITY_EPS,
) -> Dict[str, Any]:
    """Choose a threshold on calibration-ID records only.

    Abort rule: ``score >= threshold``.  Positive class = failure.

    Selection policy (lexicographic, deterministic):
      1. maximize failure recall (catch as many failures as possible);
      2. minimize the false-abort rate among maximizers (be conservative);
      3. maximize the threshold value among ties (flag the fewest episodes).

    Feasibility: ``false_abort_rate <= ceiling`` on the calibration-ID
    successes.  The "flag nothing" threshold (``+inf``) is always feasible
    (recall 0, FAR 0), so a feasible threshold always exists when the
    calibration-ID subset has both classes.

    Returns a dict with ``defined`` (False when the calibration-ID subset
    lacks a class), the numeric ``threshold`` (``+inf`` for the flag-none /
    fallback case), in-sample ``at_threshold`` statistics, and the count of
    feasible candidates.
    """
    n = len(scores)
    n_pos = sum(1 for y in labels if y == LABEL_FAILURE)
    n_neg = n - n_pos
    base = {
        "calibration_id": {"n": n, "failures": n_pos, "successes": n_neg},
        "ceiling": ceiling,
    }
    if n == 0:
        return {**base, "defined": False,
                "reason": "no calibration-ID records",
                "threshold": float("inf"), "at_threshold": None,
                "feasible_candidates": 0}
    if n_pos == 0:
        return {**base, "defined": False,
                "reason": "no failures in calibration-ID subset; "
                          "failure recall is undefined (0/0)",
                "threshold": float("inf"), "at_threshold": None,
                "feasible_candidates": 0}
    if n_neg == 0:
        return {**base, "defined": False,
                "reason": "no successes in calibration-ID subset; "
                          "false-abort rate is undefined (cannot bound ceiling)",
                "threshold": float("inf"), "at_threshold": None,
                "feasible_candidates": 0}

    distinct = sorted(set(scores), reverse=True)  # high -> low
    candidates = [float("inf")] + distinct         # +inf == flag nothing
    best_key = None
    best_tau = None
    best_stats = None
    feasible = 0
    for tau in candidates:
        tp = sum(1 for i in range(n) if labels[i] == LABEL_FAILURE and scores[i] >= tau)
        fp = sum(1 for i in range(n) if labels[i] == LABEL_SUCCESS and scores[i] >= tau)
        fn = n_pos - tp
        tn = n_neg - fp
        recall = tp / n_pos
        far = fp / n_neg
        if far <= ceiling + eps:
            feasible += 1
            # maximize recall, then minimize far (-far), then maximize tau
            key = (recall, -far, tau)
            if best_key is None or key > best_key:
                best_key = key
                best_tau = tau
                best_stats = {
                    "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                    "failure_recall": recall,
                    "false_abort_rate": far,
                    "flagged": tp + fp,
                }
    return {**base, "defined": True, "reason": None,
            "threshold": best_tau, "at_threshold": best_stats,
            "feasible_candidates": feasible}


# --------------------------------------------------------------------------- #
# ECE (reliability) -- only when calibrated probabilities are supplied
# --------------------------------------------------------------------------- #

def ece(
    probs: List[float], labels: List[int], n_bins: int = DEFAULT_ECE_BINS
) -> Dict[str, Any]:
    """Expected Calibration Error via equal-width probability bins.

    Assumes ``probs`` are *already-calibrated* estimates of P(failure)
    (supplied in the input -- no fitting is performed, so there is no
    calibration leakage).  ``labels`` use ``1`` = failure.  ECE is the
    reliability metric: sum over bins of (bin_count / n) * |accuracy -
    confidence|, where confidence = mean(prob) and accuracy = mean(label).
    """
    n = len(probs)
    if n_bins < 1:
        return _metric(False, reason="n_bins must be >= 1")
    if n == 0:
        return _metric(False, reason="no records", n=0, n_bins=n_bins)
    bin_count = [0] * n_bins
    bin_conf = [0.0] * n_bins
    bin_acc = [0.0] * n_bins
    for p, y in zip(probs, labels):
        b = min(int(p * n_bins), n_bins - 1)
        if b < 0:
            b = 0
        bin_count[b] += 1
        bin_conf[b] += p
        bin_acc[b] += y
    total = 0.0
    for b in range(n_bins):
        if bin_count[b] > 0:
            conf = bin_conf[b] / bin_count[b]
            acc = bin_acc[b] / bin_count[b]
            total += (bin_count[b] / n) * abs(acc - conf)
    return _metric(True, value=total, reason=None, n=n, n_bins=n_bins)


# --------------------------------------------------------------------------- #
# Full report
# --------------------------------------------------------------------------- #

def _metric(
    defined: bool,
    value: Optional[float] = None,
    reason: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    d: Dict[str, Any] = {"defined": defined, "value": value, "reason": reason}
    d.update(extra)
    return d


def _split_scores(
    episodes: List[Tuple[Episode, float]], domain: Optional[str] = None
) -> Tuple[List[float], List[int]]:
    scores = [s for ep, s in episodes if domain is None or ep.domain == domain]
    labels = [ep.label for ep, s in episodes if domain is None or ep.domain == domain]
    return scores, labels


def _domain_metrics(
    name: str,
    scores: List[float],
    labels: List[int],
    threshold: float,
) -> Dict[str, Any]:
    if len(scores) == 0:
        reason = f"no records in evaluation for domain {name!r}"
        return {
            "n": 0, "n_pos": 0, "n_neg": 0,
            "confusion": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
            "auroc": _metric(False, reason=reason),
            "failure_recall": _metric(False, reason=reason),
            "false_abort_rate": _metric(False, reason=reason),
            "coverage": _metric(False, reason=reason),
            "abort_rate": _metric(False, reason=reason),
        }
    a = auroc(scores, labels)
    r = rates_at(scores, labels, threshold)
    return {
        "n": r["n"], "n_pos": r["n_pos"], "n_neg": r["n_neg"],
        "confusion": r["confusion"],
        "auroc": a,
        "failure_recall": r["failure_recall"],
        "false_abort_rate": r["false_abort_rate"],
        "coverage": r["coverage"],
        "abort_rate": r["abort_rate"],
    }


def _collect_undefined(obj: Any, path: str, out: List[Dict[str, str]]) -> None:
    if isinstance(obj, dict):
        if obj.get("defined") is False:
            out.append({"path": path, "reason": obj.get("reason")})
        for k, v in obj.items():
            sub = f"{path}.{k}" if path else k
            _collect_undefined(v, sub, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _collect_undefined(v, f"{path}[{i}]", out)


def compute_report(
    episodes: List[Episode],
    source: SourceMeta,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the full deterministic metrics report dict."""
    ceiling = params["ceiling"]
    ddof = params["ddof"]
    ece_bins = params["ece_bins"]

    # Per-episode disagreement score + split.
    cal: List[Tuple[Episode, float]] = []
    eva: List[Tuple[Episode, float]] = []
    for ep in episodes:
        s = episode_disagreement(ep, ddof)
        if ep.split == "calibration":
            cal.append((ep, s))
        else:
            eva.append((ep, s))

    cal_id = [(ep, s) for ep, s in cal if ep.domain == ID_DOMAIN]
    cal_id_scores = [s for _, s in cal_id]
    cal_id_labels = [ep.label for ep, _ in cal_id]

    ts = select_threshold(cal_id_scores, cal_id_labels, ceiling)
    tau_num = ts["threshold"]            # numeric; +inf for flag-none / fallback

    # ---- evaluation (held-out) ------------------------------------------- #
    eva_scores = [s for _, s in eva]
    eva_labels = [ep.label for ep, _ in eva]
    eval_auroc = auroc(eva_scores, eva_labels)
    eval_rates = rates_at(eva_scores, eva_labels, tau_num)
    evaluation = {
        "auroc": eval_auroc,
        "failure_recall": eval_rates["failure_recall"],
        "false_abort_rate": eval_rates["false_abort_rate"],
        "coverage": eval_rates["coverage"],
        "abort_rate": eval_rates["abort_rate"],
        "confusion": eval_rates["confusion"],
        "n": eval_rates["n"],
        "n_pos": eval_rates["n_pos"],
        "n_neg": eval_rates["n_neg"],
    }

    # ---- calibration (in-sample, calibration-ID subset) ------------------- #
    cal_auroc = auroc(cal_id_scores, cal_id_labels)
    cal_rates = rates_at(cal_id_scores, cal_id_labels, tau_num)
    calibration = {
        "auroc": cal_auroc,
        "failure_recall": cal_rates["failure_recall"],
        "false_abort_rate": cal_rates["false_abort_rate"],
        "coverage": cal_rates["coverage"],
        "abort_rate": cal_rates["abort_rate"],
        "confusion": cal_rates["confusion"],
        "n": cal_rates["n"],
        "n_pos": cal_rates["n_pos"],
        "n_neg": cal_rates["n_neg"],
        "note": "in-sample statistics on the calibration-ID subset "
                "(threshold was tuned here; optimistic)",
    }

    # ---- per-domain (held-out evaluation) --------------------------------- #
    all_domains = sorted({ep.domain for ep in episodes})
    per_domain = {}
    for d in all_domains:
        d_scores = [s for ep, s in eva if ep.domain == d]
        d_labels = [ep.label for ep, s in eva if ep.domain == d]
        per_domain[d] = _domain_metrics(d, d_scores, d_labels, tau_num)

    # ---- ECE (only if calibrated probabilities supplied) ----------------- #
    prob_count = sum(1 for ep in episodes if ep.probability is not None)
    if prob_count == len(episodes) and len(episodes) > 0:
        eva_probs = [ep.probability for ep, _ in eva]
        ece_obj = ece(eva_probs, eva_labels, ece_bins)
        ece_obj["scope"] = "evaluation"
        ece_obj["probability_semantics"] = "P(failure), supplied in input"
        ece_obj["fitting"] = "none"
    elif prob_count == 0:
        ece_obj = _metric(
            False,
            reason="no calibrated probabilities in input; ECE not reported",
        )
    else:
        ece_obj = _metric(
            False,
            reason=(
                f"probability present in {prob_count}/{len(episodes)} records; "
                "ECE requires all-or-none"
            ),
        )

    metrics = {
        "evaluation": evaluation,
        "calibration": calibration,
        "per_domain": per_domain,
        "ece": ece_obj,
    }

    # ---- counts ---------------------------------------------------------- #
    def _counts(group: List[Tuple[Episode, float]]) -> Dict[str, Any]:
        n = len(group)
        n_fail = sum(1 for ep, _ in group if ep.failed)
        domains: Dict[str, Dict[str, int]] = {}
        for ep, _ in group:
            d = domains.setdefault(ep.domain, {"n": 0, "failures": 0, "successes": 0})
            d["n"] += 1
            if ep.failed:
                d["failures"] += 1
            else:
                d["successes"] += 1
        return {"n": n, "failures": n_fail, "successes": n - n_fail, "domains": domains}

    counts = {
        "total": len(episodes),
        "calibration": _counts(cal),
        "evaluation": _counts(eva),
    }

    # ---- threshold selection (output-formatted) ------------------------- #
    threshold_out = (
        None if not math.isfinite(tau_num) else tau_num
    )
    threshold_kind = "flag_none" if not math.isfinite(tau_num) else "value"
    at_threshold = ts["at_threshold"]
    threshold_selection = {
        "defined": ts["defined"],
        "reason": ts["reason"],
        "method": "maximize_failure_recall_under_id_false_abort_ceiling",
        "feasible_set": "calibration_id",
        "ceiling": ceiling,
        "calibration_id": ts["calibration_id"],
        "threshold": threshold_out,
        "threshold_kind": threshold_kind,
        "at_threshold": at_threshold,
        "feasible_candidates": ts["feasible_candidates"],
        "tiebreak": [
            "maximize_failure_recall",
            "minimize_false_abort_rate",
            "maximize_threshold",
        ],
        "abort_rule": "score >= threshold",
    }

    # ---- parameters ------------------------------------------------------ #
    parameters = {
        "false_abort_ceiling": ceiling,
        "disagreement": {
            "across_policies": "sample_variance",
            "ddof": ddof,
            "action_dims": "arithmetic_mean",
            "timesteps": "arithmetic_mean",
        },
        "threshold_selection": {
            "method": "maximize_failure_recall_under_id_false_abort_ceiling",
            "feasible_set": "calibration_id",
            "tiebreak": [
                "maximize_failure_recall",
                "minimize_false_abort_rate",
                "maximize_threshold",
            ],
            "epsilon": _FEASIBILITY_EPS,
        },
        "auroc": {
            "method": "mann_whitney_u_average_ranks_tie_corrected",
            "positive_class": "failure",
        },
        "ece": {
            "method": "reliability_equal_width_bins",
            "n_bins": ece_bins,
            "label": "failure",
            "probability_semantics": "P(failure)",
            "fitting": "none",
        },
        "abort_rule": "score >= threshold",
        "label_convention": "positive=failure(1), negative=success(0)",
    }

    report = {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": TOOL_NAME, "version": TOOL_VERSION},
        "parameters": parameters,
        "source": source.to_dict(),
        "counts": counts,
        "threshold_selection": threshold_selection,
        "metrics": metrics,
    }

    # ---- explicit undefined-metric surfacing ----------------------------- #
    undefined: List[Dict[str, str]] = []
    _collect_undefined(threshold_selection, "threshold_selection", undefined)
    _collect_undefined(metrics, "metrics", undefined)
    report["undefined"] = undefined
    return report
