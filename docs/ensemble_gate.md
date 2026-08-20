# Ensemble-Disagreement Safety-Gate Metrics (`ensemble_gate`)

> **RESEARCH INSTRUMENT — NOT A SAFETY GUARANTEE.**
> This is an **offline sim-to-sim research instrument**. It evaluates
> ensemble-disagreement scores recorded from simulation rollouts and reports
> threshold-based gating statistics. It is **not** a real-world safety
> guarantee, it performs **no** runtime monitoring, and it is **not** a
> sim-to-real validation. A good score here does **not** mean a policy is safe
> to deploy on hardware. All deployment decisions derived from its output must
> be reviewed by a qualified engineer.

A small, **pure standard-library Python** module + CLI that consumes a
documented JSONL episode format containing episode identity, a
calibration/evaluation split, a domain (`id` or a named OOD disturbance), a
binary failure label, and aligned per-policy action trajectories. It computes
an episode disagreement score from ensemble action variance, selects a gate
threshold on the **calibration-ID subset only** under an ID false-abort
ceiling, and reports held-out **evaluation** metrics with that frozen
threshold (no re-tuning on held-out data).

The package depends on **no** Isaac Sim / Isaac Lab / NumPy / SciPy /
scikit-learn. Run it with any Python ≥ 3.9.

---

## Contents
1. [Install / run](#install--run)
2. [Data contract (JSONL)](#data-contract-jsonl)
3. [Validation rules](#validation-rules)
4. [Formulas](#formulas)
5. [Output schema](#output-schema)
6. [Example](#example)
7. [Interpretation limits](#interpretation-limits)

---

## Install / run

No installation is required — the `ensemble_gate` package is pure Python and
lives at the repository root. From the repository root:

```bash
# run via the module entry point
python -m ensemble_gate path/to/episodes.jsonl

# or via the convenience shim
python scripts/ensemble_gate_metrics.py path/to/episodes.jsonl

# options
python -m ensemble_gate episodes.jsonl --ceiling 0.1 --compact
python -m ensemble_gate episodes.jsonl --out report.json --indent 2
```

**Exit codes**
| Code | Meaning |
|------|---------|
| `0`  | Success. Metrics are reported; some may be `undefined` with a reason. |
| `2`  | Malformed input / validation error (bad JSON, missing/wrong-typed fields, ragged or non-rectangular actions, NaN/Inf actions, fewer than two policies, no usable class support in the dataset). |
| `1`  | Unexpected internal error. |

---

## Data contract (JSONL)

Each **non-blank line** is one JSON object describing a single rollout episode.
Blank lines are ignored.

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `episode_id` | string or integer | yes | Unique episode identifier. |
| `split` | string | yes | `"calibration"` or `"evaluation"`. The threshold is tuned **only** on calibration. |
| `domain` | string | yes | `"id"` for the in-distribution domain, or any non-empty name for a named OOD disturbance (e.g. `"ood_push"`). The false-abort ceiling is enforced on `domain == "id"` calibration records. |
| `failed` | boolean | yes | Binary failure label. `true` ⇒ the episode was a **failure** (the positive class). |
| `actions` | object | yes | Mapping `policy_name -> trajectory`, with **at least two policies**. A trajectory is a non-empty list of action vectors; each action vector is a non-empty list of finite numbers. All policies in an episode must share the same number of timesteps `T` and the same action dimension `D`. |
| `probability` | number in `[0,1]` | no | An *already-calibrated* estimate of `P(failure)`. When present in **all** records, ECE is reported on the evaluation split (no fitting, hence no calibration leakage). When absent, ECE is explicitly reported `undefined`. |

**Cross-episode consistency.** The set of policy names and the action
dimension `D` must be identical across all episodes (so scores are
comparable). The horizon `T` may vary per episode.

**Alignment.** "Aligned per-policy action trajectories" means: for episode
`e`, the action vectors `actions[p][t]` for different policies `p` refer to
the same timestep `t` of the same rollout (so per-timestep cross-policy
variance is meaningful).

### Minimal example line

```json
{"episode_id":"c1","split":"calibration","domain":"id","failed":true,
 "actions":{"ppo_a":[[0.8,0.1]],"ppo_b":[[0.1,0.05]]}}
```

A larger worked example is in [`examples/ensemble_gate_sample.jsonl`](../examples/ensemble_gate_sample.jsonl).

---

## Validation rules

Structural validation errors are collected and cause a **non-zero exit (code
2)**. Each error names the offending line. The CLI validates:

- Each non-blank line is a valid JSON **object**.
- Required fields present and correctly typed (`episode_id` str/int, `split`
  ∈ {calibration, evaluation}, `domain` non-empty string, `failed` boolean,
  `actions` object with **≥ 2 policies**).
- `episode_id` is unique across the file.
- Each policy trajectory is a **non-empty** list of **non-empty** numeric
  action vectors; **rectangular**: equal `T` across policies and equal `D`
  across all vectors and policies within the episode.
- All action values are finite numbers (**no NaN, +Inf, −Inf**). JSON tokens
  `NaN` / `Infinity` / `-Infinity` (accepted by Python's parser) are rejected.
- Cross-episode: identical policy-name set and identical action dimension `D`.
- `probability`, if present, is a finite number in `[0, 1]` (else a hard
  error). It must be present in **all** or **none** of the records for ECE to
  be reported.
- **Usable class support (hard gate):** the dataset must contain at least one
  failure and at least one success overall.

**Soft (reported as `undefined` with a reason, exit 0):**
- Evaluation split has a single class → evaluation **AUROC** (and any rate with
  a zero denominator) is `undefined`.
- A domain has no records in the evaluation split → per-domain metrics for that
  domain are `undefined` (`"no records in evaluation for domain '…'"`).
- The calibration-ID subset has a single class (or is empty) → threshold
  selection is `undefined` and a conservative fallback (`flag none`, threshold
  `null`) is used; held-out metrics are still reported with that fallback.

---

## Formulas

### 1. Episode disagreement score

For an episode with `P` policies, `T` timesteps, `D` action dims, with
`ddof` (default `1`, i.e. sample variance with denominator `P − 1`):

```
for each (timestep t, dim d):
    values(t,d)   = [ a_policy(t,d)  for each policy ]          # P values
    mean(t,d)     = mean( values(t,d) )
    var(t,d)      = sum_policy (a - mean)^2 / (P - ddof)        # sample variance
    per_timestep(t) = mean over d of var(t,d)                   # average over dims
score = mean over t of per_timestep(t)                          # average over time
```

Aggregation choices (also echoed in `parameters.disagreement`):
- across policies: **sample variance** (`ddof` degrees of freedom);
- across action dims: **arithmetic mean**;
- across timesteps: **arithmetic mean**.

Only the *ranking* of scores affects AUROC and threshold selection, so a
global monotone rescaling would not change the metrics; the normalization is
documented purely for reproducibility.

### 2. Tie-correct AUROC

The Mann–Whitney U statistic with **average ranks for ties** (the standard
trapezoidal / tie-correct AUC, no SciPy/sklearn):

```
ranks      = average_ranks(scores)                 # ties share the mean rank
sum_pos    = sum of ranks of positive (failure) examples
U          = sum_pos - n_pos * (n_pos + 1) / 2
AUROC      = U / (n_pos * n_neg)
```

`undefined` when `n_pos == 0` or `n_neg == 0` (single-class subset). Ties are
detected by exact float equality of the disagreement scores.

### 3. Threshold selection (calibration-ID subset only)

Abort rule: **`score >= threshold`** (positive class = failure). On the
calibration-ID records:

```
failure_recall(threshold) = #{failures: score >= threshold} / #{failures}
false_abort_rate(threshold) = #{successes: score >= threshold} / #{successes}
```

Candidates: `threshold ∈ {+∞ (flag none)} ∪ {distinct scores}`. The `+∞`
candidate (flag nothing, recall 0, FAR 0) is always feasible, so a feasible
threshold always exists when the calibration-ID subset has both classes.

**Selection (lexicographic, deterministic):**
1. maximize `failure_recall`;
2. minimize `false_abort_rate` among maximizers (be conservative);
3. maximize `threshold` among ties (flag the fewest episodes).

Feasibility: `false_abort_rate <= ceiling + 1e-12` (the `1e-12` epsilon avoids
rejecting a rate that is arithmetically equal to the ceiling but differs by
floating-point rounding). The default ceiling is `0.1` (10% ID false aborts on
calibration). **The threshold is chosen only on calibration-ID data; it is then
frozen and applied to the held-out evaluation split (no re-tuning).**

### 4. Held-out evaluation metrics (at the frozen threshold)

```
TP = failure & aborted        FP = success & aborted     (false abort)
TN = success & not aborted   FN = failure & not aborted
failure_recall = TP / (TP + FN)            # = TP / n_failures
false_abort_rate = FP / (FP + TN)          # = FP / n_successes
coverage   = (TN + FN) / n                 # fraction not aborted
abort_rate = (TP + FP) / n                 # fraction aborted
```

Per-domain metrics repeat the above for each domain present anywhere in the
dataset, evaluated on the evaluation subset of that domain (a domain absent
from evaluation yields `n = 0` with `undefined` metrics and a reason).

### 5. ECE (only when calibrated probabilities are supplied)

Reliability with equal-width probability bins over `[0, 1]` (label = failure,
probability semantics = `P(failure)`, **no fitting** — so no calibration
leakage):

```
for each record:  bin = min( int(prob * n_bins), n_bins - 1 )
for each bin b:   conf_b = mean(prob in b);  acc_b = mean(label in b)
ECE = sum_b  (count_b / n) * |acc_b - conf_b|
```

Reported `undefined` when probabilities are absent or present in only some
records.

---

## Output schema

Deterministic machine-readable JSON (sorted keys, no wall-clock timestamps;
`threshold` is `null` for the flag-none case, with `threshold_kind =
"flag_none"`). Top-level keys:

```
schema_version            # "1.0.0"
tool                      # {name, version}
parameters                # ceiling, disagreement aggregation, threshold method,
                          # auroc method, ece method, abort rule, label convention
source                    # file, file_sha256 (SHA-256 of raw bytes), byte_size,
                          # total_lines, episode_count, policy_set, action_dim
counts                    # total, calibration{...}, evaluation{...} with per-domain
threshold_selection       # defined, reason, method, feasible_set="calibration_id",
                          # ceiling, calibration_id{n,failures,successes},
                          # threshold, threshold_kind, at_threshold{...},
                          # feasible_candidates, tiebreak, abort_rule
metrics:
  evaluation              # auroc, failure_recall, false_abort_rate, coverage,
                          # abort_rate, confusion{tp,fp,tn,fn}, n, n_pos, n_neg
  calibration             # same, on the calibration-ID subset (in-sample, optimistic)
  per_domain               # {domain: {...same...}} for every domain seen
  ece                     # {defined, value, reason, ...} or undefined-with-reason
undefined                 # explicit list of {path, reason} for every undefined metric
```

Every metric object carries `defined` (bool), `value` (number or `null`), and
`reason` (`null` when defined, a human-readable string when undefined), so
undefined-ness is always explicit and machine-readable.

---

## Example

```bash
python -m ensemble_gate examples/ensemble_gate_sample.jsonl --ceiling 0.1
```

Compact, machine-readable variant:

```bash
python -m ensemble_gate examples/ensemble_gate_sample.jsonl --compact
```

The CLI prints the JSON report to stdout (or to `--out FILE`). Because the
output contains no timestamps and uses sorted keys, repeated runs on the same
input produce byte-identical output.

---

## Interpretation limits

- **Offline and sim-to-sim.** All numbers describe recorded simulation
  rollouts only. They say nothing about real-hardware behaviour and are **not**
  a sim-to-real validation.
- **Not a safety guarantee.** A high failure-recall / low false-abort / high
  coverage on this instrument does **not** certify that deploying the policy is
  safe. The gate is a research signal, not a certified interlock.
- **Threshold is calibrated, not certified.** The ceiling is chosen on the
  calibration-ID subset; generalization to unseen OOD domains is reported, not
  guaranteed. OOD numbers are held-out and must not be used to re-tune the
  threshold.
- **Depends on the ensemble.** Disagreement is only as meaningful as the
  diversity and calibration of the policies. Correlated or miscalibrated
  policies can make disagreement a weak signal.
- **Tie handling is exact.** AUROC ties are detected by exact float equality;
  near-ties (numerically close but unequal) are treated as distinct.
- **No probability refit.** ECE uses probabilities exactly as supplied in the
  input. If those probabilities are uncalibrated, ECE reflects that — it does
  not fix calibration.
- **Rectangular assumption.** All policies must share the action space and the
  same horizon within each episode; trajectories of different horizons across
  episodes are fine, but different action dimensions are rejected.

## Repository layout

```
ensemble_gate/                      pure-stdlib package (no Isaac Sim dependency)
  __init__.py
  __main__.py                       enables `python -m ensemble_gate`
  core.py                           loading, validation, scoring, metrics
  cli.py                            argparse CLI + exit codes
scripts/ensemble_gate_metrics.py    convenience shim
tests/test_ensemble_gate.py         unit + CLI tests (pytest)
tests/conftest.py                   sys.path + fixtures
examples/ensemble_gate_sample.jsonl worked example input
docs/ensemble_gate.md               this file
```
