"""Tests for the offline ensemble-disagreement safety-gate metrics module.

These tests intentionally avoid Isaac Sim / Isaac Lab.  They cover:
  * a perfectly separable dataset (AUROC == 1, full recall, zero false-abort);
  * tie-correct AUROC (hand-checked values, including an all-tied case == 0.5);
  * zero / one-class undefined cases (evaluation one-class, zero-class domain,
    calibration-ID one-class, hard dataset-level one-class);
  * threshold ceiling behaviour (strict vs permissive ceiling);
  * split-leakage prevention (frozen calibration threshold across different
    held-out evaluation sets);
  * malformed / ragged actions, missing fields, bad enums, <2 policies,
    non-list / empty trajectories, non-numeric values, NaN/Inf;
  * deterministic output (byte-identical across runs);
  * ECE gating (with / without / mixed probabilities);
  * report structure + source-hash provenance.

Disagreement score convention used by the fixtures
--------------------------------------------------
With two policies, a single timestep and a single action dimension, the
episode score (default ``ddof=1``) is::

    score = (a - b)^2 / 2          # sample variance of two values

Fixtures put policy ``p0`` at ``[[diff]]`` and ``p1`` at ``[[0.0]]`` so the
score is ``diff**2 / 2``.  For ``diff`` in {1,2,3,4} the scores are
{0.5, 2.0, 4.5, 8.0}, all exactly representable in floating point.
"""
import json
import math
import struct

import pytest

from ensemble_gate import core


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #

def rec(eid, split, domain, failed, diff, prob=None):
    """Build a 2-policy / 1-timestep / 1-dim record with score = diff**2 / 2."""
    actions = {"p0": [[float(diff)]], "p1": [[0.0]]}
    r = {
        "episode_id": eid,
        "split": split,
        "domain": domain,
        "failed": failed,
        "actions": actions,
    }
    if prob is not None:
        r["probability"] = prob
    return r


def parse_report(stdout):
    return json.loads(stdout)


# --------------------------------------------------------------------------- #
# Unit: disagreement score
# --------------------------------------------------------------------------- #

def test_episode_disagreement_single_dim_single_step():
    ep = core.Episode("e", "calibration", "id", True,
                      {"p0": [[1.0]], "p1": [[0.0]]}, None, 1)
    # sample variance of (1, 0): ss = 0.5, denom = 1 -> 0.5
    assert core.episode_disagreement(ep, ddof=1) == pytest.approx(0.5)


def test_episode_disagreement_population_variance_ddof0():
    ep = core.Episode("e", "calibration", "id", True,
                      {"p0": [[1.0]], "p1": [[0.0]]}, None, 1)
    # population variance of (1, 0): ss = 0.5, denom = 2 -> 0.25
    assert core.episode_disagreement(ep, ddof=0) == pytest.approx(0.25)


def test_episode_disagreement_multi_dim_multi_step():
    # p0: t0=[0,0], t1=[0,0];  p1: t0=[2,4], t1=[0,0]
    actions = {
        "p0": [[0.0, 0.0], [0.0, 0.0]],
        "p1": [[2.0, 4.0], [0.0, 0.0]],
    }
    ep = core.Episode("e", "calibration", "id", True, actions, None, 1)
    # t0: dim0 var(0,2)=2, dim1 var(0,4)=8 -> mean 5; t1: 0 -> score 2.5
    assert core.episode_disagreement(ep, ddof=1) == pytest.approx(2.5)


# --------------------------------------------------------------------------- #
# Unit: tie-correct AUROC
# --------------------------------------------------------------------------- #

def test_auroc_perfectly_separable():
    # failures have the higher scores
    scores = [0.1, 0.2, 0.3, 0.4]
    labels = [0, 0, 1, 1]
    a = core.auroc(scores, labels)
    assert a["defined"] is True
    assert a["value"] == pytest.approx(1.0)


def test_auroc_known_nonseparable():
    # failures at 0.2 and 0.4; successes at 0.1 and 0.3 -> AUC = 0.75
    scores = [0.1, 0.2, 0.3, 0.4]
    labels = [0, 1, 0, 1]
    a = core.auroc(scores, labels)
    assert a["value"] == pytest.approx(0.75)


def test_auroc_tie_correct_all_equal():
    # all scores tied -> AUC == 0.5
    scores = [2.0, 2.0, 2.0, 2.0]
    labels = [0, 1, 0, 1]
    a = core.auroc(scores, labels)
    assert a["value"] == pytest.approx(0.5)


def test_auroc_tie_correct_partial():
    # scores [1,1,2,2], labels [0,1,0,1] -> AUC == 0.5 (hand-checked)
    scores = [1.0, 1.0, 2.0, 2.0]
    labels = [0, 1, 0, 1]
    a = core.auroc(scores, labels)
    assert a["value"] == pytest.approx(0.5)


def test_auroc_undefined_when_one_class():
    assert core.auroc([0.1, 0.2], [0, 0])["defined"] is False
    assert core.auroc([0.1, 0.2], [1, 1])["defined"] is False
    assert core.auroc([], [])["defined"] is False


# --------------------------------------------------------------------------- #
# Unit: threshold selection ceiling behaviour
# --------------------------------------------------------------------------- #

def test_threshold_ceiling_strict_flag_none():
    # cal-ID successes: 0.5, 8.0 ; failures: 2.0, 4.5
    scores = [0.5, 8.0, 2.0, 4.5]
    labels = [0, 0, 1, 1]
    ts = core.select_threshold(scores, labels, ceiling=0.1)
    assert ts["defined"] is True
    # under a 10% ceiling we may flag 0/2 successes -> flag none
    assert math.isinf(ts["threshold"])
    assert ts["at_threshold"]["failure_recall"] == pytest.approx(0.0)
    assert ts["at_threshold"]["false_abort_rate"] == pytest.approx(0.0)


def test_threshold_ceiling_permissive_catches_all():
    scores = [0.5, 8.0, 2.0, 4.5]
    labels = [0, 0, 1, 1]
    ts = core.select_threshold(scores, labels, ceiling=0.6)
    assert ts["defined"] is True
    assert ts["threshold"] == pytest.approx(2.0)   # flags both failures
    assert ts["at_threshold"]["failure_recall"] == pytest.approx(1.0)
    assert ts["at_threshold"]["false_abort_rate"] == pytest.approx(0.5)


def test_threshold_selection_undefined_when_no_failures_in_calid():
    ts = core.select_threshold([0.5, 0.8], [0, 0], ceiling=0.1)
    assert ts["defined"] is False
    assert math.isinf(ts["threshold"])
    assert "no failures" in ts["reason"]


def test_threshold_selection_undefined_when_no_successes_in_calid():
    ts = core.select_threshold([0.5, 0.8], [1, 1], ceiling=0.1)
    assert ts["defined"] is False
    assert "no successes" in ts["reason"]


# --------------------------------------------------------------------------- #
# CLI: end-to-end datasets
# --------------------------------------------------------------------------- #

def test_perfectly_separable_cli(run_cli, tmp_jsonl):
    records = [
        rec("c1", "calibration", "id", False, 1),   # success, score 0.5
        rec("c2", "calibration", "id", False, 1),   # success, score 0.5
        rec("c3", "calibration", "id", True, 4),    # failure, score 8.0
        rec("e1", "evaluation", "id", False, 1),
        rec("e2", "evaluation", "id", True, 4),
    ]
    path = tmp_jsonl(records)
    rc, out, err = run_cli(path)
    assert rc == 0, err
    rep = parse_report(out)
    assert rep["schema_version"] == core.SCHEMA_VERSION
    assert rep["metrics"]["evaluation"]["auroc"]["value"] == pytest.approx(1.0)
    assert rep["metrics"]["evaluation"]["failure_recall"]["value"] == pytest.approx(1.0)
    assert rep["metrics"]["evaluation"]["false_abort_rate"]["value"] == pytest.approx(0.0)
    # threshold chosen on calibration ID to catch the failure with zero false aborts
    assert rep["threshold_selection"]["threshold"] == pytest.approx(8.0)
    assert rep["threshold_selection"]["feasible_set"] == "calibration_id"
    assert rep["threshold_selection"]["calibration_id"] == {"n": 3, "failures": 1, "successes": 2}


def test_tied_scores_cli(run_cli, tmp_jsonl):
    records = [
        rec("c1", "calibration", "id", False, 2),
        rec("c2", "calibration", "id", False, 2),
        rec("c3", "calibration", "id", True, 2),
        rec("c4", "calibration", "id", True, 2),
        rec("e1", "evaluation", "id", False, 2),
        rec("e2", "evaluation", "id", True, 2),
    ]
    path = tmp_jsonl(records)
    rc, out, err = run_cli(path)
    assert rc == 0, err
    rep = parse_report(out)
    # all scores tied -> AUC 0.5, and under 10% ceiling the gate flags nothing
    assert rep["metrics"]["evaluation"]["auroc"]["value"] == pytest.approx(0.5)
    assert rep["threshold_selection"]["threshold_kind"] == "flag_none"
    assert rep["threshold_selection"]["threshold"] is None


def test_one_class_evaluation_undefined(run_cli, tmp_jsonl):
    # calibration has both classes; evaluation is all successes (one class)
    records = [
        rec("c1", "calibration", "id", False, 1),
        rec("c2", "calibration", "id", True, 4),
        rec("e1", "evaluation", "id", False, 1),
        rec("e2", "evaluation", "id", False, 1),
    ]
    path = tmp_jsonl(records)
    rc, out, err = run_cli(path)
    assert rc == 0, err
    rep = parse_report(out)
    ev = rep["metrics"]["evaluation"]
    assert ev["auroc"]["defined"] is False
    assert "no positive" in ev["auroc"]["reason"]
    assert ev["failure_recall"]["defined"] is False          # no failures
    assert ev["false_abort_rate"]["defined"] is True         # successes exist
    assert ev["false_abort_rate"]["value"] == pytest.approx(0.0)
    # surfaced in the explicit undefined list
    paths = {u["path"] for u in rep["undefined"]}
    assert "metrics.evaluation.auroc" in paths
    assert "metrics.evaluation.failure_recall" in paths


def test_zero_class_domain_undefined(run_cli, tmp_jsonl):
    # "ood_a" appears only in calibration, never in evaluation
    records = [
        rec("c1", "calibration", "id", False, 1),
        rec("c2", "calibration", "id", True, 4),
        rec("c3", "calibration", "ood_a", True, 4),
        rec("e1", "evaluation", "id", False, 1),
        rec("e2", "evaluation", "id", True, 4),
    ]
    path = tmp_jsonl(records)
    rc, out, err = run_cli(path)
    assert rc == 0, err
    rep = parse_report(out)
    ood = rep["metrics"]["per_domain"]["ood_a"]
    assert ood["n"] == 0
    assert ood["auroc"]["defined"] is False
    assert "no records in evaluation" in ood["auroc"]["reason"]
    assert rep["metrics"]["per_domain"]["id"]["n"] == 2


def test_threshold_ceiling_cli(run_cli, tmp_jsonl):
    # cal-ID: successes 0.5, 8.0 ; failures 2.0, 4.5
    records = [
        rec("c1", "calibration", "id", False, 1),   # 0.5
        rec("c2", "calibration", "id", False, 4),   # 8.0
        rec("c3", "calibration", "id", True, 2),    # 2.0
        rec("c4", "calibration", "id", True, 3),    # 4.5
        rec("e1", "evaluation", "id", False, 1),    # 0.5
        rec("e2", "evaluation", "id", True, 4),     # 8.0
    ]
    path = tmp_jsonl(records)

    # strict ceiling -> flag none -> held-out recall 0
    rc, out, err = run_cli(path, "--ceiling", "0.1")
    assert rc == 0, err
    rep = parse_report(out)
    assert rep["threshold_selection"]["threshold_kind"] == "flag_none"
    assert rep["metrics"]["evaluation"]["failure_recall"]["value"] == pytest.approx(0.0)

    # permissive ceiling -> threshold 2.0 -> held-out recall 1.0
    rc, out, err = run_cli(path, "--ceiling", "0.6")
    assert rc == 0, err
    rep = parse_report(out)
    assert rep["threshold_selection"]["threshold"] == pytest.approx(2.0)
    assert rep["metrics"]["evaluation"]["failure_recall"]["value"] == pytest.approx(1.0)


def test_split_leakage_prevention(run_cli, tmp_jsonl):
    # calibration ID: successes 0.5, 2.0 ; failure 8.0 -> optimal threshold 8.0
    cal_records = [
        rec("c1", "calibration", "id", False, 1),   # 0.5
        rec("c2", "calibration", "id", False, 2),   # 2.0
        rec("c3", "calibration", "id", True, 4),    # 8.0
    ]
    # evaluation variant A: a failure that the frozen threshold (8.0) misses
    eval_a = [rec("e1", "evaluation", "id", False, 1), rec("e2", "evaluation", "id", True, 2)]
    # evaluation variant B: a failure that the frozen threshold (8.0) catches
    eval_b = [rec("e1", "evaluation", "id", False, 1), rec("e2", "evaluation", "id", True, 4)]

    path_a = tmp_jsonl(cal_records + eval_a, name="a.jsonl")
    path_b = tmp_jsonl(cal_records + eval_b, name="b.jsonl")

    rc_a, out_a, err_a = run_cli(path_a)
    rc_b, out_b, err_b = run_cli(path_b)
    assert rc_a == 0, err_a
    assert rc_b == 0, err_b
    rep_a = parse_report(out_a)
    rep_b = parse_report(out_b)

    # The threshold must be identical and calibration-derived, regardless of
    # the held-out evaluation data (no leakage / re-tuning on evaluation).
    assert rep_a["threshold_selection"]["threshold"] == pytest.approx(8.0)
    assert rep_b["threshold_selection"]["threshold"] == pytest.approx(8.0)
    assert rep_a["threshold_selection"]["feasible_set"] == "calibration_id"
    assert rep_a["threshold_selection"]["calibration_id"] == {"n": 3, "failures": 1, "successes": 2}
    # but the held-out recall differs because the evaluation differs
    assert rep_a["metrics"]["evaluation"]["failure_recall"]["value"] == pytest.approx(0.0)
    assert rep_b["metrics"]["evaluation"]["failure_recall"]["value"] == pytest.approx(1.0)


def test_calibration_one_class_is_soft_undefined(run_cli, tmp_jsonl):
    # calibration-ID has only successes (no failures); evaluation has both
    records = [
        rec("c1", "calibration", "id", False, 1),
        rec("c2", "calibration", "id", False, 4),
        rec("e1", "evaluation", "id", False, 1),
        rec("e2", "evaluation", "id", True, 4),
    ]
    path = tmp_jsonl(records)
    rc, out, err = run_cli(path)
    assert rc == 0, err
    rep = parse_report(out)
    ts = rep["threshold_selection"]
    assert ts["defined"] is False
    assert ts["threshold_kind"] == "flag_none"
    assert ts["threshold"] is None
    assert "no failures" in ts["reason"]
    # evaluation still reports a real AUROC (it has both classes)
    assert rep["metrics"]["evaluation"]["auroc"]["defined"] is True


def test_usable_class_support_hard_error_all_failures(run_cli, tmp_jsonl):
    records = [
        rec("c1", "calibration", "id", True, 4),
        rec("c2", "calibration", "id", True, 3),
        rec("e1", "evaluation", "id", True, 4),
        rec("e2", "evaluation", "id", True, 2),
    ]
    path = tmp_jsonl(records)
    rc, out, err = run_cli(path)
    assert rc == 2
    assert "no usable class support" in err


def test_usable_class_support_hard_error_all_successes(run_cli, tmp_jsonl):
    records = [
        rec("c1", "calibration", "id", False, 1),
        rec("e1", "evaluation", "id", False, 1),
    ]
    path = tmp_jsonl(records)
    rc, out, err = run_cli(path)
    assert rc == 2
    assert "no usable class support" in err


# --------------------------------------------------------------------------- #
# CLI: malformed / ragged / NaN / Inf -> exit 2
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("record,desc", [
    ({"episode_id": "c1", "split": "calibration", "domain": "id",
      "failed": True, "actions": {"p0": [[1.0]], "p1": [[2.0, 3.0]]}},
     "ragged action dims within policy"),
    ({"episode_id": "c1", "split": "calibration", "domain": "id",
      "failed": True, "actions": {"p0": [[1.0], [2.0]], "p1": [[3.0]]}},
     "ragged trajectory length across policies"),
    ({"episode_id": "c1", "split": "calibration", "domain": "id",
      "failed": True, "actions": {"p0": [[1.0]]}},
     "fewer than two policies"),
    ({"episode_id": "c1", "split": "calibration", "domain": "id",
      "failed": True, "actions": {"p0": "x", "p1": [[1.0]]}},
     "non-list trajectory"),
    ({"episode_id": "c1", "split": "calibration", "domain": "id",
      "failed": True, "actions": {"p0": [], "p1": [[1.0]]}},
     "empty trajectory"),
    ({"episode_id": "c1", "split": "calibration", "domain": "id",
      "failed": True, "actions": {"p0": [["x"]], "p1": [[1.0]]}},
     "non-numeric action value"),
    ({"episode_id": "c1", "split": "train", "domain": "id",
      "failed": True, "actions": {"p0": [[1.0]], "p1": [[2.0]]}},
     "invalid split"),
    ({"episode_id": "c1", "split": "calibration", "domain": "id",
      "failed": "yes", "actions": {"p0": [[1.0]], "p1": [[2.0]]}},
     "non-boolean failed"),
    ({"episode_id": "c1", "split": "calibration", "domain": "",
      "failed": True, "actions": {"p0": [[1.0]], "p1": [[2.0]]}},
     "empty domain"),
    ({"split": "calibration", "domain": "id", "failed": True,
      "actions": {"p0": [[1.0]], "p1": [[2.0]]}},
     "missing episode_id"),
    ({"episode_id": "c1", "split": "calibration", "domain": "id",
      "failed": True},
     "missing actions"),
])
def test_malformed_records_exit_nonzero(run_cli, tmp_jsonl, record, desc):
    path = tmp_jsonl([record])
    rc, out, err = run_cli(path)
    assert rc == 2, f"{desc}: expected exit 2, got {rc}\nstdout={out}\nstderr={err}"


def test_non_object_and_invalid_json_lines_exit_nonzero(run_cli, tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('123\n')  # valid JSON, but not an object
    rc, out, err = run_cli(str(p))
    assert rc == 2
    assert "not an object" in err

    p2 = tmp_path / "bad2.jsonl"
    p2.write_text('{not valid json\n')
    rc, out, err = run_cli(str(p2))
    assert rc == 2
    assert "invalid JSON" in err


def test_nan_inf_actions_exit_nonzero(run_cli, tmp_path):
    for token, desc in [("NaN", "NaN"), ("Infinity", "Infinity"), ("-Infinity", "-Infinity")]:
        p = tmp_path / f"bad_{token}.jsonl"
        # write a record whose p0 action is the non-finite token
        p.write_text(
            '{"episode_id":"c1","split":"calibration","domain":"id",'
            f'"failed":true,"actions":{{"p0":[[{token}]],"p1":[[1.0]]}}}}\n'
        )
        rc, out, err = run_cli(str(p))
        assert rc == 2, f"{desc}: expected exit 2\nstdout={out}\nstderr={err}"
        assert "non-finite" in err


def test_inconsistent_policy_set_across_episodes(run_cli, tmp_jsonl):
    records = [
        {"episode_id": "c1", "split": "calibration", "domain": "id",
         "failed": True, "actions": {"p0": [[1.0]], "p1": [[2.0]]}},
        {"episode_id": "c2", "split": "calibration", "domain": "id",
         "failed": False, "actions": {"p0": [[1.0]], "p2": [[2.0]]}},
    ]
    path = tmp_jsonl(records)
    rc, out, err = run_cli(path)
    assert rc == 2
    assert "policy set" in err


def test_inconsistent_action_dim_across_episodes(run_cli, tmp_jsonl):
    records = [
        {"episode_id": "c1", "split": "calibration", "domain": "id",
         "failed": True, "actions": {"p0": [[1.0]], "p1": [[2.0]]}},
        {"episode_id": "c2", "split": "calibration", "domain": "id",
         "failed": False, "actions": {"p0": [[1.0, 2.0]], "p1": [[3.0, 4.0]]}},
    ]
    path = tmp_jsonl(records)
    rc, out, err = run_cli(path)
    assert rc == 2
    assert "action dimension" in err


def test_duplicate_episode_id(run_cli, tmp_jsonl):
    records = [
        rec("dup", "calibration", "id", True, 4),
        rec("dup", "calibration", "id", False, 1),
    ]
    path = tmp_jsonl(records)
    rc, out, err = run_cli(path)
    assert rc == 2
    assert "duplicate episode_id" in err


# --------------------------------------------------------------------------- #
# CLI: deterministic output + provenance
# --------------------------------------------------------------------------- #

def test_deterministic_output(run_cli, tmp_jsonl):
    records = [
        rec("c1", "calibration", "id", False, 1),
        rec("c2", "calibration", "id", True, 4),
        rec("e1", "evaluation", "id", False, 1),
        rec("e2", "evaluation", "id", True, 4),
        rec("e3", "evaluation", "ood_push", False, 2),
        rec("e4", "evaluation", "ood_push", True, 4),
    ]
    path = tmp_jsonl(records)
    rc1, out1, err1 = run_cli(path)
    rc2, out2, err2 = run_cli(path)
    assert rc1 == 0 and rc2 == 0
    assert out1 == out2  # byte-identical (deterministic, no timestamps)


def test_source_hash_provenance(run_cli, tmp_jsonl):
    records = [
        rec("c1", "calibration", "id", False, 1),
        rec("c2", "calibration", "id", True, 4),
        rec("e1", "evaluation", "id", False, 1),
        rec("e2", "evaluation", "id", True, 4),
    ]
    path = tmp_jsonl(records)
    rc, out, err = run_cli(path)
    assert rc == 0, err
    rep = parse_report(out)
    src = rep["source"]
    assert src["file"] == path
    assert len(src["file_sha256"]) == 64
    int(src["file_sha256"], 16)  # parses as hex
    assert src["episode_count"] == 4
    assert src["policy_set"] == ["p0", "p1"]
    assert src["action_dim"] == 1


def test_report_structure(run_cli, tmp_jsonl):
    records = [
        rec("c1", "calibration", "id", False, 1),
        rec("c2", "calibration", "id", True, 4),
        rec("c3", "calibration", "ood", False, 1),
        rec("e1", "evaluation", "id", False, 1),
        rec("e2", "evaluation", "id", True, 4),
        rec("e3", "evaluation", "ood", True, 4),
    ]
    path = tmp_jsonl(records)
    rc, out, err = run_cli(path)
    assert rc == 0, err
    rep = parse_report(out)
    for key in ("schema_version", "tool", "parameters", "source", "counts",
                "threshold_selection", "metrics", "undefined"):
        assert key in rep, f"missing top-level key {key}"
    # parameters echo the explicit aggregation choices
    d = rep["parameters"]["disagreement"]
    assert d["across_policies"] == "sample_variance"
    assert d["action_dims"] == "arithmetic_mean"
    assert d["timesteps"] == "arithmetic_mean"
    assert rep["parameters"]["auroc"]["method"].startswith("mann_whitney_u")
    # per-domain covers both domains present in evaluation
    assert set(rep["metrics"]["per_domain"]) == {"id", "ood"}
    # counts
    assert rep["counts"]["total"] == 6
    assert rep["counts"]["calibration"]["n"] == 3
    assert rep["counts"]["evaluation"]["n"] == 3


def test_compact_and_pretty_both_valid_json(run_cli, tmp_jsonl):
    records = [
        rec("c1", "calibration", "id", False, 1),
        rec("c2", "calibration", "id", True, 4),
        rec("e1", "evaluation", "id", False, 1),
        rec("e2", "evaluation", "id", True, 4),
    ]
    path = tmp_jsonl(records)
    rc1, out1, err1 = run_cli(path, "--compact")
    rc2, out2, err2 = run_cli(path, "--indent", "4")
    assert rc1 == 0 and rc2 == 0, err1 + err2
    assert "\n" not in out1.rstrip("\n")  # compact is a single line
    rep1 = parse_report(out1)
    rep2 = parse_report(out2)
    assert rep1 == rep2


def test_out_file(run_cli, tmp_jsonl, tmp_path):
    records = [
        rec("c1", "calibration", "id", False, 1),
        rec("c2", "calibration", "id", True, 4),
        rec("e1", "evaluation", "id", False, 1),
        rec("e2", "evaluation", "id", True, 4),
    ]
    path = tmp_jsonl(records)
    out_file = tmp_path / "report.json"
    rc, out, err = run_cli(path, "--out", str(out_file))
    assert rc == 0, err
    assert out == ""  # nothing on stdout
    rep = json.loads(out_file.read_text())
    assert rep["schema_version"] == core.SCHEMA_VERSION


# --------------------------------------------------------------------------- #
# CLI: ECE gating
# --------------------------------------------------------------------------- #

def test_ece_undefined_without_probabilities(run_cli, tmp_jsonl):
    records = [
        rec("c1", "calibration", "id", False, 1),
        rec("c2", "calibration", "id", True, 4),
        rec("e1", "evaluation", "id", False, 1),
        rec("e2", "evaluation", "id", True, 4),
    ]
    path = tmp_jsonl(records)
    rc, out, err = run_cli(path)
    assert rc == 0, err
    rep = parse_report(out)
    e = rep["metrics"]["ece"]
    assert e["defined"] is False
    assert "no calibrated probabilities" in e["reason"]


def test_ece_defined_with_probabilities(run_cli, tmp_jsonl):
    records = [
        rec("c1", "calibration", "id", False, 1, prob=0.1),
        rec("c2", "calibration", "id", True, 4, prob=0.9),
        rec("e1", "evaluation", "id", False, 1, prob=0.2),
        rec("e2", "evaluation", "id", True, 4, prob=0.8),
    ]
    path = tmp_jsonl(records)
    rc, out, err = run_cli(path)
    assert rc == 0, err
    rep = parse_report(out)
    e = rep["metrics"]["ece"]
    assert e["defined"] is True
    assert 0.0 <= e["value"] <= 1.0
    assert e["fitting"] == "none"


def test_ece_undefined_with_mixed_probabilities(run_cli, tmp_jsonl):
    records = [
        rec("c1", "calibration", "id", False, 1, prob=0.1),
        rec("c2", "calibration", "id", True, 4),          # no probability
        rec("e1", "evaluation", "id", False, 1, prob=0.2),
        rec("e2", "evaluation", "id", True, 4, prob=0.8),
    ]
    path = tmp_jsonl(records)
    rc, out, err = run_cli(path)
    assert rc == 0, err
    rep = parse_report(out)
    e = rep["metrics"]["ece"]
    assert e["defined"] is False
    assert "probability present in" in e["reason"]


def test_invalid_probability_is_hard_error(run_cli, tmp_jsonl):
    records = [
        rec("c1", "calibration", "id", False, 1, prob=1.5),
        rec("c2", "calibration", "id", True, 4, prob=0.9),
        rec("e1", "evaluation", "id", False, 1, prob=0.2),
        rec("e2", "evaluation", "id", True, 4, prob=0.8),
    ]
    path = tmp_jsonl(records)
    rc, out, err = run_cli(path)
    assert rc == 2
    assert "probability" in err
