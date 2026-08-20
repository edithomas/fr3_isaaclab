"""Command-line interface for offline ensemble-disagreement safety-gate metrics.

USAGE
    python -m ensemble_gate episodes.jsonl [options]
    python scripts/ensemble_gate_metrics.py episodes.jsonl [options]

EXIT CODES
    0  -- success (metrics reported; some may be undefined-with-reason)
    2  -- malformed input / validation error (bad JSON, missing fields,
          ragged or non-rectangular actions, NaN/Inf actions, fewer than two
          policies, no usable class support in the dataset)
    1  -- unexpected internal error

See ``docs/ensemble_gate.md`` for the data contract, formulas, examples and
the research-instrument disclaimer.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import core


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=core.TOOL_NAME,
        description=(
            "Offline ensemble-disagreement safety-gate metrics for FR3 "
            "sim-to-sim research (NOT a real-world safety guarantee)."
        ),
    )
    parser.add_argument("input", help="Path to the JSONL episode file.")
    parser.add_argument(
        "--ceiling",
        type=float,
        default=core.DEFAULT_CEILING,
        help="ID false-abort ceiling on calibration (default 0.1 = 10%%).",
    )
    parser.add_argument(
        "--ddof",
        type=int,
        default=core.DEFAULT_DDOF,
        help="Delta degrees of freedom for variance (default 1 = sample "
             "variance with denominator P-1).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=core.DEFAULT_ECE_BINS,
        help="Number of equal-width bins for ECE (default 10).",
    )
    parser.add_argument(
        "--out",
        help="Write JSON to this path instead of stdout.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent when not --compact (default 2).",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit single-line (compact) JSON.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{core.TOOL_NAME} {core.TOOL_VERSION}",
    )
    return parser


def _emit(text: str, out_path: Optional[str]) -> None:
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    else:
        sys.stdout.write(text + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    episodes, source, errors = core.load_episodes(args.input)

    if source is None:
        # File could not be read at all.
        sys.stderr.write(
            f"{core.TOOL_NAME}: cannot read input:\n"
        )
        for err in errors:
            sys.stderr.write(f"  - {err}\n")
        return 2

    if not errors and episodes:
        n_fail = sum(1 for ep in episodes if ep.failed)
        n_succ = len(episodes) - n_fail
        if n_fail == 0 or n_succ == 0:
            errors.append(
                f"no usable class support: dataset has {n_fail} failures and "
                f"{n_succ} successes (need at least one of each)"
            )

    if errors:
        sys.stderr.write(
            f"{core.TOOL_NAME}: input validation errors "
            f"({len(errors)}):\n"
        )
        for err in errors:
            sys.stderr.write(f"  - {err}\n")
        return 2

    if not episodes:
        sys.stderr.write(f"{core.TOOL_NAME}: no episodes found in input\n")
        return 2

    params = {
        "ceiling": args.ceiling,
        "ddof": args.ddof,
        "ece_bins": args.bins,
    }
    try:
        report = core.compute_report(episodes, source, params)
    except Exception as exc:  # pragma: no cover - defensive
        sys.stderr.write(f"{core.TOOL_NAME}: internal error: {exc}\n")
        return 1

    if args.compact:
        text = json.dumps(
            report, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"),
        )
    else:
        text = json.dumps(
            report, sort_keys=True, ensure_ascii=False, indent=args.indent,
        )
    _emit(text, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
