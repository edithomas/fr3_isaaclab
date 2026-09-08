"""Offline ensemble-disagreement safety-gate metrics (pure standard-library Python).

See ``core`` module docstring for the research-instrument disclaimer and
``docs/ensemble_gate.md`` for the data contract, formulas and examples.
"""
from .core import (  # noqa: F401
    DEFAULT_CEILING,
    DEFAULT_DDOF,
    DEFAULT_ECE_BINS,
    ID_DOMAIN,
    LABEL_FAILURE,
    LABEL_SUCCESS,
    SCHEMA_VERSION,
    SPLITS,
    TOOL_NAME,
    TOOL_VERSION,
    Episode,
    SourceMeta,
    auroc,
    compute_report,
    confusion_at,
    episode_disagreement,
    ece,
    load_episodes,
    rates_at,
    select_threshold,
)

__all__ = [
    "DEFAULT_CEILING",
    "DEFAULT_DDOF",
    "DEFAULT_ECE_BINS",
    "ID_DOMAIN",
    "LABEL_FAILURE",
    "LABEL_SUCCESS",
    "SCHEMA_VERSION",
    "SPLITS",
    "TOOL_NAME",
    "TOOL_VERSION",
    "Episode",
    "SourceMeta",
    "auroc",
    "compute_report",
    "confusion_at",
    "episode_disagreement",
    "ece",
    "load_episodes",
    "rates_at",
    "select_threshold",
]
