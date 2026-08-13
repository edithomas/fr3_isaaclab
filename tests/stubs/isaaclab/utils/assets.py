"""Stub `isaaclab.utils.assets` for offline exit-code tests.

Driven by environment variables so the subprocess tests can exercise every
exit-code branch of ``check_fr3_asset.py`` without Isaac Sim / Nucleus:

* ``FAKE_RETRIEVE_MODE=raise``  -> ``retrieve_file_path`` raises (asset-resolution
  failure path; expect exit 3).
* ``FAKE_RETRIEVE_MODE=ok`` (default) -> returns ``FAKE_USD_PATH`` (or a temp
  stage path); the script then ``Usd.Stage.Open``-s it (expect exit 0 or 4).
"""
from __future__ import annotations

import os

ISAAC_NUCLEUS_DIR = "https:// nucleus.example.com"


def retrieve_file_path(path: str) -> str:
    mode = os.environ.get("FAKE_RETRIEVE_MODE", "ok")
    if mode == "raise":
        raise ConnectionError(f"[stub] cannot reach Nucleus for {path}")
    fake = os.environ.get("FAKE_USD_PATH")
    if fake is None:
        raise RuntimeError("FAKE_USD_PATH not set in stub retrieve_file_path")
    return fake
