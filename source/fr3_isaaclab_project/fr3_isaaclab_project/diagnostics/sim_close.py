"""Corrected SimulationApp close helper — does NOT mask failure exit codes.

The old ``_close_with_timeout`` pattern across the repo force-exits with
``os._exit(0)`` when ``simulation_app.close()`` hangs. That converts a real
failure (whose ``sys.exit(nonzero)`` is in flight) into a false SUCCESS — the
"hang-mask" false-pass class (F3 in the test-review report). This helper instead
preserves the caller's intended exit code on a hang, so a hung close never
reports success.
"""

from __future__ import annotations

import os
import threading


def close_with_exit_code(sim_app, exit_code: int = 0, timeout_sec: int = 15) -> None:
    """Close ``sim_app``, preserving ``exit_code`` if the close hangs.

    On a clean close the caller's ``exit_code`` is returned unchanged via the
    normal control flow (the caller still calls ``sys.exit(exit_code)``). On a
    hang, the timer fires ``os._exit(exit_code)`` so the process dies with the
    caller's intended code — never silently 0.
    """
    code = int(exit_code)

    def _force_exit() -> None:
        print(
            f"\n[WARN] simulation_app.close() hung for {timeout_sec}s; "
            f"force-exiting with code {code}.",
            flush=True,
        )
        os._exit(code)

    timer = threading.Timer(timeout_sec, _force_exit)
    timer.daemon = True
    timer.start()
    try:
        sim_app.close()
    finally:
        timer.cancel()
