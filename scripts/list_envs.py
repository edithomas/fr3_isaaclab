"""List all environments registered under this external project.

Prints a table of Gym environment IDs that contain "FR3" in their name,
along with their entry point and config class.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
from prettytable import PrettyTable

# Trigger registration of FR3 environments
import fr3_isaaclab_project.tasks  # noqa: F401


def main():
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "FR3 Isaac Lab Project – Registered Environments"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    index = 0
    for task_spec in gym.registry.values():
        if "FR3" in task_spec.id:
            table.add_row([
                index + 1,
                task_spec.id,
                task_spec.entry_point,
                task_spec.kwargs.get("env_cfg_entry_point", ""),
            ])
            index += 1

    if index == 0:
        print("No FR3 environments found in gym registry. Did you install the project?")
    else:
        print(table)


import os
import signal
import sys
import threading


def _close_with_timeout(sim_app, timeout_sec: int = 10) -> None:
    """Close simulation_app with a hard timeout — workaround for hang in headless/Vulkan-less envs."""

    def _force_exit() -> None:
        print(f"\n[WARN] simulation_app.close() hung for {timeout_sec}s, force-exiting.", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    timer = threading.Timer(timeout_sec, _force_exit)
    timer.daemon = True
    timer.start()
    try:
        sim_app.close()
    finally:
        timer.cancel()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e
    finally:
        # Flush all output BEFORE close() which may hang in headless/Vulkan-less envs
        sys.stdout.flush()
        sys.stderr.flush()
        _close_with_timeout(simulation_app, timeout_sec=10)
