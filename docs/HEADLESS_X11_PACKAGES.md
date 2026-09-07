# Headless X11 system packages (pip/container path)

> Applies to the **pip-installed** Isaac Sim 4.5.0 / Isaac Lab v2.1.0 path on a
> one-L40 container with **no GUI** (`NVIDIA_DRIVER_CAPABILITIES=compute,utility`).
> This does **not** alter host drivers — it only installs userspace shared
> libraries the pip `isaacsim` wheel dlopens at startup.

## Symptom

Isaac Sim's Python C extensions dlopen several X11/shared libs at startup even
in `--headless` mode. The observed, reproducible failure is a missing
`libSM.so.6` / `libXt.so.6` (and friends) `ImportError`, raised from
`isaaclab.app.AppLauncher._load_extensions()` while importing
`omni.physx.bindings._physx`:

```
ImportError: libSM.so.6: cannot open shared object file: No such file or directory
ImportError: libXt.so.6:  cannot open shared object file: No such file or directory
```

(`ENV_FIX.md` records the same class of failure for `libhdx.so`/`libhd.so`,
which is a *separate* pip-`python.sh` `LD_LIBRARY_PATH` issue — see that file.)

## The eight required X11 packages

These are the eight system packages Isaac Sim dlopens on the headless path.
Install once on the container (do **not** install NVIDIA drivers inside the
container — the host driver is already in use, see `README_VISUALIZATION.md`):

```bash
sudo apt-get update
sudo apt-get install -y \
    libsm6 libxrender1 libfontconfig1 libxt6 libxfixes3 libxrandr2 libxcursor1 libxi6
```

| Package | Provides | Why it's needed |
|---|---|---|
| `libsm6` | `libSM.so.6` | dlopened at AppLauncher startup (the headline symptom) |
| `libxrender1` | `libXrender.so.1` | X render for offscreen/font paths |
| `libfontconfig1` | `libfontconfig.so.1` | font discovery for HUD/annotators |
| `libxt6` | `libXt.so.6` | X toolkit intrinsics (dlopened at startup) |
| `libxfixes`/`libxfixes3` | `libXfixes.so.3` | X11 fixes extension |
| `libxrandr2` | `libXrandr.so.2` | display config (libs load even headless) |
| `libxcursor1` | `libXcursor.so.1` | cursor libs (loaded unconditionally) |
| `libxi6` | `libXi.so.6` | X input extension |

`libICE.so.6` (`libice6`) may also be required on minimal images; install it
alongside if the startup error names it:

```bash
sudo apt-get install -y libice6
```

## Notes

- These are **userspace** libraries; installing them does **not** touch the
  NVIDIA kernel driver or grant graphics/display/video capabilities. To enable
  actual Vulkan/RTX rendering you must recreate the container with
  `NVIDIA_DRIVER_CAPABILITIES=all` (see `README_VISUALIZATION.md`).
- Vulkan `VkResult: ERROR_INCOMPATIBLE_DRIVER` warnings on this compute-only
  container are **non-fatal** — PhysX still runs on CUDA.
- The `ENV_FIX.md` `python.sh` `LD_LIBRARY_PATH` patch is a **different** fix
  (pip `isaacsim` bundled `.so` discovery) and may be required in addition to
  the X11 packages above.
