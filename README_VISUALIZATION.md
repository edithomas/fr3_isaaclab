# Isaac Sim Visualization — Container Requirements

## Problem

The current container was started with:

```
NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

This provides CUDA compute support (PhysX training works) but **prevents Isaac Sim visualization**:
- Vulkan / RTX renderer
- Isaac Sim GUI
- WebRTC livestream (`LIVESTREAM=2`)

These require graphics, display, and video capabilities that are not present.

## Solution

The container must be **restarted or rebuilt** with expanded NVIDIA driver capabilities.

### Minimum required environment variables

```
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=all
```

Or, for a narrower grant:

```
NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display,video
```

### Docker run

```bash
docker run --gpus all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  ...
```

### docker-compose

```yaml
services:
  app:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
```

### VS Code Dev Container

See `.devcontainer/devcontainer.json` in this repo — it includes the required `runArgs` with GPU and NVIDIA env vars.

## Verification

After restarting the container with the correct capabilities, run:

```bash
echo $NVIDIA_DRIVER_CAPABILITIES
bash scripts/verify_graphics_runtime.sh
```

Expected: `NVIDIA_DRIVER_CAPABILITIES=all` (or includes `graphics,display,video`), `vulkaninfo --summary` prints device info without errors, and `/dev/nvidia*` devices are present.

## Note

- A running container's `NVIDIA_DRIVER_CAPABILITIES` **cannot** be changed without a restart/recreate. There is no runtime workaround.
- Do not install NVIDIA drivers inside the container — the host driver is already in use.
