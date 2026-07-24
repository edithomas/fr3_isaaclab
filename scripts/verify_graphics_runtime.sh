#!/usr/bin/env bash
set -euo pipefail

echo "=============================================="
echo "  Isaac Sim Graphics Runtime Verification"
echo "=============================================="

echo ""
echo "--- nvidia-smi ---"
nvidia-smi 2>&1 || echo "[WARN] nvidia-smi not available"

echo ""
echo "--- NVIDIA environment variables ---"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<not set>}"
echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-<not set>}"

echo ""
echo "--- /dev/nvidia* ---"
ls -l /dev/nvidia* 2>&1 | head -30 || echo "[WARN] No /dev/nvidia* devices found"

echo ""
echo "--- vulkaninfo ---"
if command -v vulkaninfo &>/dev/null; then
    which vulkaninfo
    vulkaninfo --summary 2>&1 || echo "[WARN] vulkaninfo failed — driver capabilities may be insufficient"
else
    echo "[WARN] vulkaninfo not found on PATH"
fi

echo ""
echo "--- PyTorch CUDA check ---"
python -c "
import torch
print(f'  PyTorch:        {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version:   {torch.version.cuda}')
    print(f'  Device count:   {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
" 2>&1 || echo "[WARN] PyTorch CUDA check failed"

echo ""
echo "=============================================="
echo "  Verification complete"
echo "=============================================="
