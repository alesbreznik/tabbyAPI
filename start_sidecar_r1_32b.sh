#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=================================================="
echo "Launching DeepSeek-R1-Distill-32B Speculative Sidecar"
echo "Device 0: RTX 4080 Super (Ingestion Layers + 1.5B Drafter)"
echo "Device 1: RTX 4000 Ada #2 (Target 32B Saturated 20GB)"
echo "Internal Port: 50011 (Proxied to Public Port 5001)"
echo "=================================================="

export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

exec /home/ales/AI/.venv/bin/python main.py --config model_runtime_configs/config_sidecar_r1_32b.yml
