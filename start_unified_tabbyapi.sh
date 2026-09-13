#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

CONFIG="${1:-model_runtime_configs/config_qwen36_35b_speculative.yml}"

echo "=================================================="
echo "Launching Unified Speculative TabbyAPI Engine"
echo "=================================================="
echo "Model: Qwen3.6-35B-A3B EXL3 (5.60 bpw) + Qwen2.5-Coder-1.5B Q8 Drafter"
echo "GPUs: CUDA_VISIBLE_DEVICES=0,1 (RTX 4080 Super + RTX 4000 Ada)"
echo "Config: $CONFIG"
echo "Port: 50010 (Proxied via Watchdog to 5000 & 5001)"
echo "=================================================="

export CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1

exec /home/ales/AI/.venv/bin/python main.py --config "$CONFIG"
