#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

REASONER_CONFIG="model_runtime_configs/config_qwen_27b.yml"
WORKER_CONFIG="model_runtime_configs/config_4080S_gemma12b.yml"

# Default to Unified Qwen 35B Speculative Engine across both GPUs
if [ -z "$1" ] || [ "$1" == "--unified" ] || [ "$1" == "unified" ]; then
    echo "Delegating to start_unified_tabbyapi.sh (Qwen3.6-35B-A3B + Drafter across GPU 0 & 1)..."
    exec ./start_unified_tabbyapi.sh "$@"
fi

# Allow selecting legacy alternate reasoning models on-demand
if [ "$1" == "--gemma26b" ] || [ "$1" == "gemma26b" ]; then
    REASONER_CONFIG="model_runtime_configs/config_ada4000_gemma26b.yml"
    echo "Selected Reasoning Model: Gemma 4 26B MoE"
elif [ "$1" == "--hermes" ] || [ "$1" == "hermes" ]; then
    WORKER_CONFIG="model_runtime_configs/config_hermes8b.yml"
    echo "Selected Worker Model: Hermes 3 8B"
else
    echo "Selected Reasoning Model: Qwen 3.6 27B Dense (Legacy)"
fi

echo "=================================================="
echo "Launching Dual TabbyAPI Stack on Linux"
echo "=================================================="
echo "GPU 0: Worker Engine (Port 5000 - RTX 4080 Super): $WORKER_CONFIG"
echo "GPU 1: Reasoning Engine (Port 5001 - RTX 4000 Ada): $REASONER_CONFIG"
echo "=================================================="

CUDA_VISIBLE_DEVICES=0 /home/ales/AI/.venv/bin/python main.py --config "$WORKER_CONFIG" &
PID_WORKER=$!

CUDA_VISIBLE_DEVICES=1 /home/ales/AI/.venv/bin/python main.py --config "$REASONER_CONFIG" &
PID_REASONER=$!

echo "Started TabbyAPI Worker (PID: $PID_WORKER) and Reasoner (PID: $PID_REASONER)"
wait
