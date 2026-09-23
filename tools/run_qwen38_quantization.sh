#!/bin/bash
set -euo pipefail

RAW_DIR="/home/ales/AI/model_hosts/tabby_api/raw_models/Qwen3.8-27B-FP16"
OUT_DIR="/home/ales/AI/model_hosts/tabby_api/models/Qwen3.8-27B-exl3-4.85bpw-vl"
WORK_DIR="/home/ales/AI/model_hosts/tabby_api/work/qwen38_convert"
CAL_DATA="/home/ales/AI/model_hosts/tabby_api/tools/cal_trace_dualmode.safetensors"
PYTHON_BIN="/home/ales/AI/.venv/bin/python"

mkdir -p "$WORK_DIR" "$OUT_DIR"

echo "================================================================="
echo "Resuming Dedicated In-House Quantization: Qwen3.8-27B (Hybrid DeltaNet + Vision)"
echo "Target Bitrate: 4.85 bpw (lm_head: 8 bpw, mtp: 8 bpw, vision: 6 bpw, mul1 codebook, --hq)"
echo "Device: Physical GPU 2 (RTX 4000 Ada #2 dedicated via CUDA_VISIBLE_DEVICES=2)"
echo "================================================================="

export CUDA_VISIBLE_DEVICES=2
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONUNBUFFERED=1

$PYTHON_BIN -m exllamav3.conversion.convert_model \
  -r \
  -w "$WORK_DIR" \
  -d 0 \
  --out_scales always

echo "================================================================="
echo "Quantization of Qwen3.8-27B completed successfully!"
echo "Model saved to: $OUT_DIR"
echo "================================================================="
