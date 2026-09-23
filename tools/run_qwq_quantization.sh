#!/bin/bash
set -euo pipefail

RAW_DIR="/home/ales/AI/model_hosts/tabby_api/raw_models/QwQ-32B-FP16"
OUT_DIR="/home/ales/AI/model_hosts/tabby_api/models/QwQ-32B-exl3-4.15bpw"
WORK_DIR="/home/ales/AI/model_hosts/tabby_api/work/qwq_convert"
CAL_DATA="/home/ales/AI/model_hosts/tabby_api/tools/cal_trace.safetensors"
PYTHON_BIN="/home/ales/AI/.venv/bin/python"

mkdir -p "$WORK_DIR" "$OUT_DIR"

echo "================================================================="
echo "Starting Tri-GPU Parallel In-House Quantization: QwQ-32B (EXL3)"
echo "Target Bitrate: 4.15 bpw (lm_head: 8 bpw, mul1 codebook, --out_scales always)"
echo "Devices: 0,1,2 (RTX 4080S @ 1.6, Ada #1 @ 1.0, Ada #2 @ 1.0)"
echo "Calibration: $CAL_DATA (250 rows x 2048 cols CoT traces)"
echo "================================================================="

$PYTHON_BIN -m exllamav3.conversion.convert_model \
  -i "$RAW_DIR" \
  -o "$OUT_DIR" \
  -w "$WORK_DIR" \
  -b 4.15 \
  -hb 8 \
  -cd "$CAL_DATA" \
  -cr 250 \
  -cc 2048 \
  -cb mul1 \
  -d 0,1,2 \
  -dr 8,5,5 \
  --out_scales always

echo "================================================================="
echo "Quantization of QwQ-32B completed successfully!"
echo "Model saved to: $OUT_DIR"
echo "================================================================="
