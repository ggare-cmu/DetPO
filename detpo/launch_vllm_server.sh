#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server for NaturalBench evaluation.
#
# Usage:
#   bash launch_vllm_server.sh                        # defaults
#   bash launch_vllm_server.sh --model Qwen/Qwen2.5-VL-3B-Instruct
#   bash launch_vllm_server.sh --port 8001 --gpu-memory-utilization 0.7
#
# Once the server prints "Application startup complete", run:
#   python naturalbench_vllm_eval.py --num_samples 500 --output results.json

# MODEL="Qwen2.5-VL-7B-Instruct"
# MODEL="Qwen2.5-VL-72B-Instruct"
# MODEL="Qwen3-VL-8B-Instruct"
MODEL="Qwen3-VL-30B-A3B-Instruct"

PORT=8000
GPU_UTIL=0.85
MAX_MODEL_LEN=24096
TENSOR_PARALLEL=$(python3 -c "import torch; print(torch.cuda.device_count())")

# Pass any extra args straight through to vllm serve
EXTRA_ARGS=""

# Simple arg parsing for the common overrides
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)             MODEL="$2";    shift 2 ;;
        --port)              PORT="$2";     shift 2 ;;
        --gpu-memory-utilization) GPU_UTIL="$2"; shift 2 ;;
        --max-model-len)     MAX_MODEL_LEN="$2"; shift 2 ;;
        --tensor-parallel-size) TENSOR_PARALLEL="$2"; shift 2 ;;
        *)                   EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

echo "============================================"
echo " vLLM Server"
echo "  Model   : $MODEL"
echo "  Port    : $PORT"
echo "  GPU util: $GPU_UTIL"
echo "  Tensor parallel: $TENSOR_PARALLEL"
echo "============================================"

vllm serve "Qwen/$MODEL" \
    --port "$PORT" \
    --dtype float16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --trust-remote-code \
    $EXTRA_ARGS