#!/bin/bash
# Deploy vLLM server for MiniMax-M2.1 on Lambda Stack (8xA100)
# Usage: ./deploy_vllm.sh [--model MODEL] [--port PORT]

set -o pipefail

# Defaults
MODEL="${MINIMAX_MODEL_NAME:-MiniMaxAI/MiniMax-M1-80k}"
PORT="${VLLM_PORT:-8000}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-80000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --tp)
            TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --max-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "MiniMax-M2.1 vLLM Deployment"
echo "=========================================="
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Tensor Parallel: $TENSOR_PARALLEL"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "=========================================="

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Are you on a GPU instance?"
    exit 1
fi

echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Check if vLLM is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "Installing vLLM..."
    pip install vllm --quiet
fi

# Check HuggingFace token for gated models
if [ -z "$HF_TOKEN" ] && [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "WARNING: No HuggingFace token found. Set HF_TOKEN if model is gated."
fi

# Start vLLM server
echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    --enable-prefix-caching \
    --disable-log-requests
