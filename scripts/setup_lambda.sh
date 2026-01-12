#!/bin/bash
# Setup script for Lambda Stack instance
# Run this after SSH'ing into your Lambda instance

set -o pipefail

echo "=========================================="
echo "Lambda Stack Setup for MiniMax-M2.1"
echo "=========================================="

# Update system
echo "Updating system packages..."
sudo apt-get update -qq

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

# Clone repo if not present
REPO_DIR="${REPO_DIR:-$HOME/minimax-m2.1-deploy-and-test}"
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning repository..."
    git clone https://github.com/sviluppo-dev/minimax-m2.1-deploy-and-test.git "$REPO_DIR"
fi

cd "$REPO_DIR"

# Create virtual environment and install dependencies
echo "Setting up Python environment..."
uv venv
source .venv/bin/activate
uv pip install -e ".[all]"

# Install vLLM (separate due to CUDA dependencies)
echo "Installing vLLM..."
uv pip install vllm

# Create .env file template
if [ ! -f .env ]; then
    echo "Creating .env template..."
    cat > .env << 'EOF'
# vLLM Configuration
MINIMAX_VLLM_BASE_URL=http://localhost:8000/v1
MINIMAX_MODEL_NAME=MiniMaxAI/MiniMax-M1-80k

# Benchmark Configuration
MINIMAX_BENCHMARK_NUM_REQUESTS=100
MINIMAX_BENCHMARK_CONCURRENCY=10
MINIMAX_BENCHMARK_MAX_TOKENS=256

# W&B Configuration (set your values)
MINIMAX_WANDB_PROJECT=minimax-m2.1-baseline
# MINIMAX_WANDB_ENTITY=your-entity
# WANDB_API_KEY=your-api-key

# HuggingFace (if model is gated)
# HF_TOKEN=your-hf-token
EOF
    echo "Created .env file - please edit with your credentials"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env with your W&B and HF credentials"
echo "2. Start vLLM server: ./scripts/deploy_vllm.sh"
echo "3. In another terminal, run benchmark: uv run python scripts/run_benchmark.py --wandb"
echo ""
echo "Or start the API server: uv run minimax-deploy serve"
