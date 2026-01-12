# minimax-m2.1-deploy-and-test

A modular, reusable library for deploying and benchmarking MiniMax-M2.1 on cloud GPUs with vLLM.

## Features

- **Inference Client**: OpenAI-compatible async/sync client for vLLM
- **Benchmarking**: Latency, throughput, and token metrics with percentile analysis
- **W&B Integration**: Log baseline metrics to Weights & Biases
- **FastAPI Service**: HTTP API for inference and benchmarking
- **Lambda Stack Ready**: Scripts for 8xA100 deployment

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv pip install -e ".[all]"

# Or pip
pip install -e ".[all]"
```

### As a Library

```python
from minimax_deploy import InferenceClient, BenchmarkRunner, WandbLogger

# Simple inference
client = InferenceClient(base_url="http://localhost:8000/v1")
result = client.generate("Explain quantum computing", max_tokens=256)
print(result.text)

# Run benchmark
runner = BenchmarkRunner(client=client)
benchmark_result = runner.run(num_requests=100, concurrency=10)
benchmark_result.print_summary()

# Log to W&B
with WandbLogger(project="my-project") as logger:
    logger.log_benchmark(benchmark_result)
```

### CLI Usage

```bash
# Check vLLM health
uv run minimax-deploy health --url http://localhost:8000/v1

# Generate text
uv run minimax-deploy generate "Write a haiku about AI"

# Run benchmark
uv run minimax-deploy benchmark --requests 100 --concurrency 10 --wandb

# Start API server
uv run minimax-deploy serve --port 8080
```

### API Endpoints

Start the server:
```bash
uv run minimax-deploy serve
```

Endpoints:
- `GET /health` - Health check
- `POST /generate` - Text generation
- `POST /benchmark` - Run benchmark (with optional W&B logging)
- `GET /benchmark/status` - Check running benchmark status
- `GET /config` - View current configuration

Example:
```bash
# Generate text
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, world!", "max_tokens": 100}'

# Run benchmark with W&B logging
curl -X POST http://localhost:8080/benchmark \
  -H "Content-Type: application/json" \
  -d '{"num_requests": 50, "concurrency": 5, "log_to_wandb": true}'
```

## Lambda Stack Deployment (8xA100)

### 1. Setup Instance

```bash
# SSH into your Lambda instance
ssh ubuntu@your-lambda-instance

# Run setup script
curl -sSL https://raw.githubusercontent.com/sviluppo-dev/minimax-m2.1-deploy-and-test/main/scripts/setup_lambda.sh | bash
```

### 2. Configure Environment

```bash
cd ~/minimax-m2.1-deploy-and-test
# Edit .env with your credentials
nano .env
```

### 3. Start vLLM Server

```bash
# Terminal 1: Start vLLM
./scripts/deploy_vllm.sh

# Or with custom settings
./scripts/deploy_vllm.sh --model MiniMaxAI/MiniMax-M1-80k --tp 8 --port 8000
```

### 4. Run Benchmark

```bash
# Terminal 2: Run benchmark
source .venv/bin/activate
uv run python scripts/run_benchmark.py --wandb
```

## Docker Deployment

```bash
# Build image
docker build -t minimax-deploy -f docker/Dockerfile .

# Run API (connects to vLLM on host)
docker run -p 8080:8080 \
  -e MINIMAX_VLLM_BASE_URL=http://host.docker.internal:8000/v1 \
  -e WANDB_API_KEY=your-key \
  minimax-deploy

# Or use docker-compose
cd docker && docker-compose up
```

## Kubernetes Deployment

```bash
# Apply manifests (edit secrets first)
kubectl apply -f docker/k8s-deployment.yaml

# Run benchmark job
kubectl create job minimax-benchmark-$(date +%s) --from=job/minimax-benchmark
```

## Configuration

All settings can be configured via environment variables (prefix: `MINIMAX_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MINIMAX_VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM server URL |
| `MINIMAX_MODEL_NAME` | `MiniMaxAI/MiniMax-M1-80k` | Model name |
| `MINIMAX_BENCHMARK_NUM_REQUESTS` | `100` | Benchmark request count |
| `MINIMAX_BENCHMARK_CONCURRENCY` | `10` | Concurrent requests |
| `MINIMAX_BENCHMARK_MAX_TOKENS` | `256` | Max tokens per request |
| `MINIMAX_WANDB_PROJECT` | `minimax-m2.1-baseline` | W&B project |
| `MINIMAX_WANDB_ENTITY` | `None` | W&B entity |

## Benchmark Metrics

The benchmark captures:

- **Latency**: mean, median, p50, p90, p95, p99, min, max, std
- **Throughput**: requests/sec, tokens/sec
- **Token counts**: prompt, completion, total (per-request and aggregate)

## Integration with Harbor

The FastAPI service can be called from harbor commands:

```bash
# From harbor, call the API
harbor run curl -X POST http://your-lambda-ip:8080/benchmark \
  -H "Content-Type: application/json" \
  -d '{"num_requests": 100, "log_to_wandb": true}'
```

## Project Structure

```
minimax-m2.1-deploy-and-test/
├── src/minimax_deploy/
│   ├── __init__.py       # Public API
│   ├── client.py         # Inference client
│   ├── benchmark.py      # Benchmarking logic
│   ├── wandb_logger.py   # W&B integration
│   ├── config.py         # Settings management
│   ├── api.py            # FastAPI service
│   └── cli.py            # CLI commands
├── scripts/
│   ├── deploy_vllm.sh    # vLLM deployment
│   ├── setup_lambda.sh   # Lambda setup
│   └── run_benchmark.py  # Benchmark runner
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── k8s-deployment.yaml
└── pyproject.toml
```

## License

MIT
